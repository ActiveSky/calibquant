import math
import torch
import triton
import triton.language as tl
from itertools import product


DATA_TYPE = torch.float16
DATA_TYPE_TRITON = tl.float16
ACCUM_TYPE = tl.float32
DEVICE='cuda'

BITS_KV = 1

if BITS_KV==1:
    SCALE_V = 0.3
    SCALE_K = 0.5
else:
    SCALE_V = 1
    SCALE_K = 1


# pytorch implementation of quantization
def quant(x, bits, per_channel=False, scale=1, post_scale=False):
    quant_levels = 2 ** bits - 1  
    if per_channel:
        x_min = x.amin(dim=(1), keepdim=True) * scale
        x_max = x.amax(dim=(1), keepdim=True) * scale
    else:
        x_min = x.min() * scale
        x_max = x.max() * scale
        
    scale = (x_max - x_min) / quant_levels 
    zero_point = -x_min / scale  
    # quantize
    x_q = torch.round(torch.clamp(x / scale + zero_point, 0, quant_levels))
    # dequantize
    x = scale * (x_q - zero_point)
    if post_scale:
        return x_q, scale, zero_point
    else:
        return x, scale, zero_point


# triton implementation of dequantization
# @triton.jit
def get_kernel3_configs():
    block_m = (4, 8, 16)
    block_n = (16, 32, 64)
    num_warps = (4, 8, 16)
    num_stages = (2, 3)
    return [triton.Config({'BLOCK_M': m, 'BLOCK_N': n}, num_warps=wrp, num_stages=stg) for (m, n, wrp, stg) in product(block_m, block_n, num_warps, num_stages)]
    
@triton.autotune(
    configs=get_kernel3_configs(),
    key=['K', 'N'],
)
@triton.heuristics({ 'ROUNDED_K' : lambda args: triton.next_power_of_2(args['K']) })
@triton.jit
def tiled_encoded_group_vecmat_transp_kernel(
    # device tensor of matrices pointers
    a_ptr, # not encoded
    b_ptr, # encoded
    c_ptr, # output, not encoded
    lut_ptr, # value lookup table

    total_matrices, # number of matrices, this cannot be fixed because sequence length can change
    K : tl.constexpr, # reduction dimension, always a power of 2
    N : tl.constexpr, # output columns

    # encoding metadata
    BITS: tl.constexpr,
    BLOCK_M: tl.constexpr, # block on the number of martices
    BLOCK_N: tl.constexpr, # block on the output dimension 
    DATUM_PER_ELEM: tl.constexpr,
    ROUNDED_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    matrix_start = BLOCK_M * pid
    m_range = tl.arange(0, BLOCK_M)
    k_range = tl.arange(0, ROUNDED_K)
    k_range_compressed = tl.arange(0, ROUNDED_K // DATUM_PER_ELEM)
    n_range = tl.arange(0, BLOCK_N)
    m_mask = (matrix_start + m_range) < total_matrices
    k_mask = k_range < K
    K_mask_compressed = k_range_compressed < (K // DATUM_PER_ELEM)
    a_ptrs = a_ptr + (matrix_start + m_range[:, None]) * K + k_range[None, :]
    a_slice = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)[:, None, :]
    for i in tl.range(0, N, step=BLOCK_N):
        b_size = N * (K // DATUM_PER_ELEM)
        b_ptrs = b_ptr + (matrix_start + m_range[:, None, None]) * b_size + (i + n_range[None, :, None]) * (K // DATUM_PER_ELEM) + k_range_compressed[None, None, :]
        n_mask = (i + n_range) < N
        b_slice = tl.load(b_ptrs, mask=m_mask[:, None, None] & n_mask[None, :, None] & K_mask_compressed[None, None, :], other=0)
        b_slice = b_slice[:, :, :, None]
        b_slice = b_slice.broadcast_to(b_slice.shape[0], b_slice.shape[1], b_slice.shape[2], DATUM_PER_ELEM)
        shift = (BITS * (DATUM_PER_ELEM - 1 - tl.arange(0, DATUM_PER_ELEM)))[None, None, None, :]
        b_slice = (b_slice >> shift) & ((1 << BITS) - 1)
        b_slice = b_slice.reshape(b_slice.shape[0], b_slice.shape[1], b_slice.shape[2] * DATUM_PER_ELEM)
        b_slice = tl.load(lut_ptr + b_slice, mask=m_mask[:, None, None] & n_mask[None, :, None] & k_mask[None, None, :], other=0.0)
        c_slice = tl.sum(a_slice.to(tl.float32) * b_slice.to(tl.float32), axis=2)
        c_ptrs = c_ptr + (matrix_start + m_range[:, None]) * N + i + n_range[None, :]
        tl.store(c_ptrs, c_slice, mask=m_mask[:, None] &  n_mask[None, :])

def fused_kernel(unique_values : torch.Tensor, weights : torch.Tensor, features_enc : torch.Tensor, BITS=BITS_KV, DATUM_PER_ELM=int(8/BITS_KV)):
    grid = lambda META: (triton.cdiv(META['total_matrices'], META['BLOCK_M']), )
    cols = features_enc.size()[-2] 
    rows = features_enc.size()[-1] * DATUM_PER_ELM
    total = features_enc.size()[0] * features_enc.size()[1]
    output = torch.empty((weights.shape[0], weights.shape[1], 1, cols), dtype=torch.float16, device=DEVICE)
    tiled_encoded_group_vecmat_transp_kernel[grid](weights, features_enc, output, unique_values, K=rows, N=cols, total_matrices=total, BITS=BITS, DATUM_PER_ELEM=DATUM_PER_ELM)
    return output

# change this to encoded kernel in production
@torch.compile  # 使用torch.compile装饰器来优化该函数的性能
def encode(X : torch.Tensor, bits=BITS_KV, datum_per_elem=int(8/BITS_KV)):
    """
    将输入的张量X编码为更紧凑的表示形式
    
    参数:
        X (torch.Tensor): 输入的张量
        bits (int): 每个元素使用的位数，默认为BITS_KV
        datum_per_elem (int): 每个原始元素编码后的元素数量，默认为int(8/BITS_KV)
    
    返回:
        torch.uint8: 编码后的张量，数据类型为无符号8位整数
    """
    # 计算目标张量的形状，将原始张量的最后一个维度分割为两个维度
    target_shape = (X.shape[0], X.shape[1],X.shape[2],X.shape[3]//datum_per_elem,datum_per_elem)
    
    # 重塑张量以匹配目标形状
    Y = X.reshape(target_shape)
    
    # 计算每个元素需要移位的位数，用于后续的位操作
    shift = (bits*(datum_per_elem - 1 - torch.arange(0, datum_per_elem, device=X.device)))[None, None, None, None, :]
    
    # 通过移位和求和操作实现编码，并将结果转换为无符号8位整数
    return (Y << shift).sum(axis=4).to(torch.uint8)



# @triton.jit
# def reduction_or(x, y):
#     return x | y

# def get_configs_for_encode():
#     blk_size = (64, 128, 256, 512, 1024)
#     num_warps = (4, 8)
#     return [triton.Config({
#             'BLOCK_SIZE': blk
#         }, num_warps=wrp) for (blk, wrp) in product(blk_size, num_warps)]

# @triton.autotune(
#     configs=get_configs_for_encode(),
#     key=['K', 'N'],
# )

# @triton.jit
# def encode_kernel(
#     in_tensor: tl.tensor,
#     out_tensor: tl.tensor,
#     K: int,
#     N: int,
#     BITS: tl.constexpr,
#     BLOCK_SIZE: tl.constexpr, 
#     ELEM_PER_UNIT: tl.constexpr,  
# ):
#     pid = tl.program_id(axis=0)
#     output_start = pid * BLOCK_SIZE
#     input_start = pid * BLOCK_SIZE * ELEM_PER_UNIT
#     output_offsets = output_start + tl.arange(0, BLOCK_SIZE)
#     input_offsets = input_start + tl.arange(0, BLOCK_SIZE * ELEM_PER_UNIT)
#     output_mask = output_offsets < N
#     input_mask = input_offsets < K
    
#     input_tensor = tl.load(in_tensor + input_offsets, mask=input_mask)
#     input_tensor = tl.reshape(input_tensor, (BLOCK_SIZE, ELEM_PER_UNIT))
#     input_tensor = tl.cast(input_tensor, elm_type)
    
#     data = input_tensor << (BITS * (ELEM_PER_UNIT - 1 - tl.arange(0, ELEM_PER_UNIT)))
#     out = tl.reduce(data, 1, reduction_or)
#     tl.store(out_tensor + output_offsets, out, mask=output_mask)

def get_num_sm(ignored):
    return torch.cuda.get_device_properties(None).multi_processor_count

def get_kernel3_configs():
    blk_size = (2, 4, 8, 16)
    num_warps = (4, 8, 16)
    num_stages = (2, 3)
    return [triton.Config({'BLOCK_SIZE': blk}, num_warps=wrp, num_stages=stg) for (blk, wrp, stg) in product(blk_size, num_warps, num_stages)]
    
@triton.autotune(
    configs=get_kernel3_configs(),
    key=['K', 'ROUNDED_K', 'N'],
)
@triton.heuristics(values={'NUM_SM':get_num_sm, 'ROUNDED_K': lambda args: triton.next_power_of_2(args['K']) })
@triton.jit
def tiled_encoded_group_vecmat_kernel(
    # device tensor of matrices pointers
    a_ptr, # not encoded
    b_ptr, # encoded
    c_ptr, # output, not encoded
    value_ptr, # value

    # matrix configuration (M = 1)
    K : tl.constexpr, # reduction dimension
    ROUNDED_K: tl.constexpr, # K rounded to next power of 2
    N : tl.constexpr, # output columns
    total_matrices,

    # encoding metadata
    BITS: tl.constexpr,
    DATUM_PER_ELM : tl.constexpr,
    
    # how many cols in the encoded matrix do one process need to handle
    BLOCK_SIZE: tl.constexpr,
    # number of sms
    NUM_SM: tl.constexpr
):
    pid = tl.program_id(axis=0)
    N_compressed = N // DATUM_PER_ELM
    task_per_matrix = tl.cdiv(N_compressed, BLOCK_SIZE)
    total_task = total_matrices * task_per_matrix
    task_per_sm = tl.cdiv(total_task, NUM_SM)
    task_start = pid * task_per_sm
    task_end = tl.minimum((pid + 1) * task_per_sm, total_task)
    
    i = task_start
    while i < task_end:
        matrix = i // task_per_matrix
        matrix_task_end = (matrix + 1) * task_per_matrix
        block_end = tl.minimum(task_end, matrix_task_end)

        a_cursor = a_ptr + matrix * K
        b_cursor = b_ptr + matrix * K * N_compressed
        c_cursor = c_ptr + matrix * N

        row_offsets = tl.arange(0, ROUNDED_K)
        row_mask   = row_offsets < K

        # a_strip is common for the matrix
        a_stripe = tl.load(a_cursor + row_offsets[:, None], mask=row_mask[:, None], other=0)
        
        for j in tl.range(i, block_end, num_stages=4):
            task_in_matrix = j % task_per_matrix
            
            col_start = BLOCK_SIZE * task_in_matrix
            col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
            col_mask = col_offsets < N_compressed

            b_ptrs = b_cursor + row_offsets[:, None] * N_compressed + col_offsets[None, :]
            b_block = tl.load(b_ptrs, mask=row_mask[:, None], other=0)
            b_block = b_block.reshape(ROUNDED_K, BLOCK_SIZE, 1)
            b_block = b_block.broadcast_to(ROUNDED_K, BLOCK_SIZE, DATUM_PER_ELM)
            shift = (BITS*(DATUM_PER_ELM - 1 - tl.arange(0, DATUM_PER_ELM)))
            shift = shift.reshape(1, 1, DATUM_PER_ELM)
            b_block = (b_block >> shift) & ((1 << BITS) - 1)
            b_block = b_block.reshape(ROUNDED_K, BLOCK_SIZE * DATUM_PER_ELM)
            b_block = tl.load(value_ptr + b_block, mask=row_mask[:, None], other=0)
            product = a_stripe.to(tl.float32) * b_block.to(tl.float32)
            c_stripe = tl.sum(product, axis=0).to(tl.float16)

            store_start = BLOCK_SIZE * DATUM_PER_ELM * task_in_matrix 
            store_offsets = store_start + tl.arange(0, BLOCK_SIZE * DATUM_PER_ELM)
            store_mask = store_offsets < N
            tl.store(c_cursor + store_offsets, c_stripe, mask=store_mask)
        i = block_end

def fused_kernel_wv(unique_values : torch.Tensor, weights : torch.Tensor, features_enc : torch.Tensor, BITS=BITS_KV, DATUM_PER_ELM=int(8/BITS_KV)):
    grid = lambda META: (META['NUM_SM'], )
    cols = features_enc.size()[-1] * DATUM_PER_ELM
    rows = features_enc.size()[-2]
    total = math.prod(features_enc.size()[:-2])
    output = torch.empty((weights.shape[0], weights.shape[1], 1, cols), dtype=torch.float16, device=DEVICE)
    tiled_encoded_group_vecmat_kernel[grid](weights, features_enc, output, unique_values, K=rows, N=cols, total_matrices=total, BITS=BITS, DATUM_PER_ELM=DATUM_PER_ELM)
    return output