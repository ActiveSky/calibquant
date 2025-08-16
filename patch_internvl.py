import math
import os
import torch
from torch import nn
from typing import Optional, Tuple
import transformers
from transformers import GenerationConfig
from einops import rearrange

from calibquant import quant, fused_kernel, fused_kernel_wv, encode

from transformers.modeling_flash_attention_utils import _flash_attention_forward
from qjl.qjl_utils import QJLSketch, QJLKeyQuantizer, repeat_kv_quant
from qjl.matmul import cuda_quantized_bmm_dynamic
from qjl.new_pack import triton_quantize_and_pack_along_last_dim

import warnings

# global hyperparameters for quantization
BITS_KV = 1

if BITS_KV==1:
    SCALE_V = 0.3
    SCALE_K = 0.5
else:
    SCALE_V = 1
    SCALE_K = 1

LOW_BITS = True
SPROMPT_LEN = 41
CHUNK_SIZE = 1024 


@torch.no_grad()
def custom_generate(
    self,
    pixel_values: Optional[torch.FloatTensor] = None,
    input_ids: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    visual_features: Optional[torch.FloatTensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    output_hidden_states: Optional[bool] = None,
    **generate_kwargs,
)-> torch.LongTensor:
    assert self.img_context_token_id is not None
    if pixel_values is not None:
        
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            #vit_embeds = self.extract_feature(pixel_values)
            # Assuming pixel_values is already defined
            batch_size = 10
            num_samples = pixel_values.size(0)  # Total number of samples
            vit_embeds_list = []

            # Loop through the batches
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)  # Ensure the end index doesn't exceed the size
                batch = pixel_values[start_idx:end_idx]  # Slice the batch
                vit_embeds_batch = self.extract_feature(batch)  # Process the batch
                vit_embeds_list.append(vit_embeds_batch)  # Collect the results

            # Concatenate the embeddings if requiimport pdb; pdb.set_trace()red
            vit_embeds = torch.cat(vit_embeds_list, dim=0)

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        assert selected.sum() != 0
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
        
        image_token_num = int(vit_embeds.shape[0] * vit_embeds.shape[1]/B)
        os.environ['IMAGE_TOKEN_NUM'] = str(image_token_num)
        
        input_embeds = input_embeds.reshape(B, N, C)
    else:
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
    
    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        use_cache=True,
        **generate_kwargs,
    )

    return outputs


# Copied from transformers.model.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.model.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    x2.neg_() 
    return torch.cat((x2, x1), dim=-1)


# Copied from transformers.model.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # q_embed = (q * cos) + (rotate_half(q) * sin)

    q_embed = torch.empty_like(q)
    q_embed.copy_(q).mul_(cos)  
    tmp = rotate_half(q)         
    q_embed.addcmul_(tmp, sin) 
    del tmp

    k_embed = torch.empty_like(k)
    k_embed.copy_(k).mul_(cos)  
    tmp = rotate_half(k)         
    k_embed.addcmul_(tmp, sin) 
    del tmp

    # k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def forward_internlm2_calibquant(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    `forward_internlm2_calibquant` 函数是 InternLM2 模型注意力机制的前向传播函数，
    它实现了对 KV 缓存进行量化（低比特量化）以节省显存，并在推理时进行反量化计算。
    该函数旨在优化模型在生成长序列时的性能和内存占用。

    参数:
        self: 当前注意力层的实例。
        hidden_states: 输入的隐藏状态张量。
        attention_mask: 注意力掩码，用于阻止模型关注填充或未来标记。
        position_ids: 位置 ID，用于旋转位置嵌入。
        past_key_value: 过去的键值对，用于增量解码。
        output_attentions: 是否输出注意力权重。
        use_cache: 是否缓存键值对。
        **kwargs: 其他关键字参数。

    返回值:
        一个元组，包含：
        - attn_output: 注意力机制的输出。
        - attn_weights: 注意力权重（如果 output_attentions 为 True）。
        - past_key_value: 更新后的键值对缓存。
    """

    # 1. 初始化量化相关的超参数
    # 设置 SPROMPT_LEN，表示系统提示的长度。
    self.sprompt_len = SPROMPT_LEN
    # 设置 KV 值的比特数。
    self.bits_v = BITS_KV
    # 设置 V 值的缩放因子。
    self.scale_v = SCALE_V
    # 设置 V 值是否按通道量化。
    self.per_channel_v =True
    # 设置 K 值的比特数。
    self.bits_k = BITS_KV
    # 设置 K 值的缩放因子。
    self.scale_k = SCALE_K
    # 设置 K 值是否按通道量化。
    self.per_channel_k = True
    # 从环境变量中获取图像 token 的数量。
    self.image_token_num=int(os.environ.get('IMAGE_TOKEN_NUM'))
    # 根据 KV 比特数设置是否进行归一化。
    self.normalize = BITS_KV <= 2

    # 2. 处理废弃的 `padding_mask` 参数
    # 检查 `kwargs` 中是否存在 `padding_mask`。
    if 'padding_mask' in kwargs:
        # 发出警告，提示 `padding_mask` 已废弃。
        warnings.warn(
            'Passing `padding_mask` is deprecated and will be removed in v4.37. '
            'Please make sure use `attention_mask` instead.`'
        )
        # 用 `padding_mask` 覆盖 `attention_mask`。
        attention_mask = kwargs.pop('padding_mask')

    # 3. 强制设置 `output_attentions` 为 False
    # 无论输入如何，都将 `output_attentions` 设置为 False，表示不输出注意力权重。
    output_attentions = False

    # 4. 获取输入张量的维度
    # 获取批次大小 (bsz)、查询序列长度 (q_len) 和隐藏状态维度。
    bsz, q_len, _ = hidden_states.size()

    # 5. 计算 QKV 状态
    # 通过线性层 `self.wqkv` 将隐藏状态转换为 QKV 状态。
    qkv_states = self.wqkv(hidden_states)

    # 6. 重排 QKV 状态
    # 使用 `einops.rearrange` 将 QKV 状态重排为 `(b, q, h, gs, d)` 形状，
    # 其中 `gs` 是 2 + `num_key_value_groups`，用于分离 Q、K、V。
    qkv_states = rearrange(
        qkv_states,
        'b q (h gs d) -> b q h gs d',
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )

    # 7. 分离 Query, Key, Value 状态
    # 提取 Query 状态。
    query_states = qkv_states[..., : self.num_key_value_groups, :]
    # 重排 Query 状态以合并 `h` 和 `gs` 维度。
    query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
    # 提取 Key 状态。
    key_states = qkv_states[..., -2, :]
    # 提取 Value 状态。
    value_states = qkv_states[..., -1, :]

    # 8. 转置 Query, Key, Value 状态
    # 将 Query 状态的维度从 `(b, q, h, d)` 转置为 `(b, h, q, d)`。
    query_states = query_states.transpose(1, 2)
    # 将 Key 状态的维度从 `(b, q, h, d)` 转置为 `(b, h, q, d)`。
    key_states = key_states.transpose(1, 2)
    # 将 Value 状态的维度从 `(b, q, h, d)` 转置为 `(b, h, q, d)`。
    value_states = value_states.transpose(1, 2)

    # 9. 计算 KV 序列长度
    # 初始化 KV 序列长度为当前 Key 状态的序列长度。
    kv_seq_len = key_states.shape[-2]
    # 如果存在 `past_key_value` (即缓存)，则更新 KV 序列长度。
    if past_key_value is not None:
        # 如果是低比特模式，需要考虑图像 token 的数量。
        if LOW_BITS:
            kv_seq_len += (past_key_value[0].shape[-2]+int(os.environ.get('IMAGE_TOKEN_NUM')))
        # 否则，只加上缓存的 Key 序列长度。
        else:
            kv_seq_len += past_key_value[0].shape[-2]        

    # 10. 应用旋转位置嵌入 (RoPE)
    # 计算旋转嵌入的 cos 和 sin 值。
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    # 将旋转位置嵌入应用到 Query 和 Key 状态。
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # 11. 更新 `past_key_value` 缓存
    # 如果存在 `past_key_value`，则将当前 Key 和 Value 状态与缓存拼接。
    if past_key_value is not None:
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    # 如果是低比特模式且 Query 序列长度大于 1，则对 Key 和 Value 状态进行特殊处理。
    if LOW_BITS and query_states.shape[2]>1:
        # 将 Key 和 Value 状态分割成系统提示部分和非图像 token 部分，用于缓存。
        key_states_st = torch.cat([key_states[:,:,:self.sprompt_len,:],key_states[:,:,self.sprompt_len+self.image_token_num:,:]],dim=2)
        value_states_st = torch.cat([value_states[:,:,:self.sprompt_len,:],value_states[:,:,self.sprompt_len+self.image_token_num:,:]],dim=2)
        # 更新 `past_key_value` 缓存。
        past_key_value = (key_states_st, value_states_st) if use_cache else None
    # 否则，直接更新 `past_key_value` 缓存。
    else:
        past_key_value = (key_states, value_states) if use_cache else None

    # 12. 再次转置 Query, Key, Value 状态
    # 将 Query 状态的维度从 `(b, h, q, d)` 转置回 `(b, q, h, d)`。
    query_states = query_states.transpose(1, 2)
    # 将 Key 状态的维度从 `(b, h, q, d)` 转置回 `(b, q, h, d)`。
    key_states = key_states.transpose(1, 2)
    # 将 Value 状态的维度从 `(b, h, q, d)` 转置回 `(b, q, h, d)`。
    value_states = value_states.transpose(1, 2)


    # 13. 低比特量化处理 (仅当 `LOW_BITS` 为 True 且 `query_states.shape[1] > 1` 时)
    if LOW_BITS and query_states.shape[1] > 1: 
        # 提取图像相关的 Value 状态。
        value_states_v = value_states[:,self.sprompt_len:self.sprompt_len+self.image_token_num,:,:]
        # 对 Value 状态进行量化，并获取反量化后的值、缩放因子和零点。
        value_states_deq, self.scale_value, self.zero_point_value = quant(value_states_v, bits=self.bits_v, per_channel=self.per_channel_v,scale = self.scale_v, post_scale=True)

        # 提取图像相关的 Key 状态。
        key_states_v = key_states[:,self.sprompt_len:self.sprompt_len+self.image_token_num,:,:]
        # 对 Key 状态进行量化，并获取反量化后的值、缩放因子和零点。
        key_states_deq, self.scale_key, self.zero_point_key = quant(key_states_v, bits=self.bits_k, per_channel=self.per_channel_k, scale = self.scale_k, post_scale=True)

        # 转置并重复 Key 的缩放因子和零点，以匹配 Query 的维度。
        self.scale_key = self.scale_key.transpose(1, 2)
        self.scale_key = repeat_kv(self.scale_key, self.num_key_value_groups)
        self.zero_point_key = self.zero_point_key.transpose(1, 2)
        self.zero_point_key = repeat_kv(self.zero_point_key, self.num_key_value_groups)
        
        # 转置并重复 Value 的缩放因子和零点，以匹配 Query 的维度。
        self.scale_value = self.scale_value.transpose(1, 2)
        self.scale_value = repeat_kv(self.scale_value, self.num_key_value_groups)
        self.zero_point_value = self.zero_point_value.transpose(1, 2)
        self.zero_point_value = repeat_kv(self.zero_point_value, self.num_key_value_groups)
        
        # 转置反量化后的 Key 和 Value 状态。
        key_states_deq = key_states_deq.transpose(1, 2)
        value_states_deq = value_states_deq.transpose(1, 2)
        # 清空 CUDA 缓存。
        torch.cuda.empty_cache() 
        
        # 提取 Key 状态的唯一值和对应的逆索引，用于编码。
        self.unique_values_k = torch.unique(key_states_deq.flatten().float()[:10000], sorted=False).to(key_states_deq.dtype)
        unique_values = self.unique_values_k.clone().view(-1, *([1] * key_states_deq.dim()))
        mask = (key_states_deq == unique_values) 
        _, inverse_indices_k = mask.max(dim=0)
        self.unique_values_k = unique_values.view(-1)
    
        # 提取 Value 状态的唯一值和对应的逆索引，用于编码。
        self.unique_values_v = torch.unique(value_states_deq.flatten().float()[:10000], sorted=False).to(value_states_deq.dtype)
        unique_values = self.unique_values_v.clone().view(-1, *([1] * value_states_deq.dim()))
        mask = (value_states_deq == unique_values) 
        _, inverse_indices_v = mask.max(dim=0)
        self.unique_values_v = unique_values.view(-1)

        # 对 Key 的逆索引进行编码。
        self.encoded_features_k = encode(inverse_indices_k)
        # 清空 CUDA 缓存。
        torch.cuda.empty_cache() 
        
        # 将 Value 的唯一值转换为半精度浮点数。
        self.unique_values_v = self.unique_values_v.half()
        # 对 Value 的逆索引进行编码。
        self.encoded_features_v = encode(inverse_indices_v)
        # 清空 CUDA 缓存。
        torch.cuda.empty_cache() 
        
    # 14. 计算注意力权重和输出 (当 `query_states.shape[1] == 1` 时，即单 token 生成)
    if query_states.shape[1] == 1:
        # 清空 CUDA 缓存。
        torch.cuda.empty_cache() 
        # 转置 Query, Key, Value 状态。
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        # 重复 Key 和 Value 状态以匹配 Query 的头数。
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # 如果是低比特模式，则进行量化注意力计算。
        if LOW_BITS:
            # 分割 Key 状态为系统提示部分和非图像 token 部分。
            key_states_quant_s = key_states[:,:,:self.sprompt_len,:]
            key_states_quant_t = key_states[:,:,self.sprompt_len:,:]
            # 重复编码后的 Key 和 Value 特征。
            encoded_features_k = repeat_kv(self.encoded_features_k, self.num_key_value_groups)
            encoded_features_v = repeat_kv(self.encoded_features_v, self.num_key_value_groups)
            
            # 计算系统提示部分和非图像 token 部分的注意力权重。
            attn_weights_quant_s = torch.matmul(query_states, key_states_quant_s.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_weights_quant_t = torch.matmul(query_states, key_states_quant_t.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            # 计算 Key 的偏置项。
            bias_k = self.scale_key * self.zero_point_key * query_states
            bias_k = torch.sum(bias_k,dim=-1,keepdim=True)/math.sqrt(self.head_dim)

            # 使用 `fused_kernel` 计算图像部分的注意力权重，并减去偏置。
            attn_weights_quant_v = fused_kernel(self.unique_values_k, query_states * self.scale_key, encoded_features_k) / math.sqrt(self.head_dim)  - bias_k
            
            # 如果需要归一化，则对图像部分的注意力权重进行归一化。
            if self.normalize:
                current_max = attn_weights_quant_v.amax(dim=(2, 3), keepdim=True)
                current_min = attn_weights_quant_v.amin(dim=(2, 3), keepdim=True)
                norm_offset_max = -3
                norm_offset_min = 0
                target_max = current_max + norm_offset_max
                target_min = current_min + norm_offset_min
                normalized_weights = (attn_weights_quant_v - current_min) / (current_max - current_min + 1e-8)
                normalized_weights = normalized_weights * (target_max - target_min) + target_min
                attn_weights_quant_v = normalized_weights

            # 拼接所有部分的注意力权重。
            attn_weights_quant = torch.cat([attn_weights_quant_s, attn_weights_quant_v, attn_weights_quant_t],dim=-1)
        # 否则，进行标准注意力计算。
        else:
            attn_weights_quant = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 将计算出的注意力权重赋值给 `attn_weights`。
        attn_weights = attn_weights_quant
        
        # 检查注意力权重的尺寸是否正确。
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f'Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is'
                f' {attn_weights.size()}'
            )

        # 如果存在注意力掩码，则将其添加到注意力权重中。
        if attention_mask is not None:
            # 检查注意力掩码的尺寸是否正确。
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f'Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}'
                )
            attn_weights = attn_weights + attention_mask

        # 将注意力权重上采样到 fp32，然后应用 softmax，再转换回原始数据类型。
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
        # 如果是低比特模式，则进行量化注意力输出计算。
        if LOW_BITS:
            # 使用 `fused_kernel_wv` 计算图像部分的注意力输出。
            attn_output_v = fused_kernel_wv(self.unique_values_v.half(), attn_weights[:,:,:,self.sprompt_len:self.sprompt_len+self.image_token_num].half(), encoded_features_v) 
            # 乘以缩放因子。
            attn_output_v = attn_output_v *  self.scale_value
            # 计算 Value 的偏置项。
            bias_v = self.scale_value * self.zero_point_value * torch.sum(attn_weights[:,:,:,self.sprompt_len:self.sprompt_len+self.image_token_num],dim=-1,keepdim=True)
            
            # 减去偏置。
            attn_output_v = attn_output_v - bias_v 
            
            # 分割 Value 状态为系统提示部分和非图像 token 部分。
            value_states_s = value_states[:,:,:self.sprompt_len,:]
            value_states_t = value_states[:,:,self.sprompt_len:,:]
            # 计算系统提示部分和非图像 token 部分的注意力输出。
            attn_output_s = torch.matmul(attn_weights[:,:,:,:self.sprompt_len], value_states_s)
            attn_output_t = torch.matmul(attn_weights[:,:,:,self.sprompt_len+self.image_token_num:], value_states_t)
            # 拼接所有部分的注意力输出。
            attn_output = attn_output_v+attn_output_s+attn_output_t
            # 将注意力输出转换为 bfloat16 精度。
            attn_output = attn_output.to(torch.bfloat16)
        # 否则，进行标准注意力输出计算。
        else:
            attn_output = torch.matmul(attn_weights, value_states)

        # 检查注意力输出的尺寸是否正确。
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is'
                f' {attn_output.size()}'
            )
        # 转置并连续化注意力输出。
        attn_output = attn_output.transpose(1, 2).contiguous()
        # 重塑注意力输出。
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # 通过输出线性层 `self.wo` 处理注意力输出。
        attn_output = self.wo(attn_output)

        # 如果不输出注意力权重，则将其设置为 None。
        if not output_attentions:
            attn_weights = None
    # 15. 计算注意力权重和输出 (当 `query_states.shape[1] > 1` 时，即多 token 生成)
    else:
        # 使用 `_flash_attention_forward` 分块计算注意力输出。
        attn_output = []
        seq_length = query_states.shape[1]
        
        for i in range(0, seq_length, CHUNK_SIZE):
            attn_output_chunk = self._flash_attention_forward(
            query_states[:,i:i+CHUNK_SIZE], key_states[:,:i+CHUNK_SIZE], value_states[:,:i+CHUNK_SIZE], attention_mask, query_states[:,:i+CHUNK_SIZE].shape[1]
            )   
            attn_output.append(attn_output_chunk)
        # 删除不再需要的张量以释放内存。
        del query_states, key_states, value_states
        # 拼接所有分块的注意力输出。
        attn_output = torch.cat(attn_output,dim=1)
        # 重塑注意力输出。
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        # 通过输出线性层 `self.wo` 处理注意力输出。
        attn_output = self.wo(attn_output)

        # 如果不输出注意力权重，则将其设置为 None。
        if not output_attentions:
            attn_weights = None

    # 16. 返回结果
    # 返回注意力输出、注意力权重和更新后的键值对缓存。
    return attn_output, attn_weights, past_key_value


class _dummy_class_for_accessing_shape:
    def __init__(self, kv_seq_len):
        self.shape = (-1,-1,kv_seq_len,-1)


def forward_internlm2_qjl(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    self.sprompt_len = SPROMPT_LEN
    self.image_token_num=int(os.environ.get('IMAGE_TOKEN_NUM'))

    if 'padding_mask' in kwargs:
        warnings.warn(
            'Passing `padding_mask` is deprecated and will be removed in v4.37. '
            'Please make sure use `attention_mask` instead.`'
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop('padding_mask')

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    qkv_states = self.wqkv(hidden_states)

    qkv_states = rearrange(
        qkv_states,
        'b q (h gs d) -> b q h gs d',
        gs=2 + self.num_key_value_groups,
        d=self.head_dim,
    )

    query_states = qkv_states[..., : self.num_key_value_groups, :]
    query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
    key_states = qkv_states[..., -2, :]
    value_states = qkv_states[..., -1, :]

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]        

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        #  = past_key_value[0].shape[0]
        kv_quant = past_key_value[1]
        key_states_sys = past_key_value[2]
        key_states_txt = past_key_value[3]
        value_states_quant = past_key_value[4]
        value_scale = past_key_value[5]
        value_mn = past_key_value[6]
        value_states_sys = past_key_value[7]
        value_states_txt = past_key_value[8]

        # original: [system, visual, text + new]
        key_states_txt = torch.cat([key_states_txt, key_states], dim=2)
        att_qk_sys = query_states @ repeat_kv(key_states_sys, self.num_key_value_groups).transpose(2, 3)
        att_qk_txt = query_states @ repeat_kv(key_states_txt, self.num_key_value_groups).transpose(2, 3)
        att_qk = torch.cat((att_qk_sys, kv_quant.attention_score(query_states), att_qk_txt), dim=-1)
        attn_weights = att_qk / self.head_dim ** 0.5

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        value_states_txt = torch.cat([value_states_txt, value_states], dim=2)
        value_full_length = value_states_txt.shape[-2]
        if value_states_quant is None:
            raise NotImplementedError("value_states_quant is None")
        else:
            # visual part (middle)
            value_states_quant_repeat = repeat_kv_quant(value_states_quant, self.num_key_value_groups)
            value_scale_repeat = repeat_kv_quant(value_scale, self.num_key_value_groups)
            value_mn_repeat = repeat_kv_quant(value_mn, self.num_key_value_groups)
            attn_output = cuda_quantized_bmm_dynamic(
                self.config.group_size,
                attn_weights[:, :, :, self.sprompt_len:self.sprompt_len+self.image_token_num],
                value_states_quant_repeat, value_scale_repeat, value_mn_repeat, self.config.v_bits)
            # system_prompt & text parts (first and last)
            value_states_sys_repeat = repeat_kv(value_states_sys, self.num_key_value_groups)
            value_states_txt_repeat = repeat_kv(value_states_txt, self.num_key_value_groups)
            attn_output += attn_weights[..., self.sprompt_len+self.image_token_num:] @ value_states_txt_repeat \
                + attn_weights[..., :self.sprompt_len] @ value_states_sys_repeat

        attn_output = attn_output.transpose(1, 2).contiguous()

        past_key_value = (
            _dummy_class_for_accessing_shape(kv_seq_len),
            kv_quant, key_states_sys, key_states_txt,
            value_states_quant, value_scale, value_mn, value_states_sys, value_states_txt,
            ) if use_cache else None
    else:
        input_dtype = query_states.dtype
        key_states_repeat = repeat_kv(key_states, self.num_key_value_groups)
        value_states_repeat = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = _flash_attention_forward(
            query_states.transpose(1, 2), key_states_repeat.transpose(1, 2),
            value_states_repeat.transpose(1, 2), None, q_len, dropout=0.0, is_causal=self.is_causal,
        )

        split_kv = lambda x: x.split([self.sprompt_len, self.image_token_num, q_len-self.sprompt_len-self.image_token_num], dim=-2)

        kv_quant = QJLKeyQuantizer(self.config.qjl, self.config.outlier_count_general, self.config.buffer_size, self.config.group_size, self.config.k_bits)
        key_states_sys, key_states_v, key_states_txt = split_kv(key_states)
        kv_quant.build_sketch(key_states_v)
        
        value_states_sys, value_states_v, value_states_txt = split_kv(value_states)
        if value_states.shape[-2] <= self.config.buffer_size:
            raise NotImplementedError("value_states.shape[-2] <= self.config.buffer_size")
        else:
            value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(
                value_states_v.contiguous(), self.config.group_size, self.config.v_bits)

        past_key_value = (
            _dummy_class_for_accessing_shape(kv_seq_len),
            kv_quant, key_states_sys, key_states_txt,
            value_states_quant, value_scale, value_mn, value_states_sys, value_states_txt,
            ) if use_cache else None

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.wo(attn_output)

    if not output_attentions:
        attn_weights = None
    
    return attn_output, attn_weights, past_key_value


ATTN_FORWARD_FUNCTIONS = {
    "calibquant": forward_internlm2_calibquant, 
    "qjl": forward_internlm2_qjl,
}


def patch_model(model, quant_method='calibquant', **kwargs):
    print("Patching InternVLChatModel.generate function ...")
    model.__class__.generate = custom_generate

    print(f"Patching InternLM2FlashAttention2.forward function ... {quant_method}")
    model.language_model.model.layers[0].attention.__class__.forward = ATTN_FORWARD_FUNCTIONS[quant_method]

    if quant_method == 'qjl':
        model.config.llm_config.k_bits = k_bits = kwargs.get('k_bits', 2)
        model.config.llm_config.v_bits = kwargs.get('v_bits', 2)
        model.config.llm_config.group_size = kwargs.get('group_size', 32)
        model.config.llm_config.buffer_size = kwargs.get('buffer_size', 128)
        model.config.llm_config.outlier_count_general = kwargs.get('outlier_count_general', 32)
        seed = kwargs.get('seed', 1234)
        generator = torch.Generator(device='cuda').manual_seed(seed)
        model.config.llm_config.qjl = QJLSketch(dim=(128, k_bits * 128), dim_outlier=256, rot=True, rng=generator)
