import math
import os
import torch
from torch import nn
from typing import Optional, Tuple
import transformers
from transformers import GenerationConfig
from einops import rearrange

from calibquant import quant, fused_kernel, fused_kernel_wv, encode

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


def forward_internlm2flashattn2(
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
    self.bits_v = BITS_KV
    self.scale_v = SCALE_V
    self.per_channel_v =True
    self.bits_k = BITS_KV
    self.scale_k = SCALE_K
    self.per_channel_k = True
    self.image_token_num=int(os.environ.get('IMAGE_TOKEN_NUM'))
    self.normalize = BITS_KV <= 2

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
        if LOW_BITS:
            kv_seq_len += (past_key_value[0].shape[-2]+int(os.environ.get('IMAGE_TOKEN_NUM')))
        else:
            kv_seq_len += past_key_value[0].shape[-2]        

    try:
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    except:
        import pdb;pdb.set_trace();

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    if LOW_BITS and query_states.shape[2]>1:
        key_states_st = torch.cat([key_states[:,:,:self.sprompt_len,:],key_states[:,:,self.sprompt_len+self.image_token_num:,:]],dim=2)
        value_states_st = torch.cat([value_states[:,:,:self.sprompt_len,:],value_states[:,:,self.sprompt_len+self.image_token_num:,:]],dim=2)
        past_key_value = (key_states_st, value_states_st) if use_cache else None
    else:
        past_key_value = (key_states, value_states) if use_cache else None

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)


    if LOW_BITS and query_states.shape[1] > 1: 
        value_states_v = value_states[:,self.sprompt_len:self.sprompt_len+self.image_token_num,:,:]
        value_states_deq, self.scale_value, self.zero_point_value = quant(value_states_v, bits=self.bits_v, per_channel=self.per_channel_v,scale = self.scale_v, post_scale=True)

        key_states_v = key_states[:,self.sprompt_len:self.sprompt_len+self.image_token_num,:,:]
        key_states_deq, self.scale_key, self.zero_point_key = quant(key_states_v, bits=self.bits_k, per_channel=self.per_channel_k, scale = self.scale_k, post_scale=True)

        self.scale_key = self.scale_key.transpose(1, 2)
        self.scale_key = repeat_kv(self.scale_key, self.num_key_value_groups)
        self.zero_point_key = self.zero_point_key.transpose(1, 2)
        self.zero_point_key = repeat_kv(self.zero_point_key, self.num_key_value_groups)
        
        self.scale_value = self.scale_value.transpose(1, 2)
        self.scale_value = repeat_kv(self.scale_value, self.num_key_value_groups)
        self.zero_point_value = self.zero_point_value.transpose(1, 2)
        self.zero_point_value = repeat_kv(self.zero_point_value, self.num_key_value_groups)
        
        key_states_deq = key_states_deq.transpose(1, 2)
        value_states_deq = value_states_deq.transpose(1, 2)
        torch.cuda.empty_cache() 
        
        self.unique_values_k = torch.unique(key_states_deq.flatten().float()[:10000], sorted=False).to(key_states_deq.dtype)
        unique_values = self.unique_values_k.clone().view(-1, *([1] * key_states_deq.dim()))
        # unique_values = torch.tensor(self.unique_values_k, device=key_states_deq.device).view(-1, *([1] * key_states_deq.dim()))
        mask = (key_states_deq == unique_values) 
    
        _, inverse_indices_k = mask.max(dim=0)
        self.unique_values_k = unique_values.view(-1)
    
        self.unique_values_v = torch.unique(value_states_deq.flatten().float()[:10000], sorted=False).to(value_states_deq.dtype)
        # unique_values = torch.tensor(self.unique_values_v, device=value_states_deq.device).view(-1, *([1] * value_states_deq.dim()))
        unique_values = self.unique_values_v.clone().view(-1, *([1] * value_states_deq.dim()))
        mask = (value_states_deq == unique_values) 
        
        _, inverse_indices_v = mask.max(dim=0)
        self.unique_values_v = unique_values.view(-1)

        self.encoded_features_k = encode(inverse_indices_k)
        torch.cuda.empty_cache() 
        
        self.unique_values_v = self.unique_values_v.half()
        self.encoded_features_v = encode(inverse_indices_v)
        torch.cuda.empty_cache() 
        
    if query_states.shape[1] == 1:
        torch.cuda.empty_cache() 
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        if LOW_BITS:
            key_states_quant_s = key_states[:,:,:self.sprompt_len,:]
            key_states_quant_t = key_states[:,:,self.sprompt_len:,:]
            encoded_features_k = repeat_kv(self.encoded_features_k, self.num_key_value_groups)
            encoded_features_v = repeat_kv(self.encoded_features_v, self.num_key_value_groups)
            
            attn_weights_quant_s = torch.matmul(query_states, key_states_quant_s.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_weights_quant_t = torch.matmul(query_states, key_states_quant_t.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            bias_k = self.scale_key * self.zero_point_key * query_states
            bias_k = torch.sum(bias_k,dim=-1,keepdim=True)/math.sqrt(self.head_dim)

            attn_weights_quant_v = fused_kernel(self.unique_values_k, query_states * self.scale_key, encoded_features_k) / math.sqrt(self.head_dim)  - bias_k
            
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

            attn_weights_quant = torch.cat([attn_weights_quant_s, attn_weights_quant_v, attn_weights_quant_t],dim=-1)
        else:
            attn_weights_quant = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        attn_weights = attn_weights_quant
        
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f'Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is'
                f' {attn_weights.size()}'
            )


        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f'Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}'
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
        if LOW_BITS:
            attn_output_v = fused_kernel_wv(self.unique_values_v.half(), attn_weights[:,:,:,self.sprompt_len:self.sprompt_len+self.image_token_num].half(), encoded_features_v) 
            attn_output_v = attn_output_v *  self.scale_value
            bias_v = self.scale_value * self.zero_point_value * torch.sum(attn_weights[:,:,:,self.sprompt_len:self.sprompt_len+self.image_token_num],dim=-1,keepdim=True)
            
            attn_output_v = attn_output_v - bias_v 
            
            value_states_s = value_states[:,:,:self.sprompt_len,:]
            value_states_t = value_states[:,:,self.sprompt_len:,:]
            attn_output_s = torch.matmul(attn_weights[:,:,:,:self.sprompt_len], value_states_s)
            attn_output_t = torch.matmul(attn_weights[:,:,:,self.sprompt_len+self.image_token_num:], value_states_t)
            attn_output = attn_output_v+attn_output_s+attn_output_t
            attn_output = attn_output.to(torch.bfloat16)
        else:
            attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is'
                f' {attn_output.size()}'
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.wo(attn_output)

        if not output_attentions:
            attn_weights = None
    else:
        # attn_output = self._flash_attention_forward(
        #     query_states, key_states, value_states, attention_mask, q_len
        # )
        attn_output = []
        seq_length = query_states.shape[1]
        
        for i in range(0, seq_length, CHUNK_SIZE):
            attn_output_chunk = self._flash_attention_forward(
            query_states[:,i:i+CHUNK_SIZE], key_states[:,:i+CHUNK_SIZE], value_states[:,:i+CHUNK_SIZE], attention_mask, query_states[:,:i+CHUNK_SIZE].shape[1]
            )   
            attn_output.append(attn_output_chunk)
        del query_states, key_states, value_states
        attn_output = torch.cat(attn_output,dim=1)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.wo(attn_output)

        if not output_attentions:
            attn_weights = None

    return attn_output, attn_weights, past_key_value


def patch_model(model):
    print("Patching InternVLChatModel.generate function ...")
    model.__class__.generate = custom_generate

    print("Patching InternLM2FlashAttention2.forward function ...")
    model.language_model.model.layers[0].attention.__class__.forward = forward_internlm2flashattn2

    # model.language_model.model.layers[0].attention.config

    # self.sprompt_len = SPROMPT_LEN
    # self.bits_v = BITS_KV
    # self.scale_v = SCALE_V
    # self.per_channel_v =True
    # self.bits_k = BITS_KV
    # self.scale_k = SCALE_K
    # self.per_channel_k = True
    # self.image_token_num=int(os.environ.get('IMAGE_TOKEN_NUM'))
    # self.normalize = BITS_KV <= 2
