# Modified based on https://github.com/lm-sys/FastChat

import warnings
from typing import Optional, Tuple

import torch
from torch import nn
import transformers
from einops import rearrange
# from flash_attn import __version__ as flash_attn_version
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_kvpacked_func
)
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv, rotate_half
from transformers.cache_utils import Cache, DynamicCache
from flash_attn.bert_padding import unpad_input, pad_input
import math
import numpy as np

recent_size = 8
sink_size = 8
fft_ratio = 0.5
fft_span = 4096 - sink_size - recent_size
fft_size = int(fft_span * fft_ratio)
# fft_size = 2048 - sink_size


# Disable the transformation of the attention mask in LlamaModel as flash attention
# takes a boolean key_padding_mask. Fills in the past kv length for use in forward.
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    if past_key_values_length > 0 and attention_mask is not None:
        attention_mask = torch.cat(
            (
                torch.full(
                    (input_shape[0], past_key_values_length),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                ),
                attention_mask
            ),
            dim=-1
        )

    return attention_mask


def dct(x, norm='ortho'):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """

    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.fft.fft(v.to(torch.float32), dim=1)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc.real * W_r - Vc.imag * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm='ortho'):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2
    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    V = torch.view_as_complex(V)

    v = torch.fft.ifft(V, dim=1).real
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)

def dct_compress(x, compress_len=2048, seq_dim=2, kv_type="key"):
    # mean of the sequence dimension
    if compress_len == 0:
        return x[:,:,0:0]
    elif compress_len >= x.shape[seq_dim]:
        return x
    
    bsz, num_heads, q_len, head_dim = x.shape
    x = x.transpose(1, 2).reshape(bsz, q_len, num_heads * head_dim)

    # x_mean = x.mean(1, keepdim=True)

    x_dct = dct(x.transpose(1, 2), norm='ortho')
    x_dct = x_dct[:, :, :compress_len]
    x_idct = idct(x_dct, norm='ortho').transpose(1, 2) * np.sqrt(compress_len / q_len)

    # x_idct = idct(x_dct, norm='ortho').transpose(1, 2)

    compressed_x = x_idct.to(x.dtype)

    return compressed_x.reshape(bsz, compress_len, num_heads, head_dim).transpose(1, 2)


def apply_rotary_pos_emb_varlen(q, k, cos, sin, mem_len=0, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos[:, :, mem_len:]) + (rotate_half(q) * sin[:, :, mem_len:])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def forward_noiterate_flashattn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, h_size = hidden_states.size()

    cache_size = sink_size + fft_span + recent_size
    if q_len <= cache_size:
        num_group = 0
    else:
        num_group = math.ceil((q_len - cache_size) / (fft_span - fft_size))

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    # Past Key value support
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # transform the data into the format required by flash attention
    kv = torch.stack(
        [key_states[:, :, :cache_size], value_states[:, :, :cache_size]], dim=2
    )  # [bsz, nh, 2, kv_len, hd]
    kv = kv.transpose(1, 3)  # [bsz, kv_len, 2, nh, hd]

    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask

    key_padding_mask = attention_mask[:, :cache_size]
    unpadded_kv, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(kv, key_padding_mask)
    unpadded_q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, key_padding_mask[:, -cache_size:])

    attn_output = flash_attn_varlen_kvpacked_func(
        unpadded_q, unpadded_kv, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_p=dropout_rate, softmax_scale=None, causal=True
    )
    attn_output = pad_input(attn_output, indices_q, bsz, group_q_len)
    attn_output = attn_output.reshape(bsz, cache_size, self.hidden_size)

    # import pdb
    # pdb.set_trace()
    for group_idx in range(num_group):
        q = query_states[:, :, cache_size+group_idx*(fft_span-fft_size):min(cache_size+(group_idx+1)*(fft_span-fft_size), q_len)]
        q = q.transpose(1, 2)
        group_q_len = q.size(1)
        
        compressed_key = dct_compress(key_states[:, :, sink_size:cache_size+group_idx*(fft_span-fft_size)-recent_size], fft_size, 2, "key")
        # compressed_key = dct_compress(group_key[:, :, sink_size:], fft_size, 2, "key")
        group_key = torch.cat((key_states[:, :, :sink_size], compressed_key, key_states[:, :, cache_size+group_idx*(fft_span-fft_size)-recent_size:min(cache_size+(group_idx+1)*(fft_span-fft_size), q_len)]), 2)

        compressed_value = dct_compress(value_states[:, :, sink_size:cache_size+group_idx*(fft_span-fft_size)-recent_size], fft_size, 2, "value")
        # compressed_value = dct_compress(group_value[:, :, sink_size:], fft_size, 2, "value")
        group_value = torch.cat((value_states[:, :, :sink_size], compressed_value, value_states[:, :, cache_size+group_idx*(fft_span-fft_size)-recent_size:min(cache_size+(group_idx+1)*(fft_span-fft_size), q_len)]), 2)

        kv = torch.stack([group_key, group_value], 2)
        kv = kv.transpose(1, 3)

        group_attention_mask = torch.cat((attention_mask[:, :sink_size+fft_size+recent_size], attention_mask[:, cache_size+group_idx*(fft_span-fft_size):min(cache_size+(group_idx+1)*(fft_span-fft_size), q_len)]), 1)
        
        unpadded_kv, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(kv, group_attention_mask)
        unpadded_q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, group_attention_mask[:, -group_q_len:])
        group_attn_output = flash_attn_varlen_kvpacked_func(
            unpadded_q, unpadded_kv, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_p=0.0, softmax_scale=None, causal=True
        )
        group_attn_output = pad_input(group_attn_output, indices_q, bsz, group_q_len).reshape(bsz, group_q_len, h_size)
        attn_output = torch.cat((attn_output, group_attn_output), 1)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

layer_idx=0
def forward_iterate_flashattn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    # global layer_idx
    # layer_idx += 1
    # print(layer_idx)
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, h_size = hidden_states.size()

    cache_size = min(sink_size + fft_span + recent_size, q_len)
    num_group = math.ceil((q_len - cache_size) / (fft_span - fft_size))

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    # import pdb
    # pdb.set_trace()
    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    cos = cos[:, :cache_size]
    sin = sin[:, :cache_size]
    q, k = apply_rotary_pos_emb(query_states[:,:,:cache_size], key_states[:,:,:cache_size], cos, sin)
    q = q.transpose(1, 2)
    kv = torch.stack(
        [k, value_states[:, :, :cache_size]], dim=2
    )  # [bsz, nh, 2, kv_len, hd]
    kv = kv.transpose(1, 3)  # [bsz, kv_len, 2, nh, hd]
    
    key_padding_mask = torch.ones(bsz, cache_size, device=key_states.device)
    unpadded_kv, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(kv, key_padding_mask)
    unpadded_q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, key_padding_mask)

    attn_output = flash_attn_varlen_kvpacked_func(
        unpadded_q, unpadded_kv, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_p=dropout_rate, softmax_scale=None, causal=True
    )
    attn_output = pad_input(attn_output, indices_q, bsz, cache_size)
    attn_output = attn_output.reshape(bsz, cache_size, self.hidden_size)

    # import pdb
    # pdb.set_trace()
    group_key = key_states[:, :, :cache_size]
    group_value = value_states[:, :, :cache_size]
    for group_idx in range(num_group):
        q = query_states[:, :, cache_size+group_idx*(fft_span-fft_size):min(cache_size+(group_idx+1)*(fft_span-fft_size), q_len)]
        # group_cos = cos[:, sink_size+fft_size:sink_size+fft_size+q.shape[2]]
        # group_sin = sin[:, sink_size+fft_size:sink_size+fft_size+q.shape[2]]
        group_cos = cos[:, :sink_size+fft_size+q.shape[2]]
        group_sin = sin[:, :sink_size+fft_size+q.shape[2]]
        
        compressed_key = dct_compress(group_key[:, :, sink_size:sink_size+fft_span], fft_size, 2, "key")
        group_key = torch.cat((key_states[:, :, :sink_size], compressed_key, key_states[:, :, cache_size+group_idx*(fft_span-fft_size)-recent_size:min(cache_size+(group_idx+1)*(fft_span-fft_size), q_len)]), 2)
        # group_key = torch.cat((key_states[:, :, :sink_size], group_key[:, :, -fft_size:], key_states[:, :, cache_size+group_idx*(fft_span-fft_size)-recent_size:min(cache_size+(group_idx+1)*(fft_span-fft_size), q_len)]), 2)

        compressed_value = dct_compress(group_value[:, :, sink_size:sink_size+fft_span], fft_size, 2, "value")
        group_value = torch.cat((value_states[:, :, :sink_size], compressed_value, value_states[:, :, cache_size+group_idx*(fft_span-fft_size)-recent_size:min(cache_size+(group_idx+1)*(fft_span-fft_size), q_len)]), 2)
        # group_value = torch.cat((value_states[:, :, :sink_size], group_value[:, :, -fft_size:], value_states[:, :, cache_size+group_idx*(fft_span-fft_size)-recent_size:min(cache_size+(group_idx+1)*(fft_span-fft_size), q_len)]), 2)

        # k = group_key.clone()
        q, k = apply_rotary_pos_emb_varlen(q, group_key, group_cos, group_sin, sink_size+fft_size)
        q = q.transpose(1, 2)
        group_q_len = q.size(1)

        kv = torch.stack([k, group_value], 2)
        kv = kv.transpose(1, 3)
        # import pdb
        # pdb.set_trace()
        group_attention_mask = torch.ones(bsz, group_key.shape[2], device=group_key.device)
        unpadded_kv, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(kv, group_attention_mask)
        unpadded_q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, group_attention_mask[:, -group_q_len:])

        group_attn_output = flash_attn_varlen_kvpacked_func(
            unpadded_q, unpadded_kv, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            dropout_p=dropout_rate, softmax_scale=None, causal=True
        )
        group_attn_output = pad_input(group_attn_output, indices_q, bsz, group_q_len).reshape(bsz, group_q_len, h_size)
        attn_output = torch.cat((attn_output, group_attn_output), 1)

    past_key_value = (group_key, group_value) if use_cache else None

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def forward_noiterate_noflashattn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    cache_size = sink_size + fft_span + recent_size
    if q_len <= cache_size:
        num_group = 0
    else:
        num_group = math.ceil((q_len - cache_size) / (fft_span - fft_size))

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states[:, :, :cache_size], key_states[:, :, :cache_size].transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, cache_size, cache_size):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, cache_size, cache_size)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask[:, :, :cache_size, :cache_size]

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states[:, :, :cache_size])

    # import pdb
    # pdb.set_trace()
    # group_key = key_states[:, :, :cache_size]
    # group_value = value_states[:, :, :cache_size]
    for group_idx in range(num_group):
        group_query = query_states[:, :, cache_size+group_idx*(fft_span-fft_size):min(cache_size+(group_idx+1)*(fft_span-fft_size), q_len)]
        
        compressed_key = dct_compress(key_states[:, :, sink_size:cache_size+group_idx*(fft_span-fft_size)-recent_size], fft_size, 2, "key")
        # compressed_key = dct_compress(group_key[:, :, sink_size:], fft_size, 2, "key")
        group_key = torch.cat((key_states[:, :, :sink_size], compressed_key, key_states[:, :, cache_size+group_idx*(fft_span-fft_size)-recent_size:min(cache_size+(group_idx+1)*(fft_span-fft_size), q_len)]), 2)
        
        compressed_value = dct_compress(value_states[:, :, sink_size:cache_size+group_idx*(fft_span-fft_size)-recent_size], fft_size, 2, "value")
        # compressed_value = dct_compress(group_value[:, :, sink_size:], fft_size, 2, "value")
        group_value = torch.cat((value_states[:, :, :sink_size], compressed_value, value_states[:, :, cache_size+group_idx*(fft_span-fft_size)-recent_size:min(cache_size+(group_idx+1)*(fft_span-fft_size), q_len)]), 2)

        group_attn_weights = torch.matmul(group_query, group_key.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            group_attention_mask = torch.zeros_like(group_attn_weights, device=group_attn_weights.device, dtype=group_attn_weights.dtype)
            group_attention_mask[:, :, :, sink_size+fft_size+recent_size:] = attention_mask[:, :, cache_size+group_idx*(fft_span-fft_size):min(cache_size+(group_idx+1)*(fft_span-fft_size), q_len), cache_size+group_idx*(fft_span-fft_size):min(cache_size+(group_idx+1)*(fft_span-fft_size), q_len)]
            group_attn_weights = group_attn_weights + group_attention_mask

        group_attn_weights = nn.functional.softmax(group_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        group_attn_output = torch.matmul(group_attn_weights, group_value)

        # attn_weights = torch.cat((attn_weights, group_attn_weights), 2)
        attn_output = torch.cat((attn_output, group_attn_output), 2)
        
    # past_key_value = (group_key, group_value) if use_cache else None

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def forward_iterate_noflashattn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    cache_size = min(sink_size + fft_span, q_len)
    num_group = math.ceil((q_len - cache_size) / (fft_span - fft_size))

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, None)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    cos = cos[:, :cache_size]
    sin = sin[:, :cache_size]
    q, k = apply_rotary_pos_emb(query_states[:,:,:cache_size], key_states[:,:,:cache_size], cos, sin)

    attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :q.shape[-2], : k.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states[:,:,:cache_size])

    # import pdb
    # pdb.set_trace()
    group_key = key_states[:, :, :cache_size]
    group_value = value_states[:, :, :cache_size]
    for group_idx in range(num_group):
        q = query_states[:, :, cache_size+group_idx*(fft_span-fft_size):min(cache_size+(group_idx+1)*(fft_span-fft_size), q_len)]
        
        group_cos = cos[:, :sink_size+fft_size+q.shape[2]]
        group_sin = sin[:, :sink_size+fft_size+q.shape[2]]
        # compressed_key = dct_compress(key_states[:, :, sink_size:cache_size+group_idx*(fft_span-fft_size)], fft_size, 2, "key")
        compressed_key = dct_compress(group_key[:, :, sink_size:sink_size+fft_span], fft_size, 2, "key")
        group_key = torch.cat((key_states[:, :, :sink_size], compressed_key, key_states[:, :, cache_size+group_idx*(fft_span-fft_size)-recent_size:min(cache_size+(group_idx+1)*(fft_span-fft_size), q_len)]), 2)
        
        # compressed_value = dct_compress(value_states[:, :, sink_size:cache_size+group_idx*(fft_span-fft_size)], fft_size, 2, "value")
        compressed_value = dct_compress(group_value[:, :, sink_size:sink_size+fft_span], fft_size, 2, "value")
        group_value = torch.cat((value_states[:, :, :sink_size], compressed_value, value_states[:, :, cache_size+group_idx*(fft_span-fft_size)-recent_size:min(cache_size+(group_idx+1)*(fft_span-fft_size), q_len)]), 2)

        q, k = apply_rotary_pos_emb_varlen(q, group_key, group_cos, group_sin, sink_size+fft_size)

        group_attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            group_attention_mask = torch.ones_like(group_attn_weights, device=group_attn_weights.device, dtype=group_attn_weights.dtype)
            group_attention_mask[:, :, :, sink_size+fft_size+recent_size:] = attention_mask[:, :, cache_size+group_idx*(fft_span-fft_size):min(cache_size+(group_idx+1)*(fft_span-fft_size), q_len), cache_size+group_idx*(fft_span-fft_size):min(cache_size+(group_idx+1)*(fft_span-fft_size), q_len)]
            group_attn_weights = group_attn_weights + group_attention_mask

        group_attn_weights = nn.functional.softmax(group_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        group_attn_weights = nn.functional.dropout(group_attn_weights, p=self.attention_dropout, training=self.training)
        group_attn_output = torch.matmul(group_attn_weights, group_value)

        # attn_weights = torch.cat((attn_weights, group_attn_weights), 2)
        attn_output = torch.cat((attn_output, group_attn_output), 2)
        
    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(group_key, group_value, self.layer_idx, None)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def replace_llama_attn(use_flash_attn=True, is_iterate=False, sink_size0=0, recent_size0=0, fft_ratio0=0, cache_size0=4096):
    global sink_size, recent_size, fft_ratio, fft_span, fft_size
    sink_size = sink_size0
    recent_size = recent_size0
    fft_ratio = fft_ratio0
    fft_span = cache_size0 - sink_size - recent_size
    fft_size = int(fft_span * fft_ratio)
    print(f"--sink_size: {sink_size}, \n--recent_size: {recent_size}, \n--dct_ratio: {fft_ratio}, \n--dct_span: {fft_span}, \n--dct_size: {fft_size}")

    if use_flash_attn:
        cuda_major, cuda_minor = torch.cuda.get_device_capability()
        if cuda_major < 8:
            warnings.warn(
                "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
                "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
            )
        transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
            _prepare_decoder_attention_mask
        )
        if is_iterate:
            print("forward_flashattn-dct-iterate-mempe")
            transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_iterate_flashattn
            transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = forward_iterate_flashattn
            transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = forward_iterate_flashattn
        else:
            print("forward_flashattn-dct-noiterate")
            transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_noiterate_flashattn
    else:
        if is_iterate:
            print("forward_noflashattn-dct-iterate-mempe")
            transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_iterate_noflashattn
        else:
            print("forward_noflashattn-dct-noiterate")
            transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_noiterate_noflashattn
