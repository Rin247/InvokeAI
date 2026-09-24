from typing import Any

import torch
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21AttnProcessor,
    _qwenimage21_prepare_qkv,
)


class QwenImage21Int8AttnProcessor(QwenImage21AttnProcessor):
    """Use comfy-kitchen INT8 attention for repeated Qwen Image 2.1 decode steps."""

    def __call__(self, attn: torch.nn.Module, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # Prefill has a block-causal mask; the stock processor computes its exact segments.
        if kwargs.get("kv_cache_mode") != "cached":
            return super().__call__(attn, hidden_states, **kwargs)

        from comfy_kitchen import int8_attention

        query, key, value, seq_len_q = _qwenimage21_prepare_qkv(
            attn,
            hidden_states,
            kwargs.get("rotary_emb"),
            kwargs.get("layer_cache"),
            "cached",
            kwargs.get("cache_write_slice"),
        )
        output = int8_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=kwargs.get("attention_mask"),
        )
        output = output.transpose(1, 2)[:, :seq_len_q].flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](output))
