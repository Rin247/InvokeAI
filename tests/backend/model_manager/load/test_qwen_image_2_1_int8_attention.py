import sys
from types import SimpleNamespace

import torch
from torch.nn.functional import scaled_dot_product_attention

from invokeai.backend.model_manager.load.model_loaders import qwen_image_2_1_int8_attention


def test_qwen_image_2_1_cached_attention_uses_int8_kernel(monkeypatch) -> None:
    query = torch.randn(1, 4, 2, 128)
    key = torch.randn(1, 7, 2, 128)
    value = torch.randn(1, 7, 2, 128)
    mask = torch.ones(1, 1, 1, 7, dtype=torch.bool)
    mask[..., -1] = False
    calls = []

    def int8_attention(q, k, v, *, attn_mask):
        calls.append((q.shape, k.shape, attn_mask))
        return scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    monkeypatch.setitem(sys.modules, "comfy_kitchen", SimpleNamespace(int8_attention=int8_attention))
    monkeypatch.setattr(
        qwen_image_2_1_int8_attention,
        "_qwenimage21_prepare_qkv",
        lambda *_args: (query, key, value, query.shape[1]),
    )
    attn = SimpleNamespace(to_out=[torch.nn.Identity(), torch.nn.Identity()])

    actual = qwen_image_2_1_int8_attention.QwenImage21Int8AttnProcessor()(
        attn, torch.empty(0), kv_cache_mode="cached", attention_mask=mask
    )
    expected = (
        scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=mask)
        .transpose(1, 2)
        .flatten(2, 3)
    )

    torch.testing.assert_close(actual, expected)
    assert len(calls) == 1
    assert calls[0][0] == (1, 2, 4, 128)
    assert calls[0][1] == (1, 2, 7, 128)
    assert calls[0][2] is mask
