import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.qwen_image_int8_convrot_embedding import (
    QwenImageInt8ConvRotEmbedding,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.qwen_image_int8_convrot_linear import (
    QwenImageInt8ConvRotLinear,
)
from invokeai.backend.model_manager.load.model_loaders.qwen_image import (
    _extract_qwen_image_2_1_int8_convrot,
    _replace_qwen_vl_int8_modules,
    _split_qwen_image_2_1_mlp_weights,
)


def _metadata() -> torch.Tensor:
    return torch.tensor(list(b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}'), dtype=torch.uint8)


def test_qwen_2_1_gate_up_keeps_int8_scales_when_split() -> None:
    prefix = "transformer_blocks.0.img_mlp."
    sd = {
        prefix + "gate_up.weight": torch.zeros((4, 256), dtype=torch.int8),
        prefix + "gate_up.weight_scale": torch.arange(4, dtype=torch.float32).reshape(4, 1),
        prefix + "gate_up.comfy_quant": _metadata(),
    }

    _split_qwen_image_2_1_mlp_weights(sd)
    quant_sd, layer_names = _extract_qwen_image_2_1_int8_convrot(sd)

    assert layer_names == {prefix + "gate_layer", prefix + "proj"}
    assert torch.equal(quant_sd[prefix + "gate_layer.weight_scale"], torch.tensor([[0.0], [1.0]]))
    assert torch.equal(quant_sd[prefix + "proj.weight_scale"], torch.tensor([[2.0], [3.0]]))
    assert all(key.endswith(".comfy_quant") for key in sd)


def test_qwen_2_1_int8_unsupported_format_fails() -> None:
    sd = {"attn.to_q.comfy_quant": torch.tensor(list(b'{"format":"fp8"}'), dtype=torch.uint8)}
    with pytest.raises(ValueError, match="Unsupported Qwen Image 2.1 quantization"):
        _extract_qwen_image_2_1_int8_convrot(sd)


def test_qwen_2_1_int8_linear_has_no_cpu_fallback() -> None:
    original = torch.nn.Linear(256, 2, bias=False, device="meta")
    layer = QwenImageInt8ConvRotLinear(original, torch.Size((2, 256)), torch.Size((2, 1)))
    layer.load_state_dict(
        {
            "weight": torch.zeros((2, 256), dtype=torch.int8),
            "weight_scale": torch.ones((2, 1), dtype=torch.float32),
        },
        assign=True,
    )
    assert layer.state_dict()["weight"].dtype == torch.int8
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        layer(torch.zeros((1, 256), dtype=torch.bfloat16))


def test_qwen_vl_int8_embedding_and_linear_are_loaded_without_dequantizing() -> None:
    model = torch.nn.Module()
    model.encoder = torch.nn.Module()
    model.encoder.embed_tokens = torch.nn.Embedding(4, 256, device="meta")
    model.encoder.proj = torch.nn.Linear(256, 2, bias=False, device="meta")
    sd = {
        "encoder.embed_tokens.weight": torch.zeros((4, 256), dtype=torch.int8),
        "encoder.embed_tokens.weight_scale": torch.ones((4, 1), dtype=torch.float32),
        "encoder.proj.weight": torch.zeros((2, 256), dtype=torch.int8),
        "encoder.proj.weight_scale": torch.ones((2, 1), dtype=torch.float32),
    }

    _replace_qwen_vl_int8_modules(model, sd, {"encoder.embed_tokens", "encoder.proj"}, torch.bfloat16)
    model.load_state_dict(sd, assign=True)

    assert isinstance(model.encoder.embed_tokens, QwenImageInt8ConvRotEmbedding)
    assert isinstance(model.encoder.proj, QwenImageInt8ConvRotLinear)
    assert model.encoder.embed_tokens.weight.dtype == torch.int8
    assert model.encoder.embed_tokens.weight_scale.dtype == torch.float32
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        model.encoder.embed_tokens(torch.tensor([1]))
