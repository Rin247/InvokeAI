"""Helpers for normalising ComfyUI-flavoured single-file checkpoints.

Community single-file releases (CivitAI, ComfyUI-oriented Hugging Face repos)
share a small set of conventions regardless of which architecture they wrap:
an optional ``model.diffusion_model.`` key prefix, and optional fp8 weights
paired with per-tensor scale factors. These helpers undo both so a state dict
can be handed to a plain diffusers module.

Originally written for the Qwen Image loader; shared so the Wan loader doesn't
need a second copy.
"""

import json

import torch


def _build_convrot_hadamard(size: int) -> torch.Tensor:
    """Comfy-kitchen's normalized regular Hadamard matrix (self-inverse)."""
    if size not in (16, 64, 256):
        raise ValueError(f"Unsupported ComfyUI ConvRot group size: {size}")
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h / size**0.5


def _dequantize_convrot_weight(
    weight: torch.Tensor, scale: torch.Tensor, group_size: int, dtype: torch.dtype
) -> torch.Tensor:
    """Undo ComfyUI INT8 ConvRot in bounded row chunks, including large token embeddings."""
    if weight.ndim != 2 or weight.shape[1] % group_size:
        raise ValueError(f"Invalid ConvRot weight shape {tuple(weight.shape)} for group size {group_size}")
    h = _build_convrot_hadamard(group_size)
    output = torch.empty(weight.shape, dtype=dtype, device=weight.device)
    for start in range(0, weight.shape[0], 2048):
        end = min(start + 2048, weight.shape[0])
        row_scale = scale if scale.numel() == 1 else scale[start:end]
        rows = weight[start:end].to(torch.float32) * row_scale.to(torch.float32)
        rows = rows.reshape(end - start, -1, group_size)
        output[start:end] = torch.matmul(rows, h).reshape(end - start, -1).to(dtype)
    return output


def _strip_comfyui_prefix(sd: dict) -> dict:
    """Strip ComfyUI-style `model.diffusion_model.` / `diffusion_model.` prefixes from keys."""
    prefix_to_strip = None
    for prefix in ["model.diffusion_model.", "diffusion_model."]:
        if any(k.startswith(prefix) for k in sd.keys() if isinstance(k, str)):
            prefix_to_strip = prefix
            break
    if prefix_to_strip is None:
        return sd
    stripped: dict = {}
    for key, value in sd.items():
        if isinstance(key, str) and key.startswith(prefix_to_strip):
            stripped[key[len(prefix_to_strip) :]] = value
        else:
            stripped[key] = value
    return stripped


def _dequantize_comfyui_fp8(sd: dict, compute_dtype: torch.dtype) -> int:
    """Dequantize ComfyUI-style fp8_scaled weights in-place. Returns count of dequantized tensors.

    Weights are dequantized directly to `compute_dtype` (typically bf16) instead of via a
    full-precision float32 intermediate. The previous float32 path materialised a complete
    4-byte/param copy of the model before a separate downcast pass, spiking peak RAM to ~2x the
    final bf16 size (~80GB for the 20B Qwen-Image transformer). Multiplying in the target dtype
    keeps the dict at the bf16 model size plus a single transient tensor. fp8 has only 3 mantissa
    bits and bf16 shares float32's exponent range, so the bf16 multiply loses no meaningful
    precision here.

    Two key naming schemes are in the wild:
      - `<path>.weight` + `<path>.weight_scale`  (FLUX, Z-Image style)
      - `<path>.weight` + `<path>.scale_weight`  (Qwen2.5-VL fp8_scaled style, also
        emits `<path>.scale_input` for activation scaling that we discard).

    The scale is applied whatever dtype the weight is stored in. There is deliberately no
    "only if the weight is fp8" gate: not every checkpoint using these keys stores fp8
    weights, and skipping the multiply for those would produce a silently wrong model in
    the same way applying a stale scale would. The assumption is that a scale key present
    in the file is a scale that still needs applying — i.e. a checkpoint must not ship
    already-dequantized weights alongside their scales.
    """
    scale_suffixes = (".weight_scale", ".scale_weight")
    weight_scale_keys = [k for k in sd.keys() if isinstance(k, str) and k.endswith(scale_suffixes)]
    count = 0
    for scale_key in weight_scale_keys:
        for suffix in scale_suffixes:
            if scale_key.endswith(suffix):
                weight_key = scale_key[: -len(suffix)] + ".weight"
                break
        if weight_key not in sd:
            continue
        quant_key = weight_key[: -len(".weight")] + ".comfy_quant"
        quant_config = json.loads(bytes(sd[quant_key].tolist())) if quant_key in sd else None
        if quant_config is not None and quant_config.get("format") != "int8_tensorwise":
            raise ValueError(f"Unsupported ComfyUI quantization format: {quant_config.get('format')}")
        if quant_config is not None and quant_config.get("convrot"):
            sd[weight_key] = _dequantize_convrot_weight(
                sd[weight_key], sd[scale_key], int(quant_config["convrot_groupsize"]), compute_dtype
            )
            count += 1
            continue
        weight = sd[weight_key].to(compute_dtype)
        scale = sd[scale_key].to(compute_dtype)
        if scale.shape != weight.shape and scale.numel() > 1:
            for dim in range(len(weight.shape)):
                if dim < len(scale.shape) and scale.shape[dim] != weight.shape[dim]:
                    block_size = weight.shape[dim] // scale.shape[dim]
                    if block_size > 1:
                        scale = scale.repeat_interleave(block_size, dim=dim)
        sd[weight_key] = weight * scale
        count += 1
    return count


def _strip_quantization_metadata(sd: dict) -> None:
    """Strip ComfyUI fp8 quantization metadata keys in-place."""
    keys_to_drop = [
        k
        for k in sd.keys()
        if isinstance(k, str)
        and (
            k.endswith(".weight_scale")
            or k.endswith(".scale_weight")
            or k.endswith(".scale_input")
            or "comfy_quant" in k
            or k == "scaled_fp8"
        )
    ]
    for k in keys_to_drop:
        del sd[k]
