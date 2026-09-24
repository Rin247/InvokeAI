from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from accelerate import init_empty_weights

import torch

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base
from invokeai.backend.model_manager.configs.main import (
    Main_TorchAO_Flux2_Config,
    Main_TorchAO_Flux_Config,
    Main_TorchAO_QwenImage_Config,
)
from invokeai.backend.model_manager.load.load_base import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, QwenImageVariantType
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.util.state_dict_loading import load_state_dict_ignoring_extras

logger = InvokeAILogger.get_logger(__name__)


try:
    import torchao

    _torchao_available = True
except ImportError:
    _torchao_available = False


def _check_torchao_available() -> None:
    if not _torchao_available:
        raise ImportError(
            "TorchAO is required for TorchAO-quantized models. "
            "Install it with: pip install torchao"
        )


def _check_quantization_kernel_support(quant_type: str) -> None:
    """Validate that the current device supports the requested TorchAO quantization.

    INT4 weight-only requires CUDA with compute capability >= 7.0 (Turing+).
    INT8 is broadly supported on CUDA/CPU.
    FP8 weight-only requires CUDA with compute capability >= 8.9 (Ada Lovelace+).

    CPU, MPS, and older CUDA GPUs will be rejected for INT4/FP8 with a clear message.
    """
    device = TorchDevice.choose_torch_device()

    if quant_type == "int8_weight_only":
        return

    if device.type == "cpu":
        raise RuntimeError(
            f"TorchAO {quant_type} is not supported on CPU. "
            "Use a CUDA device, or fall back to int8_weight_only."
        )

    if device.type == "mps":
        raise RuntimeError(
            f"TorchAO {quant_type} is not supported on MPS. "
            "Use a CUDA device, or fall back to int8_weight_only."
        )

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for TorchAO quantization.")

        # torch.cuda.get_device_capability returns (major, minor), e.g. (8, 9) for Ada Lovelace.
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor

        if quant_type == "int4_weight_only" and compute_capability < 70:
            raise RuntimeError(
                f"TorchAO int4_weight_only requires CUDA compute capability >= 7.0 (Turing+), "
                f"but the current device reports {major}.{minor}. "
                "Fall back to int8_weight_only or use a newer GPU."
            )

        if quant_type == "fp8_weight_only" and compute_capability < 89:
            raise RuntimeError(
                f"TorchAO fp8_weight_only requires CUDA compute capability >= 8.9 (Ada Lovelace+), "
                f"but the current device reports {major}.{minor}. "
                "Fall back to int8_weight_only or int4_weight_only, or use a newer GPU."
            )


def _apply_torchao_quantization(model: Any, quant_type: str, group_size: int) -> Any:
    """Apply TorchAO weight-only quantization to an nn.Module in-place.

    Args:
        model: The full-precision model to quantize.
        quant_type: One of ``int4_weight_only``, ``int8_weight_only``, ``fp8_weight_only``.
        group_size: Group size for per-group quantization (ignored for per-tensor/fp8).

    Returns:
        The quantized model (same object, mutated in-place).
    """
    _check_torchao_available()
    _check_quantization_kernel_support(quant_type)

    if quant_type == "int4_weight_only":
        from torchao.dtypes import Int4WeightOnlyConfig
        from torchao.quantization import quantize_

        config = Int4WeightOnlyConfig(group_size=group_size)
        quantize_(model, config)
    elif quant_type == "int8_weight_only":
        from torchao.dtypes import Int8WeightOnlyConfig
        from torchao.quantization import quantize_

        config = Int8WeightOnlyConfig()
        quantize_(model, config)
    elif quant_type == "fp8_weight_only":
        from torchao.dtypes import Float8WeightOnlyConfig
        from torchao.quantization import quantize_

        config = Float8WeightOnlyConfig()
        quantize_(model, config)
    else:
        raise ValueError(f"Unsupported TorchAO quant_type: {quant_type!r}")

    model._invoke_skip_shared_weights = True  # type: ignore[attr-defined]
    return model


def _load_flux_from_singlefile(self: ModelLoader, config: Main_TorchAO_Flux_Config) -> Any:
    from invokeai.backend.flux.model import Flux
    from invokeai.backend.flux.util import get_flux_transformers_params
    from invokeai.backend.model_manager.util.model_util import convert_bundle_to_flux_transformer_checkpoint
    from safetensors.torch import load_file

    model_path = Path(config.path)

    with init_empty_weights():
        model = Flux(get_flux_transformers_params(config.variant))

    sd = load_file(model_path)
    if "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale" in sd:
        sd = convert_bundle_to_flux_transformer_checkpoint(sd)
    new_sd_size = sum([ten.nelement() * torch.bfloat16.itemsize for ten in sd.values()])
    self._ram_cache.make_room(new_sd_size)
    for k in sd.keys():
        sd[k] = sd[k].to(torch.bfloat16)
    load_state_dict_ignoring_extras(model, sd, source="FLUX transformer checkpoint", assign=True)
    return model


def _load_flux2_from_singlefile(self: ModelLoader, config: Main_TorchAO_Flux2_Config) -> Any:
    from invokeai.backend.model_manager.load.model_loaders.comfyui_state_dict_utils import (
        _dequantize_comfyui_fp8,
        _strip_comfyui_prefix,
    )
    from invokeai.backend.model_manager.util.model_util import convert_flux2_bfl_to_diffusers
    from diffusers import Flux2Transformer2DModel
    from safetensors.torch import load_file

    model_path = Path(config.path)

    sd = load_file(model_path)
    _dequantize_comfyui_fp8(sd, compute_dtype=torch.bfloat16)

    keys_to_remove = [
        k
        for k in sd.keys()
        if isinstance(k, str)
        and (k.endswith(".weight_scale") or k.endswith(".scale_weight") or "comfy_quant" in k or k == "scaled_fp8")
    ]
    for k in keys_to_remove:
        del sd[k]

    prefix_to_strip = None
    for prefix in ["model.diffusion_model.", "diffusion_model."]:
        if any(k.startswith(prefix) for k in sd.keys() if isinstance(k, str)):
            prefix_to_strip = prefix
            break

    if prefix_to_strip:
        sd = {
            (k[len(prefix_to_strip) :] if isinstance(k, str) and k.startswith(prefix_to_strip) else k): v
            for k, v in sd.items()
        }

    converted_sd = convert_flux2_bfl_to_diffusers(sd)

    double_block_indices = [
        int(k.split(".")[1])
        for k in converted_sd
        if k.startswith("transformer_blocks.") and "norm." in k
    ]
    num_double_blocks = max(double_block_indices) + 1 if double_block_indices else 0

    with init_empty_weights():
        model = Flux2Transformer2DModel.from_config(
            {
                "attention_head_dim": 128,
                "guidance_embeds": False,
                "in_channels": 64,
                "joint_attention_dim": 2048,
                "num_attention_heads": 24,
                "num_double_blocks": num_double_blocks,
                "num_single_blocks": 0,
                "patch_size": 1,
                "sample_size": 128,
            }
        )

    new_sd_size = sum(t.nelement() * t.element_size() for t in converted_sd.values())
    self._ram_cache.make_room(new_sd_size)
    load_state_dict_ignoring_extras(model, converted_sd, source="FLUX.2 transformer checkpoint", assign=True)
    return model


def _load_qwen_image_from_singlefile(self: ModelLoader, config: Main_TorchAO_QwenImage_Config) -> Any:
    from invokeai.backend.model_manager.load.model_loaders.comfyui_state_dict_utils import (
        _dequantize_comfyui_fp8,
        _strip_comfyui_prefix,
        _strip_quantization_metadata,
    )
    from invokeai.backend.model_manager.load.model_loaders.qwen_image import (
        _build_qwen_image_2_1_transformer_config,
        _build_qwen_image_transformer_config,
    )
    from diffusers import QwenImage21Transformer2DModel, QwenImageTransformer2DModel
    from safetensors.torch import load_file

    model_path = Path(config.path)

    target_device = TorchDevice.choose_torch_device()
    compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

    sd = load_file(str(model_path))
    _dequantize_comfyui_fp8(sd, compute_dtype=torch.bfloat16)

    keys_to_remove = [
        k
        for k in sd.keys()
        if isinstance(k, str)
        and (k.endswith(".weight_scale") or k.endswith(".scale_weight") or "comfy_quant" in k or k == "scaled_fp8")
    ]
    for k in keys_to_remove:
        del sd[k]

    is_2_1 = getattr(config, "variant", None) == QwenImageVariantType.V2_1
    is_edit = getattr(config, "variant", None) == QwenImageVariantType.Edit
    model_config = (
        _build_qwen_image_2_1_transformer_config(sd)
        if is_2_1
        else _build_qwen_image_transformer_config(sd, is_edit=is_edit)
    )
    if is_2_1:
        from invokeai.backend.model_manager.load.model_loaders.qwen_image import _split_qwen_image_2_1_mlp_weights
        _split_qwen_image_2_1_mlp_weights(sd)

    model_cls = QwenImage21Transformer2DModel if is_2_1 else QwenImageTransformer2DModel

    with init_empty_weights():
        model = model_cls(**model_config)

    for k in list(sd.keys()):
        if sd[k].is_floating_point():
            sd[k] = sd[k].to(compute_dtype)

    new_sd_size = sum(t.nelement() * t.element_size() for t in sd.values())
    self._ram_cache.make_room(new_sd_size)

    load_state_dict_ignoring_extras(
        model, sd, source="Qwen-Image transformer checkpoint", assign=True, allow_missing=True
    )
    return model


@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.Main, format=ModelFormat.TorchAOQuantized)
class TorchAOFluxLoader(ModelLoader):
    """Loads a full-precision FLUX.1 checkpoint and applies TorchAO weight-only quantization."""

    def _load_model(
        self,
        config: Checkpoint_Config_Base,
        submodel_type: Optional[Any] = None,
    ) -> Any:
        if not isinstance(config, Main_TorchAO_Flux_Config):
            raise ValueError(f"Expected Main_TorchAO_Flux_Config, got {type(config).__name__}")

        model = _load_flux_from_singlefile(self, config)

        logger.info(
            "Applying TorchAO %s (group_size=%s) to FLUX.1 model %s",
            config.quant_type,
            config.group_size,
            config.name,
        )
        return _apply_torchao_quantization(model, config.quant_type, config.group_size)


@ModelLoaderRegistry.register(base=BaseModelType.Flux2, type=ModelType.Main, format=ModelFormat.TorchAOQuantized)
class TorchAOFlux2Loader(ModelLoader):
    """Loads a full-precision FLUX.2 checkpoint and applies TorchAO weight-only quantization."""

    def _load_model(
        self,
        config: Checkpoint_Config_Base,
        submodel_type: Optional[Any] = None,
    ) -> Any:
        if not isinstance(config, Main_TorchAO_Flux2_Config):
            raise ValueError(f"Expected Main_TorchAO_Flux2_Config, got {type(config).__name__}")

        model = _load_flux2_from_singlefile(self, config)

        logger.info(
            "Applying TorchAO %s (group_size=%s) to FLUX.2 model %s",
            config.quant_type,
            config.group_size,
            config.name,
        )
        return _apply_torchao_quantization(model, config.quant_type, config.group_size)


@ModelLoaderRegistry.register(base=BaseModelType.QwenImage, type=ModelType.Main, format=ModelFormat.TorchAOQuantized)
class TorchAOQwenImageLoader(ModelLoader):
    """Loads a full-precision Qwen Image checkpoint and applies TorchAO weight-only quantization."""

    def _load_model(
        self,
        config: Checkpoint_Config_Base,
        submodel_type: Optional[Any] = None,
    ) -> Any:
        if not isinstance(config, Main_TorchAO_QwenImage_Config):
            raise ValueError(f"Expected Main_TorchAO_QwenImage_Config, got {type(config).__name__}")

        model = _load_qwen_image_from_singlefile(self, config)

        logger.info(
            "Applying TorchAO %s (group_size=%s) to Qwen Image model %s",
            config.quant_type,
            config.group_size,
            config.name,
        )
        return _apply_torchao_quantization(model, config.quant_type, config.group_size)
