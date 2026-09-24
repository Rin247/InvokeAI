from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from accelerate import init_empty_weights

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base
from invokeai.backend.model_manager.load.load_base import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
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

    return model


@ModelLoaderRegistry.register(base=BaseModelType.Flux, type=ModelType.Main, format=ModelFormat.TorchAOQuantized)
class TorchAOFluxLoader(ModelLoader):
    """Loads a full-precision FLUX.1 checkpoint and applies TorchAO weight-only quantization."""

    def _load_model(
        self,
        config: Checkpoint_Config_Base,
        submodel_type: Optional[Any] = None,
    ) -> Any:
        from invokeai.backend.model_manager.configs.main import Main_TorchAO_Flux_Config
        from invokeai.backend.model_manager.load.model_loaders.flux import FluxCheckpointModel

        if not isinstance(config, Main_TorchAO_Flux_Config):
            raise ValueError(f"Expected Main_TorchAO_Flux_Config, got {type(config).__name__}")

        flux_loader = FluxCheckpointModel()
        model = flux_loader._load_model(config, submodel_type)

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
        from invokeai.backend.model_manager.configs.main import Main_TorchAO_Flux2_Config
        from invokeai.backend.model_manager.load.model_loaders.flux import Flux2CheckpointModel

        if not isinstance(config, Main_TorchAO_Flux2_Config):
            raise ValueError(f"Expected Main_TorchAO_Flux2_Config, got {type(config).__name__}")

        flux2_loader = Flux2CheckpointModel()
        model = flux2_loader._load_model(config, submodel_type)

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
        from invokeai.backend.model_manager.configs.main import Main_TorchAO_QwenImage_Config
        from invokeai.backend.model_manager.load.model_loaders.qwen_image import QwenImageCheckpointModel

        if not isinstance(config, Main_TorchAO_QwenImage_Config):
            raise ValueError(f"Expected Main_TorchAO_QwenImage_Config, got {type(config).__name__}")

        qwen_loader = QwenImageCheckpointModel()
        model = qwen_loader._load_model(config, submodel_type)

        logger.info(
            "Applying TorchAO %s (group_size=%s) to Qwen Image model %s",
            config.quant_type,
            config.group_size,
            config.name,
        )
        return _apply_torchao_quantization(model, config.quant_type, config.group_size)
