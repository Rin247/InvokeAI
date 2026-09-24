import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_embedding import (
    CustomEmbedding,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_module_mixin import (
    CustomModuleMixin,
)


class QwenImageInt8ConvRotEmbedding(CustomEmbedding):
    """Gather only the requested Qwen3-VL token rows before undoing ConvRot."""

    def __init__(
        self,
        original: torch.nn.Embedding,
        weight_shape: torch.Size,
        scale_shape: torch.Size,
        output_dtype: torch.dtype,
    ):
        torch.nn.Module.__init__(self)
        CustomModuleMixin.__init__(self)
        self.num_embeddings = original.num_embeddings
        self.embedding_dim = original.embedding_dim
        self.weight = torch.nn.Parameter(
            torch.empty(weight_shape, device="meta", dtype=torch.int8), requires_grad=False
        )
        self.register_buffer("weight_scale", torch.empty(scale_shape, device="meta", dtype=torch.float32))
        self.output_dtype = output_dtype

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.device.type != "cuda":
            raise RuntimeError("Qwen Image INT8 ConvRot embedding requires a CUDA device")
        if self.get_num_patches():
            raise RuntimeError("Qwen Image INT8 ConvRot embedding does not support patches")
        from comfy_kitchen.backends.eager.quantization import dequantize_int8_embedding

        dtype_code = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}[self.output_dtype]
        weight_device = self.weight.device
        rows = dequantize_int8_embedding(
            self.weight,
            self.weight_scale.to(weight_device),
            input.to(weight_device),
            group_size=256,
            output_dtype_code=dtype_code,
        )
        return rows.to(input.device)
