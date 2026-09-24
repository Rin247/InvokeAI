import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.cast_to_device import cast_to_device
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_linear import (
    CustomLinear,
    autocast_linear_forward_sidecar_patches,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_module_mixin import (
    CustomModuleMixin,
)


class QwenImageInt8ConvRotLinear(CustomLinear):
    """Run Comfy-Org INT8 ConvRot weights without expanding them to BF16."""

    def __init__(self, original: torch.nn.Linear, weight_shape: torch.Size, scale_shape: torch.Size):
        torch.nn.Module.__init__(self)
        CustomModuleMixin.__init__(self)
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.weight = torch.nn.Parameter(
            torch.empty(weight_shape, device="meta", dtype=torch.int8), requires_grad=False
        )
        self.register_buffer("weight_scale", torch.empty(scale_shape, device="meta"))
        self.bias = original.bias

    def _autocast_forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.device.type != "cuda":
            raise RuntimeError("Qwen Image INT8 ConvRot requires a CUDA device")
        from comfy_kitchen.backends.cuda import int8_linear

        weight = cast_to_device(self.weight, input.device)
        weight_scale = cast_to_device(self.weight_scale, input.device)
        bias = self._cast_tensor_for_input(self.bias, input)
        return int8_linear(input, weight, weight_scale, bias, convrot=True, convrot_groupsize=256)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.get_num_patches():
            return autocast_linear_forward_sidecar_patches(self, input, self._patches_and_weights)
        return self._autocast_forward(input)
