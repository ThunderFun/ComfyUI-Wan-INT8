"""
int88 - Fast INT8 Tensorwise Quantization for ComfyUI

Provides:
- Int8TensorwiseOps: Custom operations for direct int8 weight loading
- OTUNetLoaderW8A8: Load int8 quantized diffusion models
- OTCheckpointLoaderW8A8: Load int8 quantized checkpoints

Uses torch._int_mm for blazing fast inference.
"""

# torch.compile + INT8 compatibility fix
# CUDA graphs are incompatible with cudaMallocAsync allocator.
# We patch the problematic functions to work around this.

import torch
import os

def _apply_cudagraph_compatibility_fix():
    """
    Patch PyTorch's CUDA graph memory pool checking to work with cudaMallocAsync.
    This allows torch.compile to work without requiring PYTORCH_ALLOC_CONF=backend:native.
    """
    
    # === Patch torch._C._cuda_checkPoolLiveAllocations ===
    # This is the C++ function that throws "cudaMallocAsync does not yet support checkPoolLiveAllocations"
    if hasattr(torch._C, '_cuda_checkPoolLiveAllocations'):
        _original_check = torch._C._cuda_checkPoolLiveAllocations
        def _safe_check(*args, **kwargs):
            try:
                return _original_check(*args, **kwargs)
            except RuntimeError as e:
                if "cudaMallocAsync" in str(e):
                    # Return True to indicate "all allocations are accounted for"
                    # This lets CUDA graphs proceed without the memory check
                    return True
                raise
        torch._C._cuda_checkPoolLiveAllocations = _safe_check
    
    # === Patch torch._C._cuda_memorySnapshot ===
    if hasattr(torch._C, '_cuda_memorySnapshot'):
        _original_snapshot = torch._C._cuda_memorySnapshot
        def _safe_snapshot(*args, **kwargs):
            try:
                return _original_snapshot(*args, **kwargs)
            except RuntimeError as e:
                if "cudaMallocAsync" in str(e) or "snapshot" in str(e).lower():
                    return {"segments": [], "device_traces": []}
                raise
        torch._C._cuda_memorySnapshot = _safe_snapshot
    
    # === Patch torch._inductor.cudagraph_trees.check_memory_pool ===
    try:
        from torch._inductor import cudagraph_trees
        if hasattr(cudagraph_trees, 'check_memory_pool'):
            _original_check_pool = cudagraph_trees.check_memory_pool
            def _safe_check_pool(*args, **kwargs):
                try:
                    return _original_check_pool(*args, **kwargs)
                except RuntimeError as e:
                    if "cudaMallocAsync" in str(e):
                        return None  # Skip the check
                    raise
            cudagraph_trees.check_memory_pool = _safe_check_pool
    except ImportError:
        pass

_apply_cudagraph_compatibility_fix()

# Register the int8_tensorwise format with ComfyUI's quant registry
# This is for metadata compatibility when saving/loading models
try:
    from comfy.quant_ops import QUANT_ALGOS, register_layout_class, QuantizedLayout
    import torch

    class Int8TensorwiseLayout(QuantizedLayout):
        """Minimal layout class to satisfy ComfyUI's registry requirements."""
        class Params:
            def __init__(self, scale=None, orig_dtype=None, orig_shape=None, **kwargs):
                self.scale = scale
                self.orig_dtype = orig_dtype
                self.orig_shape = orig_shape
            
            def clone(self):
                import torch
                return Int8TensorwiseLayout.Params(
                    scale=self.scale.clone() if isinstance(self.scale, torch.Tensor) else self.scale,
                    orig_dtype=self.orig_dtype,
                    orig_shape=self.orig_shape
                )

        @classmethod
        def state_dict_tensors(cls, qdata, params):
            return {"": qdata, "weight_scale": params.scale}
        
        @classmethod  
        def dequantize(cls, qdata, params):
            return qdata.float() * params.scale

    register_layout_class("Int8TensorwiseLayout", Int8TensorwiseLayout)

    cur_config = QUANT_ALGOS.get("int8_tensorwise")
    if cur_config is None:
        QUANT_ALGOS["int8_tensorwise"] = {
            "storage_t": torch.int8,
            "parameters": {"weight_scale", "input_scale"},
            "comfy_tensor_layout": "Int8TensorwiseLayout",
        }
    else:
        cur_config["comfy_tensor_layout"] = "Int8TensorwiseLayout"
        cur_config["parameters"] = {"weight_scale", "input_scale"}

except ImportError:
    pass

# Export the custom ops class for external use
try:
    from .int8_quant import Int8TensorwiseOps
except ImportError:
    Int8TensorwiseOps = None

from .int8_unet_loader import UNetLoaderINTW8A8

NODE_CLASS_MAPPINGS = {
    "OTUNetLoaderW8A8": UNetLoaderINTW8A8,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OTUNetLoaderW8A8": "Load Diffusion Model INT8 (W8A8)",
}
