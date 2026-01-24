import torch
from torch import Tensor, nn
import torch.nn.functional as F

#The most important parts in the following are basically fully taken from OneTrainer.



# --- Quantization Utils ---

def quantize_int8(x: Tensor, scale: float | Tensor) -> Tensor:
    return x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)

def quantize_int8_tensorwise(x: Tensor) -> tuple[Tensor, Tensor]:
    abs_max = x.abs().max()
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale

def quantize_int8_axiswise(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale

def dequantize(q: Tensor, scale: float | Tensor) -> Tensor:
    # For per-channel scales, ensure proper broadcasting
    # q shape: (out_features, in_features), scale: scalar or (out_features,) or (out_features, 1)
    if isinstance(scale, Tensor) and scale.numel() > 1:
        scale = scale.view(-1, 1) if q.ndim == 2 else scale.view(-1)
    return q.float() * scale

# --- LinearW8A8 ---

# Try to import torch.compiler.disable for compatibility with torch.compile
try:
    from torch.compiler import disable as compiler_disable
except ImportError:
    # Older PyTorch - use identity decorator
    def compiler_disable(fn=None, recursive=True):
        if fn is None:
            return lambda f: f
        return fn

@torch.no_grad()
def int8_forward_dynamic(x: Tensor, weight: Tensor, weight_scale: float | Tensor, bias: Tensor | None, compute_dtype: torch.dtype, chunk_size: int = 0) -> Tensor:
    """Forward with dynamic per-token activation quantization."""
    if chunk_size > 0 and x.shape[0] > chunk_size:
        out = torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=compute_dtype)
        for i in range(0, x.shape[0], chunk_size):
            chunk = x[i:i+chunk_size]
            out[i:i+chunk_size] = int8_forward_dynamic(chunk, weight, weight_scale, bias, compute_dtype, chunk_size=0)
            del chunk
        return out

    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    res = torch._int_mm(x_8, weight.T)
    del x_8
    
    # x_scale shape: (batch, 1), weight_scale: scalar or (1, out_features) or (out_features,)
    if isinstance(weight_scale, Tensor) and weight_scale.numel() > 1:
        weight_scale = weight_scale.reshape(1, -1)
    
    # Use explicit multiplication for scale to handle broadcasting
    scale = x_scale * weight_scale
    res_scaled = res.float().mul_(scale).to(compute_dtype)
    del res, scale, x_scale
    
    if bias is not None:
        res_scaled.add_(bias.to(compute_dtype))
    return res_scaled

@torch.no_grad()
def int8_forward_static(x: Tensor, weight: Tensor, weight_scale: float | Tensor, input_scale: float | Tensor, bias: Tensor | None, compute_dtype: torch.dtype, chunk_size: int = 0) -> Tensor:
    """Forward with static (learned) activation quantization."""
    if chunk_size > 0 and x.shape[0] > chunk_size:
        out = torch.empty((x.shape[0], weight.shape[0]), device=x.device, dtype=compute_dtype)
        for i in range(0, x.shape[0], chunk_size):
            chunk = x[i:i+chunk_size]
            out[i:i+chunk_size] = int8_forward_static(chunk, weight, weight_scale, input_scale, bias, compute_dtype, chunk_size=0)
            del chunk
        return out

    # Quantize input using learned scale
    x_8 = quantize_int8(x, input_scale)
    res = torch._int_mm(x_8, weight.T)
    del x_8
    
    # Combined scale: weight_scale * input_scale
    if isinstance(weight_scale, Tensor) and weight_scale.numel() > 1:
        weight_scale = weight_scale.reshape(1, -1)
    
    scale = weight_scale * input_scale
    res_scaled = res.float().mul_(scale).to(compute_dtype)
    del res
    
    if bias is not None:
        res_scaled.add_(bias.to(compute_dtype))
    return res_scaled



# =============================================================================
# Int8TensorwiseOps - Proper ComfyUI Custom Operations Class
# =============================================================================
# This replaces the old dequant→load→requant hack with direct int8 loading.

try:
    from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
    _COMFY_OPS_AVAILABLE = True
except ImportError:
    _COMFY_OPS_AVAILABLE = False


if _COMFY_OPS_AVAILABLE:
    def cast_int8_weights(layer, device):
        """Cast INT8 weights to target device, supporting offloading."""
        # Note: non_blocking=False to avoid cudaMallocAsync issues with WAN 2.2
        weight = layer.weight.to(device)
        
        weight_scale = layer.weight_scale
        if isinstance(weight_scale, torch.Tensor):
            weight_scale = weight_scale.to(device)
        
        input_scale = layer.input_scale
        if isinstance(input_scale, torch.Tensor):
            input_scale = input_scale.to(device)
            
        bias = None
        if layer.bias is not None:
            bias = layer.bias.to(device)
            
        return weight, weight_scale, input_scale, bias

    class Int8TensorwiseOps(manual_cast):
        """
        Custom ComfyUI operations for INT8 tensorwise quantization.
        
        This properly integrates with ComfyUI's model loading while keeping
        the blazing fast torch._int_mm forward path.
        
        Usage:
            model_options = {"custom_operations": Int8TensorwiseOps}
            model = comfy.sd.load_diffusion_model(path, model_options=model_options)
        """
        excluded_names = []
        offload_to_cpu = True
        chunk_size = 0
        auto_convert_to_int8 = True  # Auto-convert float models (e.g., flux2 klein) to INT8 on load
        
        class Linear(manual_cast.Linear):
            """Linear layer that directly loads int8 weights and uses fast _int_mm."""
            
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.weight_scale = None
                self.input_scale = None
                self._is_quantized = False
                self.compute_dtype = torch.bfloat16
                self.comfy_cast_weights = True
                self.lora_patches = []
            
            def reset_parameters(self):
                # Skip weight initialization - we load from state dict
                return None
            
            def _load_from_state_dict(
                self,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            ):
                """
                Directly load int8 weights and scales from state dict.
                No dequant/requant needed!
                """
                weight_key = prefix + "weight"
                scale_key = prefix + "weight_scale"
                input_scale_key = prefix + "input_scale"
                bias_key = prefix + "bias"
                
                # Pop scale tensors (don't let parent class see them)
                weight_scale = state_dict.pop(scale_key, None)
                input_scale = state_dict.pop(input_scale_key, None)
                
                # Pop comfy_quant metadata if present
                state_dict.pop(prefix + "comfy_quant", None)
                
                # Get weight tensor
                weight_tensor = state_dict.pop(weight_key, None)
                
                if weight_tensor is not None:
                    # Check if this is an int8 quantized weight
                    if weight_tensor.dtype == torch.int8 and weight_scale is not None:
                        # Direct int8 load - no dequant needed!
                        self._is_quantized = True
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        
                        # Store scale as scalar or tensor
                        if weight_scale is not None:
                            if hasattr(self, "weight_scale"):
                                if "weight_scale" in self._buffers:
                                    del self._buffers["weight_scale"]
                                else:
                                    del self.weight_scale
                            
                            if isinstance(weight_scale, torch.Tensor):
                                if weight_scale.numel() == self.out_features:
                                    weight_scale = weight_scale.float().reshape(1, -1)
                                else:
                                    weight_scale = weight_scale.float().mean().view(())
                            else:
                                weight_scale = torch.tensor(weight_scale, dtype=torch.float32)
                            
                            self.register_buffer("weight_scale", weight_scale)
                        
                        # Store input scale if present (for static quantization)
                        if input_scale is not None:
                            if hasattr(self, "input_scale"):
                                if "input_scale" in self._buffers:
                                    del self._buffers["input_scale"]
                                else:
                                    del self.input_scale
                            
                            if isinstance(input_scale, torch.Tensor):
                                input_scale = input_scale.float().mean().view(())
                            else:
                                input_scale = torch.tensor(input_scale, dtype=torch.float32)
                                
                            self.register_buffer("input_scale", input_scale)
                    elif weight_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
                        # High-precision weight - quantize on-the-fly if enabled
                        # This allows loading non-INT8 models (e.g., flux2 klein) and converting them
                        
                        # 0. Skip if auto-convert is disabled
                        if not Int8TensorwiseOps.auto_convert_to_int8:
                            self._is_quantized = False
                            self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                        else:
                            # 1. Skip if name is excluded
                            is_excluded = any(ex in prefix for ex in Int8TensorwiseOps.excluded_names)
                            
                            # 2. Skip if it's a "dim1" layer (in_features=1 or out_features=1)
                            # or if the weight tensor itself is 1D (ndim=1, e.g. [x])
                            is_dim1 = self.in_features == 1 or self.out_features == 1 or weight_tensor.ndim == 1
                            
                            if is_excluded or is_dim1:
                                reason = "excluded" if is_excluded else "dim1/1D"
                                #print(f"Skipping dynamic quantization for {prefix.rstrip('.')} ({reason})")
                                self._is_quantized = False
                                self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                            else:
                                # Auto-convert float weights to INT8 on load
                                #print(f"Auto-converting to INT8: {prefix.rstrip('.')} ({weight_tensor.dtype} -> INT8)")
                                q_weight, q_scale = quantize_int8_tensorwise(weight_tensor)
                                self.weight = nn.Parameter(q_weight, requires_grad=False)
                                self.weight_scale = q_scale
                                self._is_quantized = True
                    else:
                        # Non-quantized weight (and not a known float type) - store as-is
                        self._is_quantized = False
                        self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                else:
                    missing_keys.append(weight_key)
                
                # Handle bias
                bias_tensor = state_dict.pop(bias_key, None)
                if bias_tensor is not None:
                    self.bias = nn.Parameter(bias_tensor, requires_grad=False)
                else:
                    self.bias = None
            
            def forward(self, x: Tensor) -> Tensor:
                """Fast forward using torch._int_mm for quantized weights."""
                if not self._is_quantized:
                    # Non-quantized path - use standard ComfyUI cast
                    weight, bias, offload_stream = cast_bias_weight(
                        self, x, offloadable=True
                    )
                    out = F.linear(x, weight, bias)
                    uncast_bias_weight(self, weight, bias, offload_stream)
                    
                    # Apply LoRA patches to non-quantized layer
                    lora_patches = getattr(self, "lora_patches", [])
                    if lora_patches:
                        for down, up, alpha in lora_patches:
                            d = down.to(device=out.device, dtype=out.dtype)
                            u = up.to(device=out.device, dtype=out.dtype)
                            lora_out = F.linear(F.linear(x, d), u)
                            out.add_(lora_out, alpha=alpha)
                            del d, u, lora_out
                    return out
                
                # Quantized path - use fast int8 matmul
                compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
                
                # Flatten to 2D for matmul
                x_shape = x.shape
                x_2d = x.reshape(-1, x_shape[-1])
                
                # Support offloading for quantized weights
                # We use a custom cast to avoid unwanted dtype casting of int8 weights
                weight, weight_scale, input_scale, bias = cast_int8_weights(self, x_2d.device)
                
                # Use the appropriate forward based on batch size
                if x_2d.shape[0] > 16:
                    if input_scale is not None:
                        # Static quantization path
                        y = int8_forward_static(
                            x_2d, weight, weight_scale,
                            input_scale, bias, compute_dtype,
                            chunk_size=Int8TensorwiseOps.chunk_size
                        )
                    else:
                        # Dynamic activation quantization (default)
                        y = int8_forward_dynamic(
                            x_2d, weight, weight_scale,
                            bias, compute_dtype,
                            chunk_size=Int8TensorwiseOps.chunk_size
                        )
                else:
                    # Small batch - dequantize for accuracy
                    w_float = dequantize(weight, weight_scale).to(x.dtype)
                    y = F.linear(x_2d, w_float, bias)
                    del w_float
                
                # Apply LoRA patches
                lora_patches = getattr(self, "lora_patches", [])
                if lora_patches:
                    for down, up, alpha in lora_patches:
                        # Move LoRA weights to device if needed
                        d = down.to(device=y.device, dtype=y.dtype)
                        u = up.to(device=y.device, dtype=y.dtype)
                        
                        # Compute LoRA contribution: (x @ down.T) @ up.T
                        # This is mathematically equivalent to x @ (up @ down)
                        lora_out = F.linear(F.linear(x_2d, d), u)
                        y.add_(lora_out, alpha=alpha)
                        del d, u, lora_out

                # Explicitly delete local references to GPU tensors to help GC
                del weight, weight_scale, input_scale, bias
                
                # Reshape back
                return y.reshape(*x_shape[:-1], y.shape[-1])
        
        # Use standard ComfyUI implementations for non-Linear layers
        class GroupNorm(manual_cast.GroupNorm):
            pass
        
        class LayerNorm(manual_cast.LayerNorm):
            pass
        
        class Conv2d(manual_cast.Conv2d):
            pass
        
        class Conv3d(manual_cast.Conv3d):
            pass
        
        class ConvTranspose2d(manual_cast.ConvTranspose2d):
            pass
        
        class Embedding(manual_cast.Embedding):
            pass
        
        @classmethod
        def conv_nd(cls, dims, *args, **kwargs):
            if dims == 2:
                return cls.Conv2d(*args, **kwargs)
            elif dims == 3:
                return cls.Conv3d(*args, **kwargs)
            else:
                raise ValueError(f"unsupported dimensions: {dims}")
