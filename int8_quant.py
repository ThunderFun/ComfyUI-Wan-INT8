import torch
from torch import Tensor, nn
import torch.nn.functional as F


# --- Quantization Utils ---

def quantize_int8(x: Tensor, scale: float | Tensor) -> Tensor:
    return x.float().mul(1.0 / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)

def quantize_int8_chunked(x: Tensor, scale: float | Tensor, chunk_elements: int = 33_554_432) -> Tensor:
    """Memory-efficient quantization for large tensors."""
    total_elements = x.numel()
    
    if total_elements <= chunk_elements:
        # Small tensor: use original fast path
        return quantize_int8(x, scale)
    
    # Large tensor: process in chunks
    result = torch.empty_like(x, dtype=torch.int8)
    
    # Process row-wise chunks
    chunk_rows = max(1, chunk_elements // x.shape[-1])
    
    for start_row in range(0, x.shape[0], chunk_rows):
        end_row = min(start_row + chunk_rows, x.shape[0])
        chunk = x[start_row:end_row]
        
        if isinstance(scale, Tensor) and scale.shape[0] > 1:
            # Per-row scale
            chunk_scale = scale[start_row:end_row]
        else:
            chunk_scale = scale
        
        result[start_row:end_row] = quantize_int8(chunk, chunk_scale)
        del chunk
    
    return result

def quantize_int8_tensorwise(x: Tensor) -> tuple[Tensor, Tensor]:
    abs_max = x.abs().max()
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8_chunked(x, scale, CHUNK_TARGET_ELEMENTS), scale

def quantize_int8_axiswise(x: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8_chunked(x, scale, CHUNK_TARGET_ELEMENTS), scale

def dequantize(q: Tensor, scale: float | Tensor) -> Tensor:
    # For per-channel scales, ensure proper broadcasting
    # q shape: (out_features, in_features), scale: scalar or (out_features,) or (out_features, 1)
    total_elements = q.numel()
    
    # Chunk large dequantizations to avoid float32 spike
    if total_elements > CHUNK_THRESHOLD_ELEMENTS:
        result = torch.empty_like(q, dtype=torch.float32)
        chunk_rows = max(1, CHUNK_TARGET_ELEMENTS // q.shape[-1])
        
        for start_row in range(0, q.shape[0], chunk_rows):
            end_row = min(start_row + chunk_rows, q.shape[0])
            chunk = q[start_row:end_row]
            
            if isinstance(scale, Tensor) and scale.numel() > 1:
                if scale.ndim == 2:
                    chunk_scale = scale[start_row:end_row]
                else:
                    chunk_scale = scale[start_row:end_row].view(-1, 1) if q.ndim == 2 else scale[start_row:end_row]
            else:
                chunk_scale = scale
            
            result[start_row:end_row] = chunk.float() * chunk_scale
            del chunk
        
        return result
    
    # Small tensor: original fast path
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

# Debug flag for OOM diagnosis - set to True to enable memory logging
_DEBUG_OOM = False

# Chunking configuration for memory-constrained scenarios
# Tensors larger than this threshold (in elements) will be processed in chunks
# Default: 67M elements (256MB in INT32)
# Increase if you have more VRAM, decrease if still experiencing OOM
CHUNK_THRESHOLD_ELEMENTS = 67_108_864

# Target chunk size for processing (in elements)
# Default: 33M elements (~128MB in float32)
# Smaller = lower peak memory but slightly slower
CHUNK_TARGET_ELEMENTS = 33_554_432

def _log_memory(msg: str):
    """Log GPU memory usage for OOM debugging."""
    if not _DEBUG_OOM:
        return
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[MEM] {msg}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Peak={max_mem:.2f}GB")

def _log_tensor_size(name: str, t: Tensor):
    """Log tensor size for OOM debugging."""
    if not _DEBUG_OOM:
        return
    size_mb = t.numel() * t.element_size() / 1024**2
    print(f"[TENSOR] {name}: shape={tuple(t.shape)}, dtype={t.dtype}, size={size_mb:.1f}MB")

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

    _log_memory("int8_forward_dynamic start")
    _log_tensor_size("input x", x)
    
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)
    res = torch._int_mm(x_8, weight.T)
    del x_8
    
    _log_tensor_size("int_mm result (INT32)", res)
    
    # x_scale shape: (batch, 1), weight_scale: scalar or (1, out_features) or (out_features,)
    if isinstance(weight_scale, Tensor) and weight_scale.numel() > 1:
        weight_scale = weight_scale.reshape(1, -1)
    
    # Use explicit multiplication for scale to handle broadcasting
    scale = x_scale * weight_scale
    
    _log_memory("before float conversion (CRITICAL POINT)")
    
    # Memory-efficient scaling with chunking for large tensors
    # This prevents creating massive float32 intermediates
    total_elements = res.numel()
    
    if total_elements > CHUNK_THRESHOLD_ELEMENTS:
        # Process in row chunks to avoid large float32 allocation
        chunk_rows = max(1, (CHUNK_TARGET_ELEMENTS // res.shape[1]))
        
        if _DEBUG_OOM:
            print(f"[CHUNK] Large tensor detected: {res.shape}, processing in chunks of {chunk_rows} rows")
        
        res_scaled = torch.empty_like(res, dtype=compute_dtype)
        
        for start_row in range(0, res.shape[0], chunk_rows):
            end_row = min(start_row + chunk_rows, res.shape[0])
            
            # Process chunk: INT32 -> float32 -> scale -> compute_dtype
            chunk_int32 = res[start_row:end_row]
            
            # Convert chunk to float, scale, and convert to target dtype
            if scale.numel() == 1 or scale.shape[0] == 1:
                # Scalar or row-broadcast scale
                res_scaled[start_row:end_row] = chunk_int32.float().mul_(scale).to(compute_dtype)
            else:
                # Per-row scale
                chunk_scale = scale[start_row:end_row]
                res_scaled[start_row:end_row] = chunk_int32.float().mul_(chunk_scale).to(compute_dtype)
            
            del chunk_int32
        
        if _DEBUG_OOM:
            print(f"[CHUNK] Completed chunked processing")
    else:
        # Small tensor: process normally (original fast path)
        res_scaled = res.float().mul_(scale).to(compute_dtype)
    
    _log_memory("after float conversion")
    
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
    
    # Memory-efficient scaling with chunking for large tensors (same as dynamic path)
    total_elements = res.numel()
    
    if total_elements > CHUNK_THRESHOLD_ELEMENTS:
        # Process in row chunks to avoid large float32 allocation
        chunk_rows = max(1, (CHUNK_TARGET_ELEMENTS // res.shape[1]))
        
        res_scaled = torch.empty_like(res, dtype=compute_dtype)
        
        for start_row in range(0, res.shape[0], chunk_rows):
            end_row = min(start_row + chunk_rows, res.shape[0])
            chunk_int32 = res[start_row:end_row]
            res_scaled[start_row:end_row] = chunk_int32.float().mul_(scale).to(compute_dtype)
            del chunk_int32
    else:
        # Small tensor: process normally
        res_scaled = res.float().mul_(scale).to(compute_dtype)
    
    del res
    
    if bias is not None:
        res_scaled.add_(bias.to(compute_dtype))
    return res_scaled

@torch.no_grad()
def chunked_lora_forward(x: Tensor, down: Tensor, up: Tensor, alpha: float, output: Tensor):
    """
    Memory-efficient float LoRA forward pass using chunking.
    Updates 'output' in-place.
    """
    x_shape = x.shape
    x_2d = x.reshape(-1, x_shape[-1])
    
    # Determine chunk size
    chunk_rows = max(1, CHUNK_TARGET_ELEMENTS // max(down.shape[0], up.shape[0]))
    
    if x_2d.shape[0] <= chunk_rows:
        lora_out = F.linear(F.linear(x_2d, down), up)
        lora_out = lora_out.reshape(*x_shape[:-1], lora_out.shape[-1])
        output.add_(lora_out, alpha=alpha)
    else:
        for i in range(0, x_2d.shape[0], chunk_rows):
            end = min(i + chunk_rows, x_2d.shape[0])
            x_chunk = x_2d[i:end]
            lora_out = F.linear(F.linear(x_chunk, down), up)
            
            if output.ndim == x.ndim:
                output_view = output.reshape(-1, output.shape[-1])
                output_view[i:end].add_(lora_out, alpha=alpha)
            else:
                output[i:end].add_(lora_out, alpha=alpha)
            del lora_out
            
            if i % (chunk_rows * 4) == 0:
                torch.cuda.synchronize()

@torch.no_grad()
def chunked_int8_lora_forward(x: Tensor, down: Tensor, up: Tensor, down_scale: float | Tensor, up_scale: float | Tensor, alpha: float, output: Tensor):
    """
    Memory-efficient INT8 LoRA forward pass using chunking.
    Updates 'output' in-place.
    """
    x_shape = x.shape
    x_2d = x.reshape(-1, x_shape[-1])
    
    # Determine chunk size based on output dimensions
    # We want to limit the size of INT32 and float32 intermediates
    chunk_rows = max(1, CHUNK_TARGET_ELEMENTS // max(down.shape[0], up.shape[0]))
    
    if x_2d.shape[0] <= chunk_rows:
        # Small enough to process in one go
        x_int8, x_scale = quantize_int8_axiswise(x_2d, dim=-1)
        lora_inter = torch._int_mm(x_int8, down.T)  # INT32
        del x_int8
        
        lora_inter = lora_inter.to(dtype=torch.float32)
        lora_inter.mul_(x_scale)
        lora_inter.mul_(down_scale)
        
        inter_int8, inter_scale = quantize_int8_axiswise(lora_inter, dim=-1)
        del lora_inter, x_scale
        
        lora_out = torch._int_mm(inter_int8, up.T)  # INT32
        del inter_int8
        
        lora_out = lora_out.to(dtype=torch.float32)
        lora_out.mul_(inter_scale)
        lora_out.mul_(up_scale)
        
        lora_out = lora_out.reshape(*x_shape[:-1], lora_out.shape[-1])
        output.add_(lora_out, alpha=alpha)
        del lora_out, inter_scale
    else:
        # Process in chunks
        for i in range(0, x_2d.shape[0], chunk_rows):
            end = min(i + chunk_rows, x_2d.shape[0])
            x_chunk = x_2d[i:end]
            
            x_int8, x_scale = quantize_int8_axiswise(x_chunk, dim=-1)
            lora_inter = torch._int_mm(x_int8, down.T)
            del x_int8
            
            lora_inter = lora_inter.to(dtype=torch.float32)
            lora_inter.mul_(x_scale)
            lora_inter.mul_(down_scale)
            
            inter_int8, inter_scale = quantize_int8_axiswise(lora_inter, dim=-1)
            del lora_inter, x_scale
            
            lora_out = torch._int_mm(inter_int8, up.T)
            del inter_int8
            
            lora_out = lora_out.to(dtype=torch.float32)
            lora_out.mul_(inter_scale)
            lora_out.mul_(up_scale)
            
            if output.ndim == x.ndim:
                output_view = output.reshape(-1, output.shape[-1])
                output_view[i:end].add_(lora_out, alpha=alpha)
            else:
                output[i:end].add_(lora_out, alpha=alpha)
                
            del lora_out, inter_scale
            
            if i % (chunk_rows * 4) == 0:
                torch.cuda.synchronize()

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
        offload_to_cpu = False
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
                # Support multiple naming conventions for scales
                weight_scale = state_dict.pop(prefix + "weight_scale",
                               state_dict.pop(prefix + "scale",
                               state_dict.pop(prefix + "weight.scale",
                               state_dict.pop(prefix + "weight.weight_scale", None))))
                
                input_scale = state_dict.pop(prefix + "input_scale",
                              state_dict.pop(prefix + "act_scale",
                              state_dict.pop(prefix + "input.scale",
                              state_dict.pop(prefix + "input.input_scale", None))))
                
                bias_key = prefix + "bias"
                
                # Pop comfy_quant metadata if present
                state_dict.pop(prefix + "comfy_quant", None)
                
                # Get weight tensor
                weight_tensor = state_dict.pop(weight_key, None)
                
                
                if weight_tensor is not None:
                    if weight_tensor.dtype == torch.int8 and weight_scale is None:
                        print(f"INT8 Loader: WARNING - Found INT8 weight but NO SCALE for {prefix.rstrip('.')}")
                    
                    # Check if this is an int8 quantized weight
                    if weight_tensor.dtype == torch.int8 and weight_scale is not None:
                        # Check if this layer should be excluded (even if it's already INT8)
                        is_excluded = any(ex in prefix for ex in Int8TensorwiseOps.excluded_names)
                        
                        if is_excluded:
                            # Dequantize back to float for sensitive layers
                            self._is_quantized = False
                            dequant_weight = dequantize(weight_tensor, weight_scale).to(torch.bfloat16)
                            self.weight = nn.Parameter(dequant_weight, requires_grad=False)
                        else:
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
                        for patch_data in lora_patches:
                            # Unpack patch data (supports both old and new format)
                            if len(patch_data) == 3:
                                # Old format: (down, up, alpha)
                                down, up, alpha = patch_data
                                down_scale, up_scale = None, None
                            else:
                                # New format: (down, up, alpha, down_scale, up_scale)
                                down, up, alpha, down_scale, up_scale = patch_data
                            
                            # Check if this is INT8 LoRA
                            is_int8 = down.dtype == torch.int8 and up.dtype == torch.int8
                            
                            if is_int8 and down_scale is not None and up_scale is not None:
                                # INT8 LoRA path
                                x_shape = x.shape
                                x_2d = x.reshape(-1, x_shape[-1])
                                
                                if x_2d.shape[0] > 16:
                                    # Fast INT8 path using torch._int_mm
                                    down_int8 = down.to(device=out.device, non_blocking=True)
                                    up_int8 = up.to(device=out.device, non_blocking=True)
                                    down_scale_t = down_scale.to(device=out.device, non_blocking=True) if isinstance(down_scale, Tensor) else down_scale
                                    up_scale_t = up_scale.to(device=out.device, non_blocking=True) if isinstance(up_scale, Tensor) else up_scale
                                    
                                    # Use memory-efficient chunked forward
                                    chunked_int8_lora_forward(
                                        x_2d, down_int8, up_int8, 
                                        down_scale_t, up_scale_t, 
                                        alpha, out
                                    )
                                    
                                    del down_int8, up_int8, down_scale_t, up_scale_t
                                    
                                    # Force memory release after each patch if it's large
                                    if out.numel() > CHUNK_THRESHOLD_ELEMENTS:
                                        torch.cuda.synchronize()
                                        torch.cuda.empty_cache()
                                else:
                                    # Fallback: Dequantize INT8 LoRA to float for small batch sizes
                                    # torch._int_mm requires M > 16
                                    d = dequantize(down, down_scale).to(device=out.device, dtype=out.dtype)
                                    u = dequantize(up, up_scale).to(device=out.device, dtype=out.dtype)
                                    lora_out = F.linear(F.linear(x, d), u)
                                    out.add_(lora_out, alpha=alpha)
                                    del d, u, lora_out
                            else:
                                # Float LoRA path
                                d = down.to(device=out.device, dtype=out.dtype)
                                u = up.to(device=out.device, dtype=out.dtype)
                                
                                # Use memory-efficient chunked forward
                                chunked_lora_forward(x, d, u, alpha, out)
                                
                                del d, u
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
                    _log_memory(f"before LoRA application ({len(lora_patches)} patches)")
                    for idx, patch_data in enumerate(lora_patches):
                        # Unpack patch data (supports both old and new format)
                        if len(patch_data) == 3:
                            # Old format: (down, up, alpha)
                            down, up, alpha = patch_data
                            down_scale, up_scale = None, None
                        else:
                            # New format: (down, up, alpha, down_scale, up_scale)
                            down, up, alpha, down_scale, up_scale = patch_data
                        
                        # Check if this is INT8 LoRA
                        is_int8 = down.dtype == torch.int8 and up.dtype == torch.int8
                        
                        if is_int8 and down_scale is not None and up_scale is not None:
                            # INT8 LoRA path
                            _log_memory(f"INT8 LoRA {idx} start")
                            
                            if x_2d.shape[0] > 16:
                                # Fast INT8 path using torch._int_mm
                                # Ensure weights are on the correct device (should be cached by loader)
                                down_int8 = down.to(device=y.device, non_blocking=True)
                                up_int8 = up.to(device=y.device, non_blocking=True)
                                down_scale_t = down_scale.to(device=y.device, non_blocking=True) if isinstance(down_scale, Tensor) else down_scale
                                up_scale_t = up_scale.to(device=y.device, non_blocking=True) if isinstance(up_scale, Tensor) else up_scale
                                
                                _log_tensor_size(f"LoRA {idx} down (INT8)", down_int8)
                                _log_tensor_size(f"LoRA {idx} up (INT8)", up_int8)
                                
                                # Use memory-efficient chunked forward
                                chunked_int8_lora_forward(
                                    x_2d, down_int8, up_int8, 
                                    down_scale_t, up_scale_t, 
                                    alpha, y
                                )
                                
                                del down_int8, up_int8, down_scale_t, up_scale_t
                                
                                # Force memory release after each patch if it's large
                                if y.numel() > CHUNK_THRESHOLD_ELEMENTS:
                                    torch.cuda.synchronize()
                                    torch.cuda.empty_cache()
                            else:
                                # Fallback: Dequantize INT8 LoRA to float for small batch sizes
                                # torch._int_mm requires M > 16
                                d = dequantize(down, down_scale).to(device=y.device, dtype=y.dtype)
                                u = dequantize(up, up_scale).to(device=y.device, dtype=y.dtype)
                                lora_out = F.linear(F.linear(x_2d, d), u)
                                y.add_(lora_out, alpha=alpha)
                                del d, u, lora_out
                            
                            _log_memory(f"INT8 LoRA {idx} end")
                        else:
                            # Float LoRA path
                            target_dtype = y.dtype
                            
                            # Convert to target device/dtype
                            d = down.to(device=y.device, dtype=target_dtype).contiguous()
                            u = up.to(device=y.device, dtype=target_dtype).contiguous()
                            
                            # Use memory-efficient chunked forward
                            chunked_lora_forward(x_2d, d, u, alpha, y)
                            
                            del d, u
                    _log_memory("after LoRA application")

                # Explicitly delete local references to GPU tensors to help GC
                del weight, weight_scale, input_scale, bias
                
                # Reshape back
                return y.reshape(*x_shape[:-1], y.shape[-1])
        
        # Use standard ComfyUI implementations for non-Linear layers
        class GroupNorm(manual_cast.GroupNorm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.lora_patches = []
        
        class LayerNorm(manual_cast.LayerNorm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.lora_patches = []
        
        class Conv2d(manual_cast.Conv2d):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.lora_patches = []
            
            def forward(self, x):
                out = super().forward(x)
                lora_patches = getattr(self, "lora_patches", [])
                if lora_patches:
                    for patch_data in lora_patches:
                        # Unpack patch data (supports both old and new format)
                        if len(patch_data) == 3:
                            # Old format: (down, up, alpha)
                            down, up, alpha = patch_data
                        else:
                            # New format: (down, up, alpha, down_scale, up_scale)
                            down, up, alpha, _, _ = patch_data
                        
                        # Conv2d LoRA is typically float (INT8 not commonly used for Conv)
                        d = down.to(device=out.device, dtype=out.dtype)
                        u = up.to(device=out.device, dtype=out.dtype)
                        # For Conv2d, LoRA is usually applied as 1x1 convolutions or similar
                        # but most LoRAs for these models don't target Conv2d.
                        # If they do, they are usually reshaped.
                        # This is a simple implementation that assumes Linear-like LoRA
                        # which might not be correct for all Conv2d LoRAs.
                        # However, Wan/Flux LoRAs are almost exclusively Linear.
                        try:
                            lora_out = F.conv2d(F.conv2d(x, d), u)
                            out.add_(lora_out, alpha=alpha)
                        except Exception:
                            pass
                        del d, u
                return out
        
        class Conv3d(manual_cast.Conv3d):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.lora_patches = []
        
        class ConvTranspose2d(manual_cast.ConvTranspose2d):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.lora_patches = []
        
        class Embedding(manual_cast.Embedding):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.lora_patches = []
        
        @classmethod
        def conv_nd(cls, dims, *args, **kwargs):
            if dims == 2:
                return cls.Conv2d(*args, **kwargs)
            elif dims == 3:
                return cls.Conv3d(*args, **kwargs)
            else:
                raise ValueError(f"unsupported dimensions: {dims}")
