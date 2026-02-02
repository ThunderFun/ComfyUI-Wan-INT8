import os
import copy
import torch
import torch.nn.functional as F
import folder_paths
import comfy.utils
import gc
import threading
import hashlib
import math
from .lora_utils import parse_wan_lora

_object_patches_lock = threading.RLock()

_tensor_hash_lock = threading.Lock()

def _parse_env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        print(f"Warning: Invalid value for {name}, using default: {default}")
        return default

def _parse_env_bool(name, default="0"):
    val = os.environ.get(name, default)
    return val == "1"

_ENABLE_CUDA_SYNC = _parse_env_bool("INT8_ENABLE_CUDA_SYNC", "0")
_CLEAR_CACHE_STRATEGY = os.environ.get("INT8_CLEAR_CACHE", "auto")
_MAX_LORA_CACHE_SIZE = _parse_env_int("INT8_LORA_CACHE_SIZE", 32)

_MAX_FULL_HASH_ELEMENTS = 64

_HASH_SAMPLE_COUNT = 16

def _should_clear_cache():
    if _CLEAR_CACHE_STRATEGY == "always":
        return True
    if _CLEAR_CACHE_STRATEGY == "never":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        if reserved > 0:
            utilization = allocated / reserved
            return utilization < 0.8
        if allocated > 0:
            return False
    except Exception:
        pass
    return True


def _get_tensor_content_hash(t, full_hash_threshold=1024*1024):
    """Get a stable hash for tensor identity based on content."""
    if not isinstance(t, torch.Tensor):
        return (id(t),)

    shape_str = str(t.shape)
    dtype_str = str(t.dtype)
    numel = t.numel()
    
    if numel == 0:
        return (shape_str, dtype_str, 0)

    if _ENABLE_CUDA_SYNC and torch.cuda.is_available() and t.is_cuda:
        torch.cuda.synchronize(t.device)

    with _tensor_hash_lock:
        try:
            flat = t.flatten().detach()
            tensor_bytes = numel * t.element_size()
            
            if tensor_bytes <= full_hash_threshold:
                cpu_tensor = flat.cpu()
                content_bytes = cpu_tensor.numpy().tobytes()
                content_hash = hashlib.sha256(content_bytes).hexdigest()
                return (shape_str, dtype_str, numel, content_hash)
            else:
                num_samples = min(256, numel)
                
                bucket_size = numel // num_samples
                indices = []
                for i in range(num_samples):
                    idx = i * bucket_size + bucket_size // 2
                    indices.append(min(idx, numel - 1))
                
                indices.extend([0, numel - 1, numel // 2, numel // 4, 3 * numel // 4])
                indices = sorted(set(indices))
                
                sample = flat[indices].cpu()
                sample_bytes = sample.numpy().tobytes()
                sample_hash = hashlib.sha256(sample_bytes).hexdigest()
                
                flat_f64 = flat.to(torch.float64)
                tensor_sum = float(flat_f64.sum().item())
                tensor_abs_sum = float(flat_f64.abs().sum().item())
                tensor_min = float(flat_f64.min().item())
                tensor_max = float(flat_f64.max().item())
                
                return (shape_str, dtype_str, numel, sample_hash,
                        tensor_sum, tensor_abs_sum, tensor_min, tensor_max)
                
        except Exception as e:
            return (shape_str, dtype_str, numel, id(t), "fallback", str(type(e).__name__))


def _get_patch_identity(patches):
    """Get a stable identity for patches list that avoids id() reuse issues."""
    if not patches:
        return ()
    
    try:
        patches_snapshot = tuple(patches)
    except (RuntimeError, TypeError):
        import time
        return ("unstable", id(patches), time.time())
    
    identities = []
    for patch in patches_snapshot:
        if isinstance(patch, (list, tuple)) and len(patch) >= 2:
            d, u = patch[0], patch[1]
            patch_id = (
                _get_tensor_content_hash(d),
                _get_tensor_content_hash(u),
                len(patch)
            )
            if len(patch) >= 3:
                alpha = patch[2]
                if isinstance(alpha, (int, float)):
                    patch_id = patch_id + (alpha,)
                elif isinstance(alpha, torch.Tensor):
                    patch_id = patch_id + (float(alpha.item()),)
            identities.append(patch_id)
        else:
            identities.append((type(patch).__name__, id(patch)))
    return tuple(identities)


def _is_valid_alpha(a):
    """Check if alpha value is valid (not None, NaN, or zero)."""
    if a is None:
        return False
    
    if isinstance(a, torch.Tensor):
        if a.numel() == 0:
            return False
        try:
            a_val = a.item()
            return not (math.isnan(a_val) or a_val == 0)
        except:
            return False
    
    if isinstance(a, (int, float)):
        return not (a != a or a == 0)
    
    return False


def _is_zero_alpha(a):
    """Check if alpha is exactly zero (for warning suppression)."""
    if a is None:
        return False
    
    if isinstance(a, torch.Tensor):
        try:
            return a.item() == 0
        except:
            return False
    
    return a == 0


def _identify_qkv_component(key: str) -> tuple:
    """Identify which QKV component a LoRA key targets."""
    key_lower = key.lower()
    
    fused_patterns = ['.qkv', '_qkv', '.to_qkv', '_to_qkv']
    for pattern in fused_patterns:
        if pattern in key_lower:
            return (None, True)
    
    q_patterns = ['.to_q.', '.q_proj.', '_q.', '.q.', '_attn_q', '/q/']
    k_patterns = ['.to_k.', '.k_proj.', '_k.', '.k.', '_attn_k', '/k/']
    v_patterns = ['.to_v.', '.v_proj.', '_v.', '.v.', '_attn_v', '/v/']
    
    for pattern in q_patterns:
        if pattern in key_lower or key_lower.endswith(pattern.rstrip('.')):
            return ('q', False)
    
    for pattern in k_patterns:
        if pattern in key_lower or key_lower.endswith(pattern.rstrip('.')):
            return ('k', False)
    
    for pattern in v_patterns:
        if pattern in key_lower or key_lower.endswith(pattern.rstrip('.')):
            return ('v', False)
    
    return (None, False)


def _validate_lora_bounds(offset, size, out_dim, key=""):
    """Validate and adjust LoRA patch bounds to fit output tensor."""
    if offset < 0:
        print(f"Warning: Negative offset {offset} for {key}, clamping to 0")
        offset = 0
    
    if size < 0:
        print(f"Warning: Negative size {size} for {key}, treating as full size")
        size = 0
    
    if size == 0:
        size = out_dim - offset
    
    if offset >= out_dim:
        print(f"Warning: Offset {offset} >= output dim {out_dim} for {key}, skipping")
        return 0, 0, False
    
    if offset + size > out_dim:
        old_size = size
        size = out_dim - offset
        print(f"Warning: Clamping size from {old_size} to {size} for {key}")
    
    if size <= 0:
        return 0, 0, False
    
    return offset, size, True


class LoRAWeightCache:
    """Thread-safe LRU cache for LoRA weights with reference counting."""

    def __init__(self, max_size=_MAX_LORA_CACHE_SIZE):
        self.max_size = max_size
        self._cache = {}
        self._access_order = []
        self._lock = threading.RLock()
        self._ref_counts = {}
        self._total_memory_bytes = 0
        # NEW: Maximum memory limit for cache (default 2GB, configurable)
        self._max_memory_bytes = _parse_env_int("INT8_LORA_CACHE_MAX_MB", 2048) * 1024 * 1024

    def get(self, key):
        """Get cached value. Returns a deep copy of tensor values."""
        with self._lock:
            if key in self._cache:
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
                self._access_order.append(key)
                self._ref_counts[key] = self._ref_counts.get(key, 0) + 1
                value = self._cache[key]
                if isinstance(value, dict):
                    result = {}
                    for k, v in value.items():
                        if isinstance(v, torch.Tensor):
                            result[k] = v.detach().clone().contiguous()
                        else:
                            result[k] = v
                    return result
                elif isinstance(value, torch.Tensor):
                    return value.detach().clone().contiguous()
                return value
            return None

    def get_or_create(self, key, factory_fn):
        """Atomically get existing value or create and cache new one."""
        with self._lock:
            if key in self._cache:
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
                self._access_order.append(key)
                self._ref_counts[key] = self._ref_counts.get(key, 0) + 1
                
                value = self._cache[key]
                if isinstance(value, dict):
                    result = {}
                    for k, v in value.items():
                        if isinstance(v, torch.Tensor):
                            result[k] = v.detach().clone().contiguous()
                        else:
                            result[k] = v
                    return (result, False)
                elif isinstance(value, torch.Tensor):
                    return (value.detach().clone().contiguous(), False)
                return (value, False)
            
            if len(self._cache) >= self.max_size:
                self._evict_oldest()
            
            value = factory_fn()
            
            self._cache[key] = value
            self._access_order.append(key)
            self._ref_counts[key] = 1
            
            if isinstance(value, dict):
                result = {}
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        result[k] = v.detach().clone().contiguous()
                    else:
                        result[k] = v
                return (result, True)
            elif isinstance(value, torch.Tensor):
                return (value.detach().clone().contiguous(), True)
            return (value, True)

    def release(self, key):
        """Release a reference obtained via get()."""
        with self._lock:
            if key in self._ref_counts:
                self._ref_counts[key] -= 1
                if self._ref_counts[key] <= 0:
                    del self._ref_counts[key]

    def set(self, key, value, initial_ref_count=0):
        """Set cached value with optional initial reference count."""
        with self._lock:
            # NEW: Calculate memory usage of value
            value_memory = self._estimate_memory(value)
            
            # NEW: Evict if memory limit would be exceeded
            while self._total_memory_bytes + value_memory > self._max_memory_bytes and self._cache:
                self._evict_oldest()
            
            if key in self._cache:
                # Subtract old value's memory
                old_value = self._cache[key]
                self._total_memory_bytes -= self._estimate_memory(old_value)
                
                self._cache[key] = value
                self._total_memory_bytes += value_memory
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
                self._access_order.append(key)
                if initial_ref_count > 0:
                    self._ref_counts[key] = self._ref_counts.get(key, 0) + initial_ref_count
            else:
                if len(self._cache) >= self.max_size:
                    self._evict_oldest()
                self._cache[key] = value
                self._total_memory_bytes += value_memory
                self._access_order.append(key)
                if initial_ref_count > 0:
                    self._ref_counts[key] = initial_ref_count

    def _evict_oldest(self):
        """Evict oldest entry with zero reference count."""
        evicted = False
        for i, key in enumerate(list(self._access_order)):
            if self._ref_counts.get(key, 0) == 0:
                self._access_order.remove(key)
                if key in self._cache:
                    value = self._cache.pop(key)
                    # NEW: Update memory tracking
                    self._total_memory_bytes -= self._estimate_memory(value)
                    self._release_value(value)
                evicted = True
                break
        
        if not evicted and len(self._cache) >= self.max_size:
            # NEW: Force evict oldest even if in use when memory critical
            if self._access_order:
                key = self._access_order.pop(0)
                if key in self._cache:
                    value = self._cache.pop(key)
                    self._total_memory_bytes -= self._estimate_memory(value)
                    self._release_value(value)
                self._ref_counts.pop(key, None)

    def _estimate_memory(self, value):
        """Estimate memory usage of a cached value."""
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        elif isinstance(value, dict):
            total = 0
            for v in value.values():
                if isinstance(v, torch.Tensor):
                    total += v.numel() * v.element_size()
            return total
        return 0

    def _release_value(self, value):
        """Release value references when evicting from cache."""
        pass

    def clear(self):
        """Clear cache entries with zero reference count."""
        with self._lock:
            keys_to_evict = []
            for key in self._access_order:
                if self._ref_counts.get(key, 0) == 0:
                    keys_to_evict.append(key)
            
            for key in keys_to_evict:
                if key in self._cache:
                    value = self._cache.pop(key)
                    self._total_memory_bytes -= self._estimate_memory(value)
                    self._release_value(value)
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
                self._ref_counts.pop(key, None)


class LoRAWrapperModule(torch.nn.Module):
    def __init__(self, wrapped_module, lora_patches_list):
        super().__init__()
        if lora_patches_list is not None and not isinstance(lora_patches_list, (list, tuple)):
            raise TypeError(f"lora_patches_list must be a list or tuple, got {type(lora_patches_list).__name__}")
        
        self._getting_attr = threading.local()
        self._getattr_lock = threading.Lock()
        
        # Store as a submodule so PyTorch can find it
        self._wrapped_module = wrapped_module
        self.lora_patches = lora_patches_list if lora_patches_list is not None else []
        self._lora_weight_cache = LoRAWeightCache(_MAX_LORA_CACHE_SIZE)
        self._last_patch_ids = None
        self._patch_ids_lock = threading.Lock()

    def __setstate__(self, state):
        """Restore object state from pickle, handling backwards compatibility."""
        # Handle old pickles that have 'wrapped_module' instead of '_wrapped_module'
        if 'wrapped_module' in state and '_wrapped_module' not in state:
            state['_wrapped_module'] = state.pop('wrapped_module')
        # Handle case where neither exists (shouldn't happen, but be safe)
        if '_wrapped_module' not in state:
            state['_wrapped_module'] = None
        # Restore threading locals and locks (can't be pickled)
        state['_getting_attr'] = threading.local()
        state['_getattr_lock'] = threading.Lock()
        state['_patch_ids_lock'] = threading.Lock()
        # Restore the cache
        state['_lora_weight_cache'] = LoRAWeightCache(_MAX_LORA_CACHE_SIZE)
        self.__dict__.update(state)

    @property
    def wrapped_module(self):
        # Backwards compatibility: check both old and new attribute names
        if '_wrapped_module' in self.__dict__:
            return self.__dict__['_wrapped_module']
        if 'wrapped_module' in self.__dict__:
            return self.__dict__['wrapped_module']
        raise AttributeError("wrapped_module not set")
    
    @wrapped_module.setter
    def wrapped_module(self, value):
        self._wrapped_module = value

    @property
    def weight(self):
        """Expose weight directly for ComfyUI compatibility."""
        wrapped = self.wrapped_module  # Use property for backwards compatibility
        if wrapped is not None and hasattr(wrapped, 'weight'):
            return wrapped.weight
        raise AttributeError("wrapped module has no weight")

    @property
    def bias(self):
        """Expose bias directly for ComfyUI compatibility."""
        wrapped = self.wrapped_module  # Use property for backwards compatibility
        if wrapped is not None and hasattr(wrapped, 'bias'):
            return wrapped.bias
        return None

    def __deepcopy__(self, memo):
        """Custom deepcopy that handles threading.local() and locks."""
        new_instance = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_instance
        
        # Use property for backwards compatibility with old objects
        wrapped = self.wrapped_module
        if wrapped is not None:
            new_instance._wrapped_module = copy.deepcopy(wrapped, memo)
        else:
            new_instance._wrapped_module = None
        
        new_instance.lora_patches = copy.deepcopy(self.lora_patches, memo)
        
        new_instance._getting_attr = threading.local()
        new_instance._getattr_lock = threading.Lock()
        new_instance._patch_ids_lock = threading.Lock()
        
        new_instance._lora_weight_cache = LoRAWeightCache(_MAX_LORA_CACHE_SIZE)
        new_instance._last_patch_ids = self._last_patch_ids
        
        return new_instance

    def _is_getting_attr(self, name):
        """Check if we're already getting this attribute (per-thread)."""
        try:
            return name in self._getting_attr.set
        except AttributeError:
            return False

    def _add_getting_attr(self, name):
        """Add to getting_attr set (per-thread)."""
        try:
            self._getting_attr.set.add(name)
        except AttributeError:
            self._getting_attr.set = {name}

    def _remove_getting_attr(self, name):
        """Remove from getting_attr set (per-thread)."""
        try:
            self._getting_attr.set.discard(name)
        except AttributeError:
            pass

    def __getattr__(self, name):
        # Handle access to wrapped_module attributes when they don't exist (backwards compatibility)
        if name in ('_wrapped_module', 'wrapped_module'):
            # Check all possible locations
            if '_wrapped_module' in self.__dict__:
                return self.__dict__['_wrapped_module']
            if 'wrapped_module' in self.__dict__:
                return self.__dict__['wrapped_module']
            # Check if stored as a submodule by PyTorch
            if '_modules' in self.__dict__:
                if 'wrapped_module' in self._modules:
                    return self._modules['wrapped_module']
                if '_wrapped_module' in self._modules:
                    return self._modules['_wrapped_module']
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Check if we have _getting_attr initialized
        try:
            getting_attr = object.__getattribute__(self, '_getting_attr')
        except AttributeError:
            # Fallback during initialization
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        if self._is_getting_attr(name):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}' (recursion detected)")
        
        self._add_getting_attr(name)
        try:
            # Try to get from wrapped module (check both old and new attribute names)
            wrapped = None
            if '_wrapped_module' in self.__dict__:
                wrapped = self.__dict__['_wrapped_module']
            elif 'wrapped_module' in self.__dict__:
                wrapped = self.__dict__['wrapped_module']
            
            if wrapped is not None:
                return getattr(wrapped, name)
            
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        finally:
            self._remove_getting_attr(name)

    def __delattr__(self, name):
        """Safe attribute deletion with protected attributes."""
        _protected_attrs = {'_getting_attr', '_getattr_lock', '_patch_ids_lock',
                            '_wrapped_module', 'lora_patches', '_lora_weight_cache'}
        
        if name in _protected_attrs:
            raise AttributeError(f"Cannot delete protected attribute '{name}'")
        
        if name in self.__dict__:
            object.__delattr__(self, name)
            return
        
        if name in self._modules:
            del self._modules[name]
            return
        if name in self._parameters:
            del self._parameters[name]
            return
        if name in self._buffers:
            del self._buffers[name]
            return
        
        # Use property for backwards compatibility
        try:
            wrapped = self.wrapped_module
        except AttributeError:
            wrapped = None
        if wrapped is not None:
            try:
                delattr(wrapped, name)
                return
            except AttributeError:
                pass
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def forward(self, x):
        # Use property for backwards compatibility with old objects
        wrapped = self.wrapped_module
        original_out = wrapped(x)
        
        patches = self.lora_patches
        if not patches:
            return original_out

        try:
            from .int8_quant import chunked_int8_lora_forward, chunked_lora_forward, CHUNK_THRESHOLD_ELEMENTS
        except ImportError as e:
            print(f"ERROR: Failed to import int8_quant module: {e}. LoRA will not be applied.")
            return original_out

        with self._patch_ids_lock:
            current_patch_ids = _get_patch_identity(patches)
            if self._last_patch_ids != current_patch_ids:
                self._lora_weight_cache.clear()
            self._last_patch_ids = current_patch_ids

        x_shape = x.shape
        x_2d = x.reshape(-1, x_shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()

        out = None
        keys_to_release = set()
        any_patch_applied = False
        
        # NEW: Track intermediate tensors for cleanup
        tensors_to_cleanup = []
        
        try:
            for patch_data in patches:
                if not isinstance(patch_data, (list, tuple)):
                    print(f"Warning: Invalid patch_data type: {type(patch_data)}")
                    continue

                patch_len = len(patch_data)
                if patch_len == 3:
                    d, u, a = patch_data
                    d_scale, u_scale = None, None
                    offset, size = 0, 0
                elif patch_len == 5:
                    d, u, a, d_scale, u_scale = patch_data
                    offset, size = 0, 0
                elif patch_len == 7:
                    d, u, a, d_scale, u_scale, offset, size = patch_data
                else:
                    print(f"Warning: Unexpected patch_data length: {patch_len}")
                    continue

                if not all(isinstance(t, torch.Tensor) for t in [d, u]):
                    print(f"Warning: Expected d and u to be tensors")
                    continue

                if not _is_valid_alpha(a):
                    if not _is_zero_alpha(a):
                        print(f"Warning: Invalid alpha value ({a}), skipping patch")
                    continue

                is_int8 = d.dtype == torch.int8 and u.dtype == torch.int8
                expected_input_dim = d.shape[1]
                
                if expected_input_dim != x_2d.shape[-1]:
                    print(f"Warning: Shape mismatch: d.shape={d.shape}, x_2d.shape={x_2d.shape}")
                    continue

                offset, size, is_valid = _validate_lora_bounds(
                    offset, size, original_out.shape[-1],
                    key=f"patch_{id(patch_data)}"
                )
                if not is_valid:
                    continue

                if out is None:
                    out = original_out.clone()
                
                try:
                    if is_int8 and d_scale is not None and u_scale is not None:
                        cache_key = (
                            "int8",
                            _get_tensor_content_hash(d), _get_tensor_content_hash(u),
                            _get_tensor_content_hash(d_scale), _get_tensor_content_hash(u_scale),
                            offset, size, original_out.shape[-1],
                        )
                        
                        def create_int8_cache():
                            return {
                                'd': d.to(device=original_out.device, non_blocking=False),
                                'u': u.to(device=original_out.device, non_blocking=False),
                                'd_scale': d_scale.to(device=original_out.device, non_blocking=False) if isinstance(d_scale, torch.Tensor) else d_scale,
                                'u_scale': u_scale.to(device=original_out.device, non_blocking=False) if isinstance(u_scale, torch.Tensor) else u_scale,
                            }

                        cached, was_created = self._lora_weight_cache.get_or_create(cache_key, create_int8_cache)
                        keys_to_release.add(cache_key)

                        chunked_int8_lora_forward(
                            x_2d, cached['d'], cached['u'],
                            cached['d_scale'], cached['u_scale'],
                            a, out, offset=offset, size=size
                        )
                        any_patch_applied = True
                        
                        # NEW: Clean up cached tensors after use if memory is tight
                        if was_created and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        curr_x = x_2d.to(dtype=original_out.dtype) if x_2d.dtype != original_out.dtype else x_2d
                        if curr_x is not x_2d:
                            tensors_to_cleanup.append(curr_x)

                        cache_key = (
                            "float",
                            _get_tensor_content_hash(d), _get_tensor_content_hash(u),
                            original_out.dtype, offset, size, original_out.shape[-1],
                        )

                        def create_float_cache():
                            return {
                                'd': d.to(device=original_out.device, dtype=original_out.dtype, non_blocking=False),
                                'u': u.to(device=original_out.device, dtype=original_out.dtype, non_blocking=False),
                            }

                        cached, was_created = self._lora_weight_cache.get_or_create(cache_key, create_float_cache)
                        keys_to_release.add(cache_key)

                        chunked_lora_forward(
                            curr_x, cached['d'], cached['u'],
                            a, out, offset=offset, size=size
                        )
                        any_patch_applied = True
                        
                        # NEW: Clean up cached tensors after use if memory is tight
                        if was_created and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                except Exception as e:
                    print(f"Error computing LoRA patch contribution: {e}")
                    continue

            if out is not None and any_patch_applied:
                return out
            else:
                if out is not None:
                    del out
                return original_out

        finally:
            # NEW: Clean up intermediate tensors
            for t in tensors_to_cleanup:
                del t
            tensors_to_cleanup.clear()
            
            for key in keys_to_release:
                try:
                    self._lora_weight_cache.release(key)
                except Exception:
                    pass
            
            # NEW: Periodic cache cleanup to prevent memory buildup
            if torch.cuda.is_available():
                try:
                    allocated = torch.cuda.memory_allocated()
                    reserved = torch.cuda.memory_reserved()
                    if reserved > 0 and allocated / reserved < 0.5:
                        torch.cuda.empty_cache()
                except Exception:
                    pass


class WanLoRALoaderWithCLIP:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "offload_to_cpu": (["enable", "disable"], {"default": "disable"}),
            },
            "optional": {
                "clip": ("CLIP",),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "WanVideo/INT8"

    def load_lora(self, model, lora_name, strength_model, strength_clip, clip=None, offload_to_cpu="disable", debug=False):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            print(f"LoRA not found: {lora_name}")
            return (model, clip)

        print(f"Loading LoRA: {lora_name}")
        offload_enabled = (offload_to_cpu == "enable")

        lora_state_dict = None
        lora_weights = None
        modules = None
        
        try:
            lora_state_dict = comfy.utils.load_torch_file(lora_path)
            lora_weights = parse_wan_lora(lora_state_dict, 1.0, debug=debug)
            
            if lora_state_dict is not None:
                del lora_state_dict
                lora_state_dict = None
            gc.collect()

            if lora_weights is None:
                print(f"Failed to parse LoRA: {lora_name}")
                return (model, clip)

            new_model = model.clone()
            
            with _object_patches_lock:
                if not hasattr(new_model, "object_patches"):
                    new_model.object_patches = {}
                else:
                    original_patches = new_model.object_patches
                    new_object_patches = {}
                    
                    for key, value in list(original_patches.items()):
                        if isinstance(value, LoRAWrapperModule):
                            value._lora_weight_cache.clear()
                        try:
                            new_object_patches[key] = copy.deepcopy(value)
                        except Exception as e:
                            print(f"Warning: Failed to deepcopy patch {key}: {e}")
                            new_object_patches[key] = value
                    
                    new_model.object_patches = new_object_patches

            torch_model = new_model.model
            modules = dict(torch_model.named_modules())

            patched_count = 0
            
            patches_to_apply = []

            for key in lora_weights.weights:
                # Skip CLIP keys - they're handled separately in patch_patcher
                clip_key_patterns = ["text_encoder", "qwen", "lora_te", "te_model", "clip", "cond_stage_model", "t5", "bert"]
                is_clip_key = any(pattern in key.lower() for pattern in clip_key_patterns)
                if is_clip_key:
                    continue
                # Also skip keys that don't look like diffusion model keys
                if not key.startswith("diffusion_model.") and "diffusion_model." not in key:
                    if not any(pattern in key for pattern in [".block", ".transformer", ".conv", ".norm", ".attn"]):
                        continue

                target_module = None
                target_key = None

                candidates = [key]
                
                if key.startswith("diffusion_model."):
                    candidates.append(key[len("diffusion_model."):])
                else:
                    candidates.append("diffusion_model." + key)

                var_candidates = []
                for c in candidates:
                    if ".self_attn." in c:
                        var_candidates.append(c.replace(".self_attn.", ".attn."))
                    elif ".attn." in c:
                        var_candidates.append(c.replace(".attn.", ".self_attn."))
                    if ".to_out.0" in c:
                        var_candidates.append(c.replace(".to_out.0", ".to_out"))
                    elif ".to_out" in c and ".to_out.0" not in c:
                        var_candidates.append(c.replace(".to_out", ".to_out.0"))

                candidates = candidates + var_candidates

                replacements = [
                    (".to_q", ".q"), (".to_k", ".k"), (".to_v", ".v"), (".to_out", ".o"),
                    (".q_proj", ".qkv"), (".k_proj", ".qkv"), (".v_proj", ".qkv"),
                    (".to_q", ".qkv"), (".to_k", ".qkv"), (".to_v", ".qkv")
                ]

                final_pass_candidates = list(candidates)
                for c in candidates:
                    for old, new in replacements:
                        if old in c:
                            final_pass_candidates.append(c.replace(old, new))

                candidates = final_pass_candidates

                patch_offset = 0
                patch_size = 0

                for cand in candidates:
                    if cand in modules:
                        target_module = modules[cand]
                        target_key = cand

                        if ".qkv" in cand and isinstance(target_module, torch.nn.Linear):
                            total_out = target_module.weight.shape[0]
                            
                            if total_out % 3 != 0:
                                print(f"Warning: Cannot apply LoRA to QKV module {cand}: "
                                      f"output dimension {total_out} is not divisible by 3. "
                                      f"This LoRA patch will be skipped for key: {key}")
                                target_module = None
                                target_key = None
                                break
                            
                            head_dim = total_out // 3
                            
                            is_int8 = lora_weights.is_int8.get(key, False)
                            if is_int8:
                                down, up, down_scale, up_scale, alpha = lora_weights.weights[key]
                            else:
                                down, up, alpha = lora_weights.weights[key]
                            
                            lora_out_dim = up.shape[0] if up is not None and hasattr(up, 'shape') else 0
                            
                            if lora_out_dim == total_out:
                                patch_offset = 0
                                patch_size = 0
                                if debug:
                                    print(f"Full QKV LoRA detected for {key}: out_dim={lora_out_dim}")
                            elif lora_out_dim == head_dim:
                                component, is_fused = _identify_qkv_component(key)
                                
                                if component == 'q':
                                    patch_offset = 0
                                    patch_size = head_dim
                                elif component == 'k':
                                    patch_offset = head_dim
                                    patch_size = head_dim
                                elif component == 'v':
                                    patch_offset = head_dim * 2
                                    patch_size = head_dim
                                else:
                                    print(f"Warning: Could not determine Q/K/V type for partial LoRA key: {key}")
                                    target_module = None
                                    target_key = None
                            elif lora_out_dim > 0:
                                print(f"Warning: LoRA output dimension {lora_out_dim} doesn't match "
                                      f"expected full QKV ({total_out}) or single head ({head_dim}) for {key}")
                                target_module = None
                                target_key = None
                            break

                if target_module is not None and isinstance(target_module, torch.nn.Linear):
                    is_int8 = lora_weights.is_int8.get(key, False)
                    if is_int8:
                        down, up, down_scale, up_scale, alpha = lora_weights.weights[key]
                    else:
                        down, up, alpha = lora_weights.weights[key]
                        down_scale = None
                        up_scale = None

                    if hasattr(target_module, "weight"):
                        mod_out, mod_in = target_module.weight.shape

                        if patch_size > 0:
                            if patch_offset + patch_size > mod_out:
                                if debug:
                                    print(f"Warning: clamping patch for {key}")
                                patch_size = max(0, mod_out - patch_offset)

                        if mod_in != down.shape[1]:
                            if debug:
                                print(f"Skipping {key}: Input mismatch {mod_in} vs {down.shape[1]}")
                            continue

                    patch_tuple = (down, up, alpha, down_scale, up_scale, patch_offset, patch_size)
                    patches_to_apply.append((target_key, target_module, patch_tuple))

            wrappers_to_add = []
            for target_key, target_module, patch_tuple in patches_to_apply:
                current_patches = []
                if target_key in new_model.object_patches:
                    existing_obj = new_model.object_patches[target_key]
                    if hasattr(existing_obj, "lora_patches"):
                        current_patches = list(existing_obj.lora_patches)

                new_patch_list = current_patches + [patch_tuple]

                # Safely determine the raw module to wrap
                raw_module = target_module
                
                # Unwrap if target_module is already a LoRAWrapperModule
                if isinstance(raw_module, LoRAWrapperModule):
                    raw_module = raw_module.wrapped_module  # Use property
                
                # Check object_patches for existing wrapper
                if target_key in new_model.object_patches:
                    existing = new_model.object_patches[target_key]
                    if isinstance(existing, LoRAWrapperModule):
                        raw_module = existing.wrapped_module  # Use property
                    elif isinstance(existing, torch.nn.Linear):
                        raw_module = existing
                
                # Final safety: ensure we're not wrapping a wrapper
                while isinstance(raw_module, LoRAWrapperModule):
                    raw_module = raw_module.wrapped_module  # Use property

                wrapper = LoRAWrapperModule(raw_module, new_patch_list)
                wrappers_to_add.append((target_key, wrapper))

            with _object_patches_lock:
                for target_key, wrapper in wrappers_to_add:
                    new_model.add_object_patch(target_key, wrapper)
                    patched_count += 1

            # PASSTHROUGH: Skip CLIP patching entirely to avoid OOM issues
            # The CLIP model is returned as-is without applying LoRA patches
            new_clip = clip

            print(f"Applied {patched_count} LoRA patches (CLIP passthrough - no LoRA applied to CLIP).")
            
            # CLIP is passed through unchanged (no LoRA patching to avoid OOM)
            return (new_model, clip)
            
        except Exception as e:
            print(f"Error loading LoRA {lora_name}: {e}")
            raise
            
        finally:
            if lora_state_dict is not None:
                del lora_state_dict
            if lora_weights is not None:
                del lora_weights
            if modules is not None:
                del modules
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def patch_patcher(self, patcher, lora_weights, strength, offload_enabled, debug, is_clip=False):
        """Apply LoRA patches to a patcher object."""
        with _object_patches_lock:
            # Ensure object_patches exists
            if not hasattr(patcher, "object_patches"):
                patcher.object_patches = {}
            
            # Ensure add_object_patch method exists
            if not hasattr(patcher, "add_object_patch"):
                def add_object_patch(key, obj):
                    patcher.object_patches[key] = obj
                patcher.add_object_patch = add_object_patch
            
            # Handle different patcher structures for CLIP vs diffusion models
            torch_model = patcher.model
            
            if is_clip:
                # CLIP models may have nested structure or multiple encoders
                modules = {}
                if hasattr(torch_model, 'transformer'):
                    # Single transformer model
                    modules = dict(torch_model.transformer.named_modules())
                    # Add root-level modules too
                    modules.update({f"transformer.{k}": v for k, v in torch_model.transformer.named_modules()})
                elif hasattr(torch_model, 'text_model'):
                    modules = dict(torch_model.text_model.named_modules())
                    modules.update({f"text_model.{k}": v for k, v in torch_model.text_model.named_modules()})
                elif hasattr(torch_model, 'model'):
                    # Nested model structure
                    inner_model = torch_model.model
                    if hasattr(inner_model, 'named_modules'):
                        modules = dict(inner_model.named_modules())
                    else:
                        modules = dict(torch_model.named_modules())
                else:
                    modules = dict(torch_model.named_modules())
                    
                # Also check for multiple text encoders (SDXL style)
                if hasattr(patcher, 'text_encoders'):
                    for enc_name, encoder in patcher.text_encoders.items():
                        if hasattr(encoder, 'named_modules'):
                            enc_modules = dict(encoder.named_modules())
                            modules.update({f"{enc_name}.{k}": v for k, v in enc_modules.items()})
            else:
                if not hasattr(torch_model, 'named_modules'):
                    print(f"Error: Model does not have named_modules method")
                    return
                modules = dict(torch_model.named_modules())

            patched_count = 0

            for key in lora_weights.weights:
                if is_clip and key.startswith("diffusion_model."):
                    continue

                target_module = None
                target_key = None

                candidates = [key]

                if not is_clip:
                    if key.startswith("diffusion_model."):
                        candidates.append(key[len("diffusion_model."):])
                    else:
                        candidates.append("diffusion_model." + key)
                else:
                    # Comprehensive CLIP prefix handling
                    clip_prefixes = [
                        "text_encoders.", "lora_te.", "lora_te1.", "lora_te2.", "te_model.",
                        "clip_l.", "clip_g.", "clip_h.", "cond_stage_model.", "text_encoder.",
                        "text_encoder_1.", "text_encoder_2.", "clip.", "te.", "qwen.",
                        "transformer.text_model.", "text_model."
                    ]
                    for prefix in clip_prefixes:
                        if key.startswith(prefix):
                            stripped = key[len(prefix):]
                            candidates.append(stripped)
                            # Also try with common CLIP internal structure
                            if not stripped.startswith("transformer."):
                                candidates.append("transformer." + stripped)
                            if not stripped.startswith("text_model."):
                                candidates.append("text_model." + stripped)

                transformations = [
                    (".self_attn.", ".attn."), (".attn.", ".self_attn."),
                    (".to_out.0", ".to_out"), (".to_out", ".to_out.0"),
                    (".to_q", ".q"), (".to_k", ".k"), (".to_v", ".v"), (".to_out", ".o"),
                    (".q", ".to_q"), (".k", ".to_k"), (".v", ".to_v"), (".o", ".to_out"),
                    (".to_q", ".qkv"), (".to_k", ".qkv"), (".to_v", ".qkv"),
                    (".to_out.0", ".out"), (".to_out", ".out"),
                    (".q_proj", ".qkv"), (".k_proj", ".qkv"), (".v_proj", ".qkv"),
                ]

                current_candidates = set(candidates)
                for _ in range(3):
                    new_cands = set()
                    for c in current_candidates:
                        for old, new in transformations:
                            if old in c:
                                new_cands.add(c.replace(old, new))
                    if not new_cands:
                        break
                    if new_cands.issubset(current_candidates):
                        break
                    current_candidates.update(new_cands)

                if key.startswith("diffusion_model."):
                    for c in list(current_candidates):
                        if not c.startswith("diffusion_model."):
                            current_candidates.add("diffusion_model." + c)

                candidates = list(current_candidates)

                patch_offset = 0
                patch_size = 0

                seen = set()
                for cand in candidates:
                    if cand in seen:
                        continue
                    seen.add(cand)
                    if cand in modules:
                        target_module = modules[cand]
                        target_key = cand

                        if ".qkv" in cand:
                            total_out = target_module.weight.shape[0]
                            if total_out % 3 == 0:
                                if ".to_q" in key or ".q_proj" in key or "_attn_q" in key:
                                    patch_size = total_out // 3
                                    patch_offset = 0
                                elif ".to_k" in key or ".k_proj" in key or "_attn_k" in key:
                                    patch_size = total_out // 3
                                    patch_offset = patch_size
                                elif ".to_v" in key or ".v_proj" in key or "_attn_v" in key:
                                    patch_size = total_out // 3
                                    patch_offset = patch_size * 2
                            else:
                                if debug:
                                    print(f"Warning: QKV module {cand} has non-uniform dimensions: {total_out}")
                                target_module = None
                                target_key = None
                        break

                if is_clip and target_module is None:
                    for cand in candidates:
                        if "_" in cand:
                            dot_cand = cand.replace("_", ".")
                            if dot_cand in modules:
                                target_module = modules[dot_cand]
                                target_key = dot_cand
                                break

                if target_module is not None and isinstance(target_module, torch.nn.Linear):
                    is_int8 = lora_weights.is_int8.get(key, False)
                    if is_int8:
                        down, up, down_scale, up_scale, alpha = lora_weights.weights[key]
                    else:
                        down, up, alpha = lora_weights.weights[key]
                        down_scale = None
                        up_scale = None

                    alpha = alpha * strength

                    if hasattr(target_module, "weight"):
                        expected_out, expected_in = target_module.weight.shape
                        actual_out, actual_rank = up.shape
                        actual_rank_down, actual_in = down.shape

                        validation_out = patch_size if patch_size > 0 else expected_out

                        if validation_out != actual_out or expected_in != actual_in:
                            if debug:
                                print(f"  [!] Dimension mismatch for {target_key}: Model {validation_out}x{expected_in}, LoRA {actual_out}x{actual_in}")
                            continue

                    device = getattr(patcher, "load_device", torch.device("cpu"))

                    if is_int8 or not offload_enabled:
                        down = down.to(device=device, non_blocking=True)
                        up = up.to(device=device, non_blocking=True)
                        if isinstance(down_scale, torch.Tensor):
                            down_scale = down_scale.to(device=device, non_blocking=True)
                        if isinstance(up_scale, torch.Tensor):
                            up_scale = up_scale.to(device=device, non_blocking=True)

                    if patch_size > 0 and hasattr(target_module, "weight"):
                        expected_out = target_module.weight.shape[0]
                        if patch_offset + patch_size > expected_out:
                            patch_size = max(0, expected_out - patch_offset)
                            if patch_size <= 0:
                                continue

                    if patch_size <= 0 and patch_offset > 0:
                        continue

                    if is_int8:
                        patch_tuple = (down, up, alpha, down_scale, up_scale, patch_offset, patch_size)
                    else:
                        patch_tuple = (down, up, alpha, down_scale, up_scale, patch_offset, patch_size)

                    current_patches = []
                    if target_key in patcher.object_patches:
                        existing_obj = patcher.object_patches[target_key]
                        if hasattr(existing_obj, "lora_patches"):
                            current_patches = list(existing_obj.lora_patches)

                    new_patch_list = current_patches + [patch_tuple]

                    # Safely determine the raw module to wrap
                    raw_module = target_module
                    
                    # Unwrap if target_module is already a LoRAWrapperModule
                    if isinstance(raw_module, LoRAWrapperModule):
                        raw_module = raw_module.wrapped_module  # Use property
                    
                    # Check object_patches for existing wrapper
                    if target_key in patcher.object_patches:
                        existing = patcher.object_patches[target_key]
                        if isinstance(existing, LoRAWrapperModule):
                            raw_module = existing.wrapped_module  # Use property
                        elif isinstance(existing, torch.nn.Linear):
                            raw_module = existing
                    
                    # Final safety: ensure we're not wrapping a wrapper
                    while isinstance(raw_module, LoRAWrapperModule):
                        raw_module = raw_module.wrapped_module  # Use property

                    wrapper = LoRAWrapperModule(raw_module, new_patch_list)
                    patcher.add_object_patch(target_key, wrapper)
                    patched_count += 1

            type_name = "CLIP" if is_clip else "Model"
            print(f"{type_name} LoRA: Applied {patched_count} patches.")
