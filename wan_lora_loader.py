import torch
import torch.nn.functional as F
import folder_paths
import comfy.utils
import gc
from .lora_utils import parse_wan_lora
from .int8_quant import Int8TensorwiseOps

class WanLoRALoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "offload_to_cpu": (["enable", "disable"], {"default": "disable"}),
            },
            "optional": {
                "debug": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "WanVideo/INT8"
    
    def load_lora(self, model, lora_name, strength, offload_to_cpu="disable", debug=False):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if strength == 0:
            return (model,)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            print(f"LoRA not found: {lora_name}")
            return (model,)

        print(f"Loading LoRA: {lora_name}")
        
        # Determine if we should offload float LoRAs to CPU
        offload_enabled = (offload_to_cpu == "enable")
        
        lora_state_dict = comfy.utils.load_torch_file(lora_path)

        # Parse LoRA weights and map to model keys
        lora_weights = parse_wan_lora(lora_state_dict, strength, debug=debug)
        
        # Clear state dict immediately to save memory
        del lora_state_dict
        gc.collect()
        
        # Clone model to avoid mutating the original patcher
        new_model = model.clone()
        
        # Get the underlying torch model
        # In ComfyUI, model.model is the BaseModel
        torch_model = new_model.model
        
        # Map of module name -> module
        modules = dict(torch_model.named_modules())
        
        if debug:
            print(f"\n[DEBUG] Available model modules ({len(modules)}):")
            linear_modules = [k for k, v in modules.items() if isinstance(v, torch.nn.Linear)]
            print(f"[DEBUG] Linear modules: {len(linear_modules)}")
            for i, key in enumerate(sorted(linear_modules)[:20]):  # Show first 20 Linear modules
                mod = modules[key]
                print(f"  {i+1}. {key} ({mod.weight.shape[0]}x{mod.weight.shape[1]})")
            if len(linear_modules) > 20:
                print(f"  ... and {len(linear_modules) - 20} more Linear modules")
        
        patched_count = 0
        failed_count = 0
        
        failed_keys = []  # Track which keys failed
        dim_mismatch_count = 0
        
        for key in lora_weights.weights:
            target_module = None
            target_key = None
            
            if debug:
                print(f"\n[DEBUG] Processing LoRA key: {key}")
            
            # Generate candidate keys
            candidates = [key]
            
            # 1. Prefix variations
            if key.startswith("diffusion_model."):
                candidates.append(key[len("diffusion_model."):])
            else:
                candidates.append("diffusion_model." + key)
            
            # 2. Attn variations
            new_candidates = []
            for c in candidates:
                if ".self_attn." in c:
                    new_candidates.append(c.replace(".self_attn.", ".attn."))
                elif ".attn." in c:
                    new_candidates.append(c.replace(".attn.", ".self_attn."))
            candidates.extend(new_candidates)
            
            # 3. to_out variations
            new_candidates = []
            for c in candidates:
                if ".to_out.0" in c:
                    new_candidates.append(c.replace(".to_out.0", ".to_out"))
                elif ".to_out" in c and ".to_out.0" not in c:
                    new_candidates.append(c.replace(".to_out", ".to_out.0"))
            candidates.extend(new_candidates)
            
            # 4. q/k/v/o variations
            new_candidates = []
            replacements = {
                ".to_q": ".q", ".to_k": ".k", ".to_v": ".v", ".to_out": ".o",
                ".q": ".to_q", ".k": ".to_k", ".v": ".to_v", ".o": ".to_out"
            }
            for c in candidates:
                for old, new in replacements.items():
                    if old in c:
                        new_candidates.append(c.replace(old, new))
            candidates.extend(new_candidates)
            
            if debug:
                print(f"[DEBUG] Generated {len(candidates)} candidates")
            
            # 5. Remove duplicates and try matching
            seen = set()
            for cand in candidates:
                if cand in seen: continue
                seen.add(cand)
                if cand in modules:
                    target_module = modules[cand]
                    target_key = cand
                    if debug:
                        print(f"[DEBUG] ✓ Matched to model key: {target_key}")
                    break

            if target_module is not None:
                # If module doesn't have lora_patches, try to add it and patch forward
                if not hasattr(target_module, "lora_patches"):
                    if isinstance(target_module, torch.nn.Linear):
                        target_module.lora_patches = []
                        
                        # Patch forward method to support lora_patches
                        original_forward = target_module.forward
                        def make_patched_forward(mod, orig_fwd):
                            def patched_forward(x):
                                out = orig_fwd(x)
                                patches = getattr(mod, "lora_patches", [])
                                if patches:
                                    for patch_data in patches:
                                        # Unpack patch data (supports both old and new format)
                                        if len(patch_data) == 3:
                                            # Old format: (down, up, alpha)
                                            d, u, a = patch_data
                                            d_scale, u_scale = None, None
                                        else:
                                            # New format: (down, up, alpha, down_scale, up_scale)
                                            d, u, a, d_scale, u_scale = patch_data
                                        
                                        # Check if this is INT8 LoRA
                                        is_int8 = d.dtype == torch.int8 and u.dtype == torch.int8
                                        
                                        if is_int8 and d_scale is not None and u_scale is not None:
                                            # INT8 LoRA path using torch._int_mm
                                            from .int8_quant import chunked_int8_lora_forward, CHUNK_THRESHOLD_ELEMENTS
                                            
                                            # Flatten x to 2D for matmul
                                            x_shape = x.shape
                                            x_2d = x.reshape(-1, x_shape[-1])
                                            
                                            # Move INT8 weights and scales to device (should be cached)
                                            d_int8 = d.to(device=out.device, non_blocking=True)
                                            u_int8 = u.to(device=out.device, non_blocking=True)
                                            d_scale_t = d_scale.to(device=out.device, non_blocking=True) if isinstance(d_scale, torch.Tensor) else d_scale
                                            u_scale_t = u_scale.to(device=out.device, non_blocking=True) if isinstance(u_scale, torch.Tensor) else u_scale
                                            
                                            # Use memory-efficient chunked forward
                                            chunked_int8_lora_forward(
                                                x_2d, d_int8, u_int8, 
                                                d_scale_t, u_scale_t, 
                                                a, out
                                            )
                                            
                                            del d_int8, u_int8, d_scale_t, u_scale_t
                                            
                                            # Force memory release after each patch if it's large
                                            if out.numel() > CHUNK_THRESHOLD_ELEMENTS:
                                                torch.cuda.synchronize()
                                                torch.cuda.empty_cache()
                                        else:
                                            # Float LoRA path
                                            from .int8_quant import chunked_lora_forward
                                            d_t = d.to(device=out.device, dtype=out.dtype)
                                            u_t = u.to(device=out.device, dtype=out.dtype)
                                            
                                            # Use memory-efficient chunked forward
                                            chunked_lora_forward(x, d_t, u_t, a, out)
                                            
                                            del d_t, u_t
                                return out
                            return patched_forward
                        
                        target_module.forward = make_patched_forward(target_module, original_forward)

                if hasattr(target_module, "lora_patches"):
                    # Check if this is INT8 LoRA
                    is_int8 = lora_weights.is_int8.get(key, False)
                    
                    if is_int8:
                        # INT8 LoRA: (down, up, down_scale, up_scale, alpha)
                        down, up, down_scale, up_scale, alpha = lora_weights.weights[key]
                    else:
                        # Float LoRA: (down, up, alpha)
                        down, up, alpha = lora_weights.weights[key]
                        down_scale = None
                        up_scale = None
                    
                    # Dimension validation
                    # down: (rank, in_features), up: (out_features, rank)
                    # Linear weight: (out_features, in_features)
                    if hasattr(target_module, "weight"):
                        expected_out, expected_in = target_module.weight.shape
                        actual_out, actual_rank = up.shape
                        actual_rank_down, actual_in = down.shape
                        
                        if expected_out != actual_out or expected_in != actual_in:
                            print(f"  [!] Dimension mismatch for {target_key}:")
                            print(f"      Model: {expected_out}x{expected_in}")
                            print(f"      LoRA:  {actual_out}x{actual_in} (rank {actual_rank})")
                            dim_mismatch_count += 1
                            continue
    
                    # Get current patches from the module (might be already patched)
                    current_patches = getattr(target_module, "lora_patches", [])
                    
                    # Cache weights on device to avoid redundant transfers on every forward
                    # We use the model's load_device if available
                    device = getattr(model, "load_device", torch.device("cpu"))
                    
                    if is_int8 or not offload_enabled:
                        # Move to device during loading if it's INT8 or offloading is disabled
                        down = down.to(device=device, non_blocking=True)
                        up = up.to(device=device, non_blocking=True)
                        if isinstance(down_scale, torch.Tensor):
                            down_scale = down_scale.to(device=device, non_blocking=True)
                        if isinstance(up_scale, torch.Tensor):
                            up_scale = up_scale.to(device=device, non_blocking=True)
                    # If offload_enabled is True and it's a float LoRA, we keep it on CPU to save VRAM
                    # and only move it to device during the forward pass.

                    # Create a new list with the additional patch
                    # Store as (down, up, alpha, down_scale, up_scale) for INT8 or (down, up, alpha, None, None) for float
                    new_patches = current_patches + [(down, up, alpha, down_scale, up_scale)]
                    
                    # DIRECTLY set the attribute on the module.
                    # ComfyUI's set_model_patch_replace is for attention processors,
                    # not for arbitrary attributes on INT8 modules.
                    target_module.lora_patches = new_patches
                    
                    # Also register it in the patcher so it's tracked (optional but good for compatibility)
                    try:
                        new_model.set_model_patch_replace(new_patches, target_key, "lora_patches")
                    except Exception:
                        pass
                        
                    patched_count += 1
                else:
                    # Module found but doesn't support lora_patches (e.g. not an INT8 Linear)
                    pass
            else:
                failed_count += 1
                failed_keys.append(key)
                if debug:
                    print(f"[DEBUG] ✗ Failed to match key - tried {len(seen)} candidates")

        print(f"LoRA Application Summary:")
        print(f"  - Successfully patched: {patched_count} layers")
        if dim_mismatch_count > 0:
            print(f"  - Dimension mismatches: {dim_mismatch_count} (skipped)")
        if failed_count > 0:
            print(f"  - Keys not found in model: {failed_count}")
            if debug:
                print(f"\n[DEBUG] Failed keys:")
                for fk in failed_keys:
                    print(f"  - {fk}")
        
        # Clear modules dict and collect garbage
        del modules
        gc.collect()
                
        return (new_model,)
