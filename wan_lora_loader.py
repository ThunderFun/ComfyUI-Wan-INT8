import torch
import torch.nn.functional as F
import folder_paths
import comfy.utils
import gc
from .lora_utils import parse_wan_lora

class WanLoRALoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "WanVideo/INT8"
    
    def load_lora(self, model, lora_name, strength):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if strength == 0:
            return (model,)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            print(f"LoRA not found: {lora_name}")
            return (model,)

        print(f"Loading LoRA: {lora_path} with strength {strength}")
        lora_state_dict = comfy.utils.load_torch_file(lora_path)

        # Parse LoRA weights and map to model keys
        lora_weights = parse_wan_lora(lora_state_dict, strength)
        
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
        
        patched_count = 0
        failed_count = 0
        
        failed_keys = []  # Track which keys failed
        dim_mismatch_count = 0
        
        for key in lora_weights.weights:
            target_module = None
            target_key = None
            
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
            
            # 5. Remove duplicates and try matching
            seen = set()
            for cand in candidates:
                if cand in seen: continue
                seen.add(cand)
                if cand in modules:
                    target_module = modules[cand]
                    target_key = cand
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
                                    for d, u, a in patches:
                                        # Move to device/dtype of output
                                        d_t = d.to(device=out.device, dtype=out.dtype)
                                        u_t = u.to(device=out.device, dtype=out.dtype)
                                        
                                        # Compute LoRA contribution: (x @ d.T) @ u.T
                                        # We use F.linear which handles multidimensional x correctly
                                        lora_out = F.linear(F.linear(x, d_t), u_t)
                                            
                                        out.add_(lora_out, alpha=a)
                                        del d_t, u_t, lora_out
                                return out
                            return patched_forward
                        
                        target_module.forward = make_patched_forward(target_module, original_forward)

                if hasattr(target_module, "lora_patches"):
                    down, up, alpha = lora_weights.weights[key]
                    
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
                    
                    # Create a new list with the additional patch
                    new_patches = current_patches + [(down, up, alpha)]
                    
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

        print(f"LoRA Application Summary:")
        print(f"  - Successfully patched: {patched_count} layers")
        if dim_mismatch_count > 0:
            print(f"  - Dimension mismatches: {dim_mismatch_count} (skipped)")
        if failed_count > 0:
            print(f"  - Keys not found in model: {failed_count}")
        
        # Clear modules dict and collect garbage
        del modules
        gc.collect()
                
        return (new_model,)
