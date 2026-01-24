import os
import torch
import folder_paths
import comfy.sd
import comfy.utils

try:
    from .int8_quant import Int8TensorwiseOps
except ImportError:
    Int8TensorwiseOps = None


class WanVideoINT8Loader:
    """
    Load INT8 tensorwise quantized diffusion models.
    
    Uses Int8TensorwiseOps for direct int8 loading.
    Inference uses fast torch._int_mm for blazing speed. (insert rocket emoji, fire emoji to taste)
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "model_type": (["wan2.2", "wan2.1", "flux2"],),
                "offload_to_cpu": (["enable", "disable"],),
                "chunk_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 512}),
                "auto_convert_to_int8": (["enable", "disable"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "WanVideo/INT8"
    DESCRIPTION = "Load INT8 tensorwise quantized models with fast torch._int_mm inference."

    def load_unet(self, unet_name, model_type, offload_to_cpu, chunk_size, auto_convert_to_int8):
        import comfy.model_management
        import gc
        from comfy.sd import load_diffusion_model
        
        # === CRITICAL: Free memory BEFORE loading new model ===
        # This prevents OOM when loading a second model (e.g., Wan 2.2 low after Wan 2.2 high)
        comfy.model_management.unload_all_models()
        gc.collect()
        try:
            comfy.model_management.soft_empty_cache()
        except RuntimeError as e:
            # Ignore cudaMallocAsync checkPoolLiveAllocations errors
            if "checkPoolLiveAllocations" not in str(e):
                raise
        
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        
        # Use Int8TensorwiseOps for proper direct int8 loading
        model_options = {"custom_operations": Int8TensorwiseOps}
        
        # Set offloading preference
        # NOTE: These class-level attributes are not thread-safe if multiple models 
        # are loaded concurrently, but ComfyUI typically loads models sequentially.
        offload_enabled = (offload_to_cpu == "enable")
        Int8TensorwiseOps.offload_to_cpu = offload_enabled
        Int8TensorwiseOps.chunk_size = chunk_size
        
        # Set auto-convert preference (for loading non-INT8 models like flux2 klein)
        Int8TensorwiseOps.auto_convert_to_int8 = (auto_convert_to_int8 == "enable")
        
        # Reset exclusions (in case this is the second load)
        Int8TensorwiseOps.excluded_names = []
        
        # Check explicit model_type for exclusions
        if model_type == "flux2":
            Int8TensorwiseOps.excluded_names = [
                'img_in', 'time_in', 'guidance_in', 'txt_in', 'final_layer',
                'double_stream_modulation_img', 'double_stream_modulation_txt',
                'single_stream_modulation'
            ]
            #print(f"Applying Flux2-specific exclusions to Int8TensorwiseOps: {Int8TensorwiseOps.excluded_names}")
        elif model_type in ["wan2.1", "wan2.2"]:
            Int8TensorwiseOps.excluded_names = [
                'patch_embed', 'text_projection', 'time_projection', 'head',
                'modulation', 'guidance', 'img_emb', 'txt_emb', 'time_emb',
                'final_layer', 'output_projection'
            ]
            #print(f"Applying Wan-specific exclusions to Int8TensorwiseOps: {Int8TensorwiseOps.excluded_names}")

        # Load model directly - Int8TensorwiseOps handles int8 weights natively
        model = load_diffusion_model(unet_path, model_options=model_options)
        
        return (model,)
