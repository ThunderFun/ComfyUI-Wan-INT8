# FORK EDITS

### W8A8 Triton Kernel:
Added a custom W8A8 (8-bit weights, 8-bit activations) Triton kernel for optimized INT8 inference. The kernel performs fused quantization, matrix multiplication, and dequantization for improved performance on supported GPUs.

### INT8 Support:
- The **WAN 2.2** INT8 files are functional.

- [wan2.2_i2v_A14b_high_noise_int8_lightx2v_4step_1030.safetensors](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)

- [wan2.2_i2v_A14b_low_noise_int8_lightx2v_4step.safetensors](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)

QuIP quantization is recommended because it typically offers high precision and tends to work well with LoRA adapters, primarily when applied to transformer‑based models: https://github.com/ThunderFun/convert_to_quant_QuIP_INT8

### CPU Offloading (Optional, Buggy):
Added optional CPU offloading that can be enabled/disabled via node settings. When enabled, float LoRA weights are kept on CPU and moved to GPU during inference to save VRAM. May exhibit bugs when enabled.

### LoRA Support:
- Introduced a custom LoRA node.  
- Not thoroughly tested; it may work well in some cases and not in others, depending on how the LoRA was trained or on the precision of the INT8 model.
- Supports LoRAs quantized to INT8, though it incurs higher memory usage.

### Compatibility Note:
The following models have been tested with LoRA support (all quantized using the [convert_to_quant_QuIP_INT8](https://github.com/ThunderFun/convert_to_quant_QuIP_INT8) project): **Wan 2.2, Flux Klein & Z-Image Turbo**. However, both LoRA functionality and model functionality may not be stable and could exhibit issues depending on the specific model and LoRA combination.

---

# Wan 2.2 INT8 Acceleration

This node speeds up Wan 2.2 in ComfyUI by using INT8 quantization. It's unlikely to be faster than proper FP8 on 40-Series and above.
Works with LoRA and torch compile (needed to get full speedup).

## Requirements

Working ComfyKitchen (needs latest comfy and possibly pytorch with cu130)

Triton

Windows untested, but triton-windows may work.

## Hadamard-QuIP Kernel

The Hadamard-QuIP kernel exists but is **disabled by default** due to high VRAM requirements. It may also be very buggy and unusable.

## Credits

- **BobJohnson24** for the original code this fork is based on.

### Original Project Credits:
- dxqb for the INT8 code: https://github.com/Nerogar/OneTrainer/pull/1034
- silveroxides for the base conversion code: https://github.com/silveroxides/convert_to_quant
- silveroxides for showing how to register new data types to Comfy: https://github.com/silveroxides/ComfyUI-QuantOps
