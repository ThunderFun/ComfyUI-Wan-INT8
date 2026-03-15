# Request: Add support for QuIP INT4 (int4_quip) models

## Description

There is now an official `INT4` branch in the quantization tool:

→ https://github.com/ThunderFun/convert_to_quant_QuIP_INT8/tree/INT4

This branch fully implements **QuIP INT4** quantization using the `--int4` flag and produces models with quantization type **`int4_quip`**.

### Current limitation
The current nodes in this repository (`WanVideoINT8Loader`, etc.) only support INT8 formats (`int8_tensorwise` and `int8_blockwise`). INT4 models cannot be loaded.

### Why it is important
- Models become ~2x smaller compared to INT8 (~25% of FP16)
- Noticeable speed improvement on RTX 30xx/40xx series due to reduced memory bandwidth
- Many users are already quantizing models to INT4 and need a proper loader

### Suggested implementation
- Either extend existing loaders to support `int4_quip`
- Or (preferred) add a new node `WanVideoINT4Loader` / `UNETLoaderINT4`

I (and many others) would really appreciate native INT4 support.

---

**Related links:**
- Converter INT4 branch: https://github.com/ThunderFun/convert_to_quant_QuIP_INT8/tree/INT4
- Original INT8 node: this repository

Thank you for your great work!
