import torch
from torch import Tensor
import torch.nn.functional as F
import re

# Standard LoRA key patterns for Wan models
WAN_LORA_KEY_MAP = {
    # Transformer blocks (Kohya style)
    r"lora_transformer_blocks_(\d+)_attn_to_q": "diffusion_model.blocks.{i}.self_attn.to_q",
    r"lora_transformer_blocks_(\d+)_attn_to_k": "diffusion_model.blocks.{i}.self_attn.to_k",
    r"lora_transformer_blocks_(\d+)_attn_to_v": "diffusion_model.blocks.{i}.self_attn.to_v",
    r"lora_transformer_blocks_(\d+)_attn_to_out": "diffusion_model.blocks.{i}.self_attn.to_out.0",
    
    # Cross attention
    r"lora_transformer_blocks_(\d+)_cross_attn_to_q": "diffusion_model.blocks.{i}.cross_attn.to_q",
    r"lora_transformer_blocks_(\d+)_cross_attn_to_k": "diffusion_model.blocks.{i}.cross_attn.to_k",
    r"lora_transformer_blocks_(\d+)_cross_attn_to_v": "diffusion_model.blocks.{i}.cross_attn.to_v",
    r"lora_transformer_blocks_(\d+)_cross_attn_to_out": "diffusion_model.blocks.{i}.cross_attn.to_out.0",
    
    # FFN layers
    r"lora_transformer_blocks_(\d+)_ffn_fc1": "diffusion_model.blocks.{i}.ffn.fc1",
    r"lora_transformer_blocks_(\d+)_ffn_fc2": "diffusion_model.blocks.{i}.ffn.fc2",

    # XLabs style mappings
    r"transformer\.blocks\.(\d+)\.attn\.to_q": "diffusion_model.blocks.{i}.self_attn.to_q",
    r"transformer\.blocks\.(\d+)\.attn\.to_k": "diffusion_model.blocks.{i}.self_attn.to_k",
    r"transformer\.blocks\.(\d+)\.attn\.to_v": "diffusion_model.blocks.{i}.self_attn.to_v",
    r"transformer\.blocks\.(\d+)\.attn\.to_out\.0": "diffusion_model.blocks.{i}.self_attn.to_out.0",
    r"transformer\.blocks\.(\d+)\.ffn\.fc1": "diffusion_model.blocks.{i}.ffn.fc1",
    r"transformer\.blocks\.(\d+)\.ffn\.fc2": "diffusion_model.blocks.{i}.ffn.fc2",

    # Simple blocks style (some trainers)
    r"blocks\.(\d+)\.attn\.q_proj": "diffusion_model.blocks.{i}.self_attn.to_q",
    r"blocks\.(\d+)\.attn\.k_proj": "diffusion_model.blocks.{i}.self_attn.to_k",
    r"blocks\.(\d+)\.attn\.v_proj": "diffusion_model.blocks.{i}.self_attn.to_v",
    r"blocks\.(\d+)\.attn\.out_proj": "diffusion_model.blocks.{i}.self_attn.to_out.0",

    # New patterns from debug output (lora_unet_blocks style)
    r"lora_unet_blocks_(\d+)_self_attn_q": "diffusion_model.blocks.{i}.self_attn.q",
    r"lora_unet_blocks_(\d+)_self_attn_k": "diffusion_model.blocks.{i}.self_attn.k",
    r"lora_unet_blocks_(\d+)_self_attn_v": "diffusion_model.blocks.{i}.self_attn.v",
    r"lora_unet_blocks_(\d+)_self_attn_o": "diffusion_model.blocks.{i}.self_attn.o",
    
    r"lora_unet_blocks_(\d+)_cross_attn_q": "diffusion_model.blocks.{i}.cross_attn.q",
    r"lora_unet_blocks_(\d+)_cross_attn_k": "diffusion_model.blocks.{i}.cross_attn.k",
    r"lora_unet_blocks_(\d+)_cross_attn_v": "diffusion_model.blocks.{i}.cross_attn.v",
    r"lora_unet_blocks_(\d+)_cross_attn_o": "diffusion_model.blocks.{i}.cross_attn.o",
    
    r"lora_unet_blocks_(\d+)_ffn_0": "diffusion_model.blocks.{i}.ffn.0",
    r"lora_unet_blocks_(\d+)_ffn_2": "diffusion_model.blocks.{i}.ffn.2",

    # Flux patterns (Kohya/X-Labs style)
    r"lora_unet_double_blocks_(\d+)_img_attn_qkv": "diffusion_model.double_blocks.{i}.img_attn.qkv",
    r"lora_unet_double_blocks_(\d+)_img_attn_proj": "diffusion_model.double_blocks.{i}.img_attn.proj",
    r"lora_unet_double_blocks_(\d+)_txt_attn_qkv": "diffusion_model.double_blocks.{i}.txt_attn.qkv",
    r"lora_unet_double_blocks_(\d+)_txt_attn_proj": "diffusion_model.double_blocks.{i}.txt_attn.proj",
    r"lora_unet_double_blocks_(\d+)_img_mlp_0": "diffusion_model.double_blocks.{i}.img_mlp.0",
    r"lora_unet_double_blocks_(\d+)_img_mlp_2": "diffusion_model.double_blocks.{i}.img_mlp.2",
    r"lora_unet_double_blocks_(\d+)_txt_mlp_0": "diffusion_model.double_blocks.{i}.txt_mlp.0",
    r"lora_unet_double_blocks_(\d+)_txt_mlp_2": "diffusion_model.double_blocks.{i}.txt_mlp.2",
    
    r"lora_unet_single_blocks_(\d+)_linear1": "diffusion_model.single_blocks.{i}.linear1",
    r"lora_unet_single_blocks_(\d+)_linear2": "diffusion_model.single_blocks.{i}.linear2",
}

class LoRAWeights:
    def __init__(self):
        self.weights = {}  # key -> (lora_down, lora_up, alpha)
        self.device = torch.device("cpu")
    
    def add(self, key: str, down: Tensor, up: Tensor, alpha: float = 1.0):
        # Convert to float16 for consistency and to avoid bfloat16->float16 conversion issues
        # bfloat16->float16 conversion can introduce non-determinism during inference
        target_dtype = torch.float16 if down.dtype in (torch.bfloat16, torch.float32) else down.dtype
        self.weights[key] = (
            down.to(device=self.device, dtype=target_dtype),
            up.to(device=self.device, dtype=target_dtype),
            alpha
        )
    
    def get_for_layer(self, key: str, device: torch.device):
        if key not in self.weights:
            return None, None, None
        down, up, alpha = self.weights[key]
        return down.to(device), up.to(device), alpha

def detect_lora_format(state_dict):
    keys = list(state_dict.keys())
    
    if any("lora_down" in k for k in keys):
        return "kohya"
    elif any("lora_A" in k for k in keys):
        return "peft"
    elif any(".down.weight" in k for k in keys) or any(".down_proj.weight" in k for k in keys):
        return "standard"
    
    return "unknown"

def parse_wan_lora(state_dict, strength=1.0):
    lora_format = detect_lora_format(state_dict)
    print(f"Detected LoRA format: {lora_format}")
    
    parsed_weights = LoRAWeights()
    
    # Group keys by their base name
    groups = {}
    
    if lora_format == "kohya":
        # Kohya format: lora_transformer_blocks_0_attn_to_q.lora_down.weight
        keys = list(state_dict.keys())
        for key in keys:
            if ".lora_down.weight" in key:
                base_key = key.replace(".lora_down.weight", "")
                groups[base_key] = {
                    "down": state_dict.pop(key),
                    "up": state_dict.pop(base_key + ".lora_up.weight"),
                    "alpha": state_dict.pop(base_key + ".alpha", None)
                }
    elif lora_format == "peft":
        # PEFT format: base_model.model.diffusion_model.blocks.0.self_attn.to_q.lora_A.weight
        keys = list(state_dict.keys())
        for key in keys:
            if key not in state_dict: continue
            # Handle both .lora_A.weight and .lora_A.default.weight patterns
            if ".lora_A.weight" in key or ".lora_A.default.weight" in key:
                if ".lora_A.default.weight" in key:
                    base_key = key.replace(".lora_A.default.weight", "")
                    lora_b_key = base_key + ".lora_B.default.weight"
                else:
                    base_key = key.replace(".lora_A.weight", "")
                    lora_b_key = base_key + ".lora_B.weight"
                
                if lora_b_key not in state_dict: continue
                
                # Map PEFT keys back to model keys
                model_key = base_key
                prefixes_to_strip = ["base_model.model.model.", "base_model.model.", "model."]
                for prefix in prefixes_to_strip:
                    if model_key.startswith(prefix):
                        model_key = model_key[len(prefix):]
                        break
                
                down_weight = state_dict.pop(key)
                up_weight = state_dict.pop(lora_b_key)
                
                alpha_value = None
                for alpha_key in [base_key + ".alpha", base_key + ".lora_alpha", base_key + ".scaling"]:
                    if alpha_key in state_dict:
                        alpha_value = state_dict.pop(alpha_key)
                        break
                
                groups[model_key] = {"down": down_weight, "up": up_weight, "alpha": alpha_value}
    elif lora_format == "standard":
        # Standard format: diffusion_model.blocks.0.self_attn.to_q.down.weight
        for key in state_dict:
            if ".down.weight" in key:
                base_key = key.replace(".down.weight", "")
                groups[base_key] = {
                    "down": state_dict[key],
                    "up": state_dict[base_key + ".up.weight"],
                    "alpha": state_dict.get(base_key + ".alpha", None)
                }
            elif ".down_proj.weight" in key:
                base_key = key.replace(".down_proj.weight", "")
                groups[base_key] = {
                    "down": state_dict[key],
                    "up": state_dict[base_key + ".up_proj.weight"],
                    "alpha": state_dict.get(base_key + ".alpha", None)
                }

    # Map groups to model keys
    for base_key, weights in groups.items():
        model_key = None
        
        # Try regex mapping for Kohya-style keys
        for pattern, replacement in WAN_LORA_KEY_MAP.items():
            match = re.match(pattern, base_key)
            if match:
                i = match.group(1)
                model_key = replacement.format(i=i)
                break
        
        # If no regex match, assume the base_key is already the model key (PEFT/Standard)
        if not model_key:
            model_key = base_key
            
        alpha = weights["alpha"]
        if alpha is not None:
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.item()
            # LoRA scaling is alpha / rank
            rank = weights["down"].shape[0]
            scale = alpha / rank
        else:
            scale = 1.0
            
        parsed_weights.add(model_key, weights["down"], weights["up"], scale * strength)
        
    return parsed_weights
