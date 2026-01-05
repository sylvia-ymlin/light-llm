import torch
import torch.nn as nn
from dataclasses import dataclass, field
import math

@dataclass
class LoraConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["wq", "wk", "wv", "output", "w1", "w2", "w3"])

class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear layer and adds Low-Rank Adaptation (LoRA).
    y = Wx + (B @ A)x * (alpha / r)
    """
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freezing the base layer is usually done by the caller (apply_lora), 
        # but we ensure it's recorded here.
        # self.base_layer.requires_grad_(False) # Can be done here if desired strictly.

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # LoRA weights: A (in -> r), B (r -> out)
        # Initialize A with Kaiming uniform (like linear) and B with zeros 
        # so initial contribution is zero.
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialization as recommended in the paper
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward
        # We rely on base_layer being valid.
        base_out = self.base_layer(x)
        
        # LoRA forward
        # x: (..., in_features)
        # lora_A: (r, in) -> x @ A.T -> (..., r)
        # lora_B: (out, r) -> (..., r) @ B.T -> (..., out)
        
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        
        return base_out + lora_out * self.scaling

def apply_lora(model: nn.Module, config: LoraConfig):
    """
    Replaces targeted nn.Linear layers with LoRALinear layers and freezes base params.
    """
    # 1. Freeze entire model first
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Iterate and replace
    # We need to iterate carefully to replace modules in-place.
    # Recursion is easiest.
    
    def replace_module(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if this child should be targeted
            # Simple check: is it Linear and does its name match target_modules?
            is_target = any(t in name for t in config.target_modules)
            
            if isinstance(child, nn.Linear) and is_target:
                print(f"Applying LoRA to {full_name}")
                # Create LoRA wrapper
                # Note: We must share the SAME weight tensor, not copy it, 
                # but nn.Linear(..., bias=...) creates new parameters.
                # So we pass the child instance itself to LoRALinear.
                new_layer = LoRALinear(child, config.rank, config.alpha, config.dropout)
                setattr(module, name, new_layer)
            else:
                # Recurse
                replace_module(child, full_name)

    replace_module(model)
    
    # 3. Ensure LoRA params are trainable
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
            
    return model
