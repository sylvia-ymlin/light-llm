import torch
import torch.nn as nn
from llm_scratch.model.lora import LoRALinear, apply_lora, LoraConfig

def test_lora_linear_forward():
    in_dim, out_dim = 16, 32
    base = nn.Linear(in_dim, out_dim)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    
    x = torch.randn(2, in_dim)
    y = lora(x)
    
    assert y.shape == (2, out_dim)
    # Initially LoRA B is zero, so output should match base layer
    y_base = base(x)
    assert torch.allclose(y, y_base)

def test_apply_lora_structure():
    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 5)
    )
    # Target simple linear layers "0" and "2" (in Sequential they are named "0", "1", "2")
    # But apply_lora targets based on string containment.
    # In Sequential, names are "0", "2".
    # Let's use a custom module for clearer naming
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(10, 10)
            self.w2 = nn.Linear(10, 5)
    
    toy = ToyModel()
    config = LoraConfig(target_modules=["w1"], rank=2)
    
    apply_lora(toy, config)
    
    assert isinstance(toy.w1, LoRALinear)
    assert isinstance(toy.w2, nn.Linear)
    assert not isinstance(toy.w2, LoRALinear)

def test_lora_trainable_params():
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(10, 10)
    
    toy = ToyModel()
    config = LoraConfig(target_modules=["w1"], rank=2)
    apply_lora(toy, config)
    
    # Check requires_grad
    for n, p in toy.named_parameters():
        if "lora_" in n:
            assert p.requires_grad
        elif "base_layer" in n:
            assert not p.requires_grad
