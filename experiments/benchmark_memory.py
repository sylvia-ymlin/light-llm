import torch
import sys
from llm_scratch.model.base import GPTModern
from llm_scratch.model.lora import apply_lora, LoraConfig

def get_model_size_mb(model):
    """Calculates model parameter size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024**2

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_memory():
    print("Benchmarking Model Memory Footprint...")
    
    # Configuration usually used for 124M param model (GPT-2 Small equivalent)
    # Reducing slightly for faster local test, or keep standard to be impressive?
    # Let's use a "Medium" conf to show significant savings.
    config = dict(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768)
    
    # 1. Full Model
    print("\n--- Full Model (Standard SFT) ---")
    model_full = GPTModern(**config)
    mem_full = get_model_size_mb(model_full)
    params_full = count_trainable_params(model_full)
    print(f"Total Params: {params_full:,}")
    print(f"Model Size (FP32): {mem_full:.2f} MB")
    
    # 2. LoRA Model
    print("\n--- LoRA Model (Rank=8) ---")
    model_lora = GPTModern(**config)
    apply_lora(model_lora, LoraConfig(rank=8))
    
    params_lora_trainable = count_trainable_params(model_lora)
    params_lora_total = sum(p.numel() for p in model_lora.parameters())
    
    # For LoRA, the "gradients" only exist for trainable params. 
    # Optimizer state is also proportional to trainable params.
    # roughly: Memory = Model Weights + Gradients + Optimizer States + Activations
    # LoRA reduces Gradient + Optimizer state memory massively.
    
    print(f"Total Params: {params_lora_total:,}")
    print(f"Trainable Params: {params_lora_trainable:,}")
    reduction = (1 - params_lora_trainable / params_full) * 100
    print(f"Trainable Parameter Reduction: {reduction:.2f}%")
    
    # 3. Simulate Training Memory (Theoretical Estimate for Resume)
    # AdamW stores 2 states per param (momentum, variance).
    # Full: 2 * 4 bytes * N_params
    # LoRA: 2 * 4 bytes * N_lora_params
    
    optim_mem_full = (params_full * 8) / 1024**2
    optim_mem_lora = (params_lora_trainable * 8) / 1024**2
    
    print(f"\n--- Theoretical Optimizer Memory (AdamW) ---")
    print(f"Full Fine-Tuning: {optim_mem_full:.2f} MB")
    print(f"LoRA Fine-Tuning: {optim_mem_lora:.2f} MB")
    print(f"Memory Savings (Optimizer): {(1 - optim_mem_lora/optim_mem_full)*100:.2f}%")

if __name__ == "__main__":
    benchmark_memory()
