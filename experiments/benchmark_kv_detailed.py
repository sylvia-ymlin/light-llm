#!/usr/bin/env python3
"""
Detailed KV Cache benchmark across different devices and configurations.
"""

import torch
import time
import platform
from llm_scratch.model.base import GPTModern
from llm_scratch.data.tokenizers import ByteTokenizer

def get_device_info():
    """Get detailed device information."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_type = "CUDA"
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    elif torch.backends.mps.is_available():
        device_name = "Apple Silicon (MPS)"
        device_type = "MPS"
        memory_gb = "Unknown"
    else:
        device_name = platform.processor()
        device_type = "CPU"
        memory_gb = "Unknown"
    
    return device_type, device_name, memory_gb

def benchmark_kv_cache(model_config, sequence_lengths=[50, 100, 200]):
    """Benchmark KV cache across different sequence lengths."""
    device_type, device_name, memory_gb = get_device_info()
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    print(f"🔧 Device: {device_type} - {device_name}")
    if memory_gb != "Unknown":
        print(f"💾 Memory: {memory_gb:.1f} GB")
    print(f"🏗️ Model: {model_config['n_layer']}L-{model_config['n_embd']}D")
    print("=" * 60)
    
    # Create model
    model = GPTModern(**model_config).to(device)
    model.eval()
    
    tokenizer = ByteTokenizer()
    prompt = "Hello world, this is a benchmark to see how fast we can go with different sequence lengths."
    input_ids = tokenizer.encode(prompt).to(device).unsqueeze(0)
    
    results = []
    
    for seq_len in sequence_lengths:
        print(f"\n📏 Testing sequence length: {seq_len} tokens")
        print("-" * 40)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model.generate(input_ids, max_new_tokens=10, temperature=1.0)
        
        # Benchmark WITH KV Cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=seq_len, temperature=1.0, top_k=5)
        
        end_time = time.time()
        dur_kv = end_time - start_time
        tps_kv = seq_len / dur_kv
        
        # Benchmark WITHOUT KV Cache (naive O(N^2) approach)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        start_time = time.time()
        
        curr_ids = input_ids.clone()
        with torch.no_grad():
            for _ in range(seq_len):
                logits, _, _ = model(curr_ids)
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                curr_ids = torch.cat([curr_ids, next_token], dim=1)
        
        end_time = time.time()
        dur_no_kv = end_time - start_time
        tps_no_kv = seq_len / dur_no_kv
        
        speedup = tps_kv / tps_no_kv
        
        print(f"✅ With KV Cache:    {dur_kv:.4f}s ({tps_kv:.2f} tok/s)")
        print(f"❌ Without KV Cache: {dur_no_kv:.4f}s ({tps_no_kv:.2f} tok/s)")
        print(f"🚀 Speedup Factor:   {speedup:.2f}x")
        
        results.append({
            'seq_len': seq_len,
            'with_kv': tps_kv,
            'without_kv': tps_no_kv,
            'speedup': speedup
        })
    
    return results, device_type, device_name

def benchmark_different_models():
    """Benchmark different model sizes."""
    configs = [
        {"name": "Tiny", "vocab_size": 256, "block_size": 512, "n_layer": 2, "n_head": 2, "n_embd": 64},
        {"name": "Small", "vocab_size": 256, "block_size": 512, "n_layer": 4, "n_head": 4, "n_embd": 128},
        {"name": "Medium", "vocab_size": 256, "block_size": 512, "n_layer": 6, "n_head": 8, "n_embd": 256},  # 修复：8头可以被256整除
    ]
    
    all_results = {}
    
    for config in configs:
        print(f"\n🧪 Testing {config['name']} Model")
        print("=" * 50)
        
        model_config = {k: v for k, v in config.items() if k != 'name'}
        results, device_type, device_name = benchmark_kv_cache(model_config, [100, 200])
        all_results[config['name']] = results
    
    return all_results, device_type, device_name

def create_benchmark_report():
    """Create a comprehensive benchmark report."""
    print("🚀 KV Cache Comprehensive Benchmark")
    print("=" * 60)
    
    results, device_type, device_name = benchmark_different_models()
    
    print(f"\n📊 Summary Report")
    print("=" * 60)
    print(f"Device: {device_type} - {device_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Platform: {platform.system()} {platform.release()}")
    
    print(f"\n{'Model':<8} {'Seq Len':<8} {'With KV':<12} {'Without KV':<12} {'Speedup':<10}")
    print("-" * 60)
    
    max_speedup = 0
    best_config = None
    
    for model_name, model_results in results.items():
        for result in model_results:
            print(f"{model_name:<8} {result['seq_len']:<8} "
                  f"{result['with_kv']:<12.2f} {result['without_kv']:<12.2f} "
                  f"{result['speedup']:<10.2f}x")
            
            if result['speedup'] > max_speedup:
                max_speedup = result['speedup']
                best_config = (model_name, result['seq_len'])
    
    print(f"\n🏆 Best Performance:")
    print(f"   {max_speedup:.2f}x speedup with {best_config[0]} model at {best_config[1]} tokens")
    
    # Device-specific notes
    if device_type == "MPS":
        print(f"\n📝 Note: MPS (Apple Silicon) results may be lower than NVIDIA RTX 3090")
        print(f"   Expected RTX 3090 speedup: ~7-8x for longer sequences")
    elif device_type == "CUDA":
        print(f"\n📝 Note: CUDA results on high-end GPUs (RTX 3090/4090) typically show")
        print(f"   higher speedups (7-10x) due to superior memory bandwidth")
    
    return max_speedup, device_type

if __name__ == "__main__":
    max_speedup, device_type = create_benchmark_report()
    
    # Save results to file
    with open("kv_cache_benchmark_results.txt", "w") as f:
        f.write(f"KV Cache Benchmark Results\n")
        f.write(f"Device: {device_type}\n")
        f.write(f"Max Speedup: {max_speedup:.2f}x\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"Platform: {platform.system()} {platform.release()}\n")
    
    print(f"\n💾 Results saved to: kv_cache_benchmark_results.txt")