import torch
import time
from llm_scratch.model.base import GPTModern
from llm_scratch.data.tokenizers import ByteTokenizer

def benchmark_inference():
    """
    Measures generation speed (tokens/sec) with and without KV Cache.
    """
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Benchmarking on {device}...")
    
    # Tiny model for benchmark
    model = GPTModern(
        vocab_size=256, 
        block_size=512, 
        n_layer=4, 
        n_head=4, 
        n_embd=256
    ).to(device)
    model.eval()
    
    tokenizer = ByteTokenizer()
    prompt = "Hello world, this is a benchmark to see how fast we can go."
    input_ids = tokenizer.encode(prompt).to(device).unsqueeze(0)
    
    # 1. Benchmark WITH KV Cache (Standard)
    # -------------------------------------
    start = time.time()
    # Generate 200 tokens to let O(N^2) really hurt
    _ = model.generate(input_ids, max_new_tokens=200, temperature=1.0, top_k=5)
    end = time.time()
    dur_kv = end - start
    tps_kv = 200 / dur_kv
    print(f"With KV Cache: {dur_kv:.4f}s ({tps_kv:.2f} tok/s)")
    
    # 2. Benchmark WITHOUT KV Cache (Naive)
    # -------------------------------------
    # We need to forcefully disable caching in the model or simulating it by re-forwarding everything.
    # Since our generate implementations uses KV by default, let's simulate the "naive" loop manually.
    
    start = time.time()
    curr_ids = input_ids.clone()
    for _ in range(200):
        # Naive: forward full sequence every time, don't pass past_kv
        # This is O(N^2) complexity
        logits, _, _ = model(curr_ids)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        curr_ids = torch.cat([curr_ids, next_token], dim=1)
        
    end = time.time()
    dur_no_kv = end - start
    tps_no_kv = 200 / dur_no_kv
    print(f"No KV Cache:   {dur_no_kv:.4f}s ({tps_no_kv:.2f} tok/s)")
    
    speedup = tps_kv / tps_no_kv
    print(f"\nSpeedup Factor: {speedup:.2f}x")
    
    return speedup

if __name__ == "__main__":
    benchmark_inference()
