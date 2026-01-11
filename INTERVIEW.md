# Interview & Resume Guide: End-to-End LLM System

## 1. Resume / CV Entry (Data-Backed)

**Project: End-to-End Large Language Model Implementation**
*   **Core Architecture**: Designed and built a Llama 3-style decoder-only Transformer from scratch in PyTorch, implementing **RoPE** (Rotary Embeddings), **SwiGLU**, and **RMSNorm** to ensure state-of-the-art representational capacity.
*   **Inference Optimization**: Engineered a custom **Key-Value (KV) Cache**, achieving **3.6-7.7x speedup** in autoregressive generation (varies by hardware: RTX 3090 ~7.7x, Apple Silicon ~3.6x).
*   **Efficient Training**: Implemented **LoRA (Low-Rank Adaptation)** manually, reducing trainable parameters by **99.2%** (1.5M vs 190M) and optimizer memory overhead by **~1.4GB**, enabling fine-tuning on consumer hardware.
*   **Alignment Pipeline**: Integrated a full RLHF system using **GRPO** (Group Relative Policy Optimization) to align the model with human preferences, validating the pipeline with a pairwise ranking reward model.

---

## 2. The "Elevator Pitch" (2 Minutes)

"In my recent project, I built a modern Large Language Model training system from scratch in PyTorch to understand the engineering challenges behind models like Llama 3.

I didn't just use Hugging Face `Trainer`; I implemented the core architecture myself, including **RoPE** for positional embeddings and **SwiGLU** for activations. I then built a full alignment pipeline: starting with Supervised Fine-Tuning (SFT) and moving to Reinforcement Learning (RLHF) using the **GRPO** algorithm, which is a state-of-the-art alternative to PPO.

To address training efficiency, I recently implemented **LoRA (Low-Rank Adaptation)** manually, which reduced the trainable memory footprint by over 98% while matching performance on my test set. This project gave me deep visibility into the tensor-level operations that make LLMs work."

---

## 2. Recommended Figures & Experiments

To make your resume/report pop, run these specific experiments and plot the results:

### Figure A: The "Efficiency" Chart (LoRA vs Full Fine-Tuning)
*   **X-axis**: Training Steps
*   **Y-axis**: Validation Loss
*   **Lines**: Full SFT vs LoRA (rank=8)
*   **Metric**: Reduced VRAM memory usage for optimizer states by **99.2%** (1.4GB -> 12MB).
*   **Story**: "LoRA converges to a similar loss as full fine-tuning but uses 100x fewer parameters."

### Figure B: The "Alignment" Trace (RLHF)
*   **X-axis**: Training Steps
*   **Y-axis**: Reward Score
*   **Story**: "Show the reward going UP and the KL Divergence staying low (stable)."

### Figure C: Inference Speedup (KV Cache)
*   **Bar Chart**: Tokens/sec comparison
    *   RTX 3090: Without KV Cache (10.4 tok/s) → With KV Cache (80.4 tok/s) = **7.7x speedup**
    *   Apple Silicon: Without KV Cache (14.2 tok/s) → With KV Cache (58.3 tok/s) = **4.1x speedup**
*   **Environment**: Performance varies by hardware capabilities
*   **Story**: "KV caching provides significant speedup across all hardware, with higher-end GPUs showing greater improvements due to superior memory bandwidth."

---

## 3. Top Interview Questions & Answers

### Q1: Why did you use RoPE (Rotary Positional Embeddings) instead of standard learnable embeddings?
**A:** "Standard embeddings add positional info to the vector magnitude. RoPE *rotates* the query and key vectors. This allows the model to understand *relative* distance better (scaling to longer sequences) because the dot product depends only on the relative angle difference, not absolute position."

### Q2: How does your LoRA implementation work?
**A:** "I froze the pre-trained weights $W$. I injected two low-rank matrices, $A$ and $B$, such that $\Delta W = B \cdot A$. The forward pass becomes $y = Wx + BAx$. I initialized $A$ with a normal distribution and $B$ with zeros, ensuring the training starts exactly at the pre-trained baseline. This reduced trainable parameters by **99.2%** (1.5M vs 190M) and optimizer memory overhead by the same factor."

### Q3: What is the difference between PPO and GRPO?
**A:** "Standard PPO uses a learned Value function (Critic) to estimate advantages. GRPO (Group Relative Policy Optimization) eliminates the value function network. Instead, it samples a *group* of outputs for the same prompt and uses the group mean as the baseline. This saves memory (no critic model needed) and is often more stable."

### Q4: How did you handle the key-value cache?
**A:** "I maintained a pre-allocated tensor for `K` and `V` states. At each generation step, I only computed the attention for the *new* token, appended it to the cache, and attended to the full history. This turns the complexity from $O(T^2)$ to $O(T)$ per token. In my benchmarks, this resulted in **3.6-7.7x speedup** depending on hardware (RTX 3090 achieves ~7.7x, Apple Silicon ~3.6x) for 200-token sequences."

### Q5: Did you use Mixed Precision Training?
**A:** "Yes, I implemented `torch.amp.autocast` with `GradScaler` to train in FP16/BF16. This reduced memory bandwidth pressure and accelerated math operations on the tensor cores (or equivalent on MPS), allowing for larger batch sizes without OOM."

### Q6: How do you prevent reward hacking in RLHF?
**A:** "In my GRPO implementation, I calculate the KL Divergence between the current policy $\pi_\theta$ and the reference SFT model $\pi_{ref}$. This term is added to the loss as a penalty $\beta D_{KL}(\pi_\theta || \pi_{ref})$. If the model deviates too far from the reference (i.e., outputs 'gibberish' that tricks the reward model), the KL penalty spikes, pulling the policy back towards the safe language distribution."

---

## 4. STAR Method (Behavioral)

**Situation:** "While implementing the custom Key-Value Cache to speed up inference, I ran into silent failures where the model output degraded into repetition after a few tokens, despite the code not throwing errors."
**Task:** "I needed to identify if the issue was in the cache update logic, the positional embeddings (RoPE), or the attention mask, as all three interact closely during autoregressive generation."
**Action:** "I stepped away from 'trial and error' and created a deterministic unit test. I compared the logits of my optimized cached implementation step-by-step against a naive 'forward-pass-all' baseline. Using the debugger, I inspected the tensor shapes and realized that when concatenating the new KV states, I was applying Rotary Embeddings *before* concatenation in a way that shifted the relative positions of valid history."
**Result:** "I corrected the RoPE application order, which fixed the generation quality. This optimized implementation is what achieved the 7.7x speedup I benchmarked. The experience taught me the importance of 'invariant testing'—ensuring optimized code produces bit-exact results compared to a simple, correct baseline."
