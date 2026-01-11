from __future__ import annotations
import argparse
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List

from ..model.reward import RewardModel
from ..data.formatting import format_example, format_prompt_only, Example
from ..data.tokenizers import BPETokenizer, ByteTokenizer
from .policy import PolicyWithValue

# -----------------------------------------------------------------------------
# GRPO/PPO Loss Logic
# -----------------------------------------------------------------------------

@dataclass
class PolicyOnlyLossOut:
    policy_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    kl_ref: torch.Tensor
    total_loss: torch.Tensor

def ppo_policy_only_losses(new_logp, old_logp, adv, clip_ratio=0.2, ent_coef=0.0,
                           kl_coef: float = 0.0, kl_mean: torch.Tensor | None = None):
    """
    PPO-style clipped policy loss, *policy only* (no value head),
    plus a separate KL(π||π_ref) penalty term:  total = L_PPO + kl_coef * KL.
    Inputs are flat over action tokens: new_logp, old_logp, adv: (N_act,).
    kl_mean is a scalar tensor (mean over action tokens).
    """
    device = new_logp.device if new_logp.is_cuda else None
    if new_logp.numel() == 0:
        zero = torch.tensor(0.0, device=device)
        return PolicyOnlyLossOut(zero, zero, zero, zero, zero)

    ratio = torch.exp(new_logp - old_logp)  # (N,)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    policy_loss = -torch.mean(torch.min(unclipped, clipped))

    entropy = -new_logp.mean() if ent_coef != 0.0 else new_logp.new_tensor(0.0)
    approx_kl = torch.mean(old_logp - new_logp)

    kl_ref = kl_mean if kl_mean is not None else new_logp.new_tensor(0.0)

    total = policy_loss - ent_coef * entropy + kl_coef * kl_ref
    return PolicyOnlyLossOut(policy_loss, entropy, approx_kl, kl_ref, total)

# -----------------------------------------------------------------------------
# Rollout / Logprob Utilities
# -----------------------------------------------------------------------------

def shift_labels(x: torch.Tensor) -> torch.Tensor:
    # For causal LM: predict x[t+1] from x[:t]
    return x[:, 1:].contiguous()

def gather_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

@torch.no_grad()
def model_logprobs(model, x: torch.Tensor) -> torch.Tensor:
    # compute log p(x[t+1] | x[:t]) for t
    logits, _, _ = model.lm(x, None) if hasattr(model, 'lm') else model(x, None)
    labels = shift_labels(x)
    lp = gather_logprobs(logits[:, :-1, :], labels)
    return lp  # (B, T-1)

def sample_prompts(n: int) -> List[str]:
    # Try using datasets if available, else fallback
    try:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train[:24]")
        arr = []
        for r in ds:
            inst = (r.get('instruction') or '').strip()
            inp = (r.get('input') or '').strip()
            if inp:
                inst = inst + "\n" + inp
            if inst:
                arr.append(inst)
            if len(arr) >= n:
                break
        if arr:
            return arr
    except Exception:
        pass
    
    base = [
        "Explain the purpose of attention in transformers.",
        "Give two pros and cons of BPE tokenization.",
        "Summarize why PPO is used in RLHF.",
        "Write a tiny Python function that reverses a list.",
    ]
    return (base * ((n+len(base)-1)//len(base)))[:n]

class RLHFTokenizer:
    def __init__(self, block_size: int, bpe_dir: str | None = None, vocab_size: int = 8000):
        self.block_size = block_size
        self.tok = None
        
        # Try BPE first, but handle failures gracefully
        if bpe_dir:
            try:
                self.tok = BPETokenizer(vocab_size=vocab_size)
                self.tok.load(bpe_dir)
            except Exception:
                print(f"Warning: Failed to load BPE tokenizer from {bpe_dir}, falling back to ByteTokenizer")
                self.tok = None
        
        # Fallback to ByteTokenizer if BPE failed or not requested
        if self.tok is None:
            self.tok = ByteTokenizer()

    @property
    def vocab_size(self) -> int:
        return getattr(self.tok, 'vocab_size', 256)

    def encode(self, text: str) -> List[int]:
        if self.tok is None:
            return list(text.encode('utf-8'))
        ids = self.tok.encode(text)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        elif hasattr(ids, 'ids'):  # BPE tokenizer result
            ids = ids.ids
        return ids

    def decode(self, ids: List[int]) -> str:
        if hasattr(self.tok, 'decode'):
            return self.tok.decode(ids)
        return bytes(ids).decode('utf-8', errors='ignore')

@torch.no_grad()
def compute_reward(reward_model: RewardModel, tok: RLHFTokenizer, prompt_text: str, response_ids: list[int], device) -> float:
    """Compute reward for a prompt-response pair using the trained reward model."""
    resp_text = tok.decode(response_ids)
    text = format_example(Example(prompt_text, resp_text))
    ids = tok.encode(text)
    
    # Truncate to model's block size
    max_len = getattr(reward_model, 'block_size', tok.block_size)
    ids = ids[:max_len]
    
    # Pad if necessary (pad token = 2)
    if len(ids) < max_len:
        ids = ids + [2] * (max_len - len(ids))
    
    x = torch.tensor([ids], dtype=torch.long, device=device)
    r = reward_model(x)
    return float(r[0].item())

# -----------------------------------------------------------------------------
# Main Training Loop (GRPO)
# -----------------------------------------------------------------------------

def train_grpo(
    policy_ckpt: str,
    reward_ckpt: str,
    out_dir: str = 'runs/grpo-demo',
    steps: int = 100,
    batch_prompts: int = 32,
    group_size: int = 4,
    block_size: int = 256,
    resp_len: int = 64,
    kl_coef: float = 0.01,
    lr: float = 1e-5,
    bpe_dir: str | None = None,
    device: str = 'cpu'
):
    device = torch.device(device)
    print(f"Starting GRPO on {device}...")

    # tokenizer
    tok = RLHFTokenizer(block_size=block_size, bpe_dir=bpe_dir)

    # Load SFT policy (and a frozen reference)
    ckpt = torch.load(policy_ckpt, map_location=device)
    cfg = ckpt.get('config', {})
    vocab_size = cfg.get('vocab_size', tok.vocab_size)
    # Override config with args if needed, but best to respect ckpt
    model_block_size = cfg.get('block_size', block_size)
    n_layer = cfg.get('n_layer', 2)
    n_head  = cfg.get('n_head', 2)
    n_embd  = cfg.get('n_embd', 128)

    policy = PolicyWithValue(vocab_size, model_block_size, n_layer, n_head, n_embd).to(device)
    policy.lm.load_state_dict(ckpt['model'])
    policy.eval()

    ref = PolicyWithValue(vocab_size, model_block_size, n_layer, n_head, n_embd).to(device)
    ref.lm.load_state_dict(ckpt['model'])
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref.eval()

    # Reward model
    rckpt = torch.load(reward_ckpt, map_location=device)
    rm = RewardModel(
        vocab_size=rckpt['config'].get('vocab_size', tok.vocab_size),
        block_size=rckpt['config'].get('block_size', tok.block_size),
        n_layer=rckpt['config'].get('n_layer', 4),
        n_head=rckpt['config'].get('n_head', 4),
        n_embd=rckpt['config'].get('n_embd', 256)
    ).to(device)
    rm.load_state_dict(rckpt['model'])
    rm.eval()

    opt = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.999))

    prompts_pool = sample_prompts(16)
    step = 0
    pool_idx = 0
    G = group_size

    while step < steps:
        P = max(1, batch_prompts)
        if pool_idx + P > len(prompts_pool):
            pool_idx = 0
        batch_prompts_list = prompts_pool[pool_idx: pool_idx + P]
        pool_idx += P

        prompt_texts = [format_prompt_only(p).replace("</s>", "") for p in batch_prompts_list]
        prompt_in_ids = [tok.encode(t) for t in prompt_texts]

        # Generate G completions per prompt
        seq_list = []
        boundary_list = []
        prompt_id_of = []
        raw_rewards = []

        with torch.no_grad():
            for pid, p_ids in enumerate(prompt_in_ids):
                for g in range(G):
                    idx = torch.tensor([p_ids], dtype=torch.long, device=device)
                    # Use policy generation
                    out = policy.generate(idx, max_new_tokens=resp_len, temperature=2, top_k=3)
                    full_ids = out[0].tolist()

                    boundary = len(p_ids) # Simple boundary, simplistic if clipping happened? 
                    # Actually original text uses clipped prompt length to context.
                    # We assume no clipping for simplicity here or handled by models. 
                    # Re-check logic in original code:
                    # boundary = len(p_ids[-block_size:])
                    
                    # Let's be safer:
                    p_len = len(p_ids)
                    if p_len > model_block_size:
                        p_len = model_block_size 
                        # This assumes generation handles context window, which it does in generate()
                    
                    resp_ids = full_ids[p_len:]
                    r_scalar = compute_reward(rm, tok, batch_prompts_list[pid], resp_ids, device)

                    seq_list.append(torch.tensor(full_ids, dtype=torch.long))
                    boundary_list.append(p_len)
                    prompt_id_of.append(pid)
                    raw_rewards.append(r_scalar)

        # Pad to batch
        B = len(seq_list)
        max_len = min(model_block_size, max(s.numel() for s in seq_list))
        seq = torch.zeros(B, max_len, dtype=torch.long, device=device)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        
        for i, (ids, bnd) in enumerate(zip(seq_list, boundary_list)):
            L_full = ids.numel()
            L = min(L_full, max_len)
            drop = L_full - L
            b = max(0, bnd - drop)
            seq[i, :L] = ids[-L:]
            if L < max_len:
                seq[i, L:] = 2 # pad
            mask[i, b:L] = True

        # Logprobs
        with torch.no_grad():
            pol_lp_full = model_logprobs(policy, seq)
            ref_lp_full = model_logprobs(ref, seq)

        act_mask = mask[:, 1:]
        old_logp = pol_lp_full[act_mask].detach()
        ref_logp = ref_lp_full[act_mask].detach()
        kl_tok = (old_logp - ref_logp)

        # Advantages
        traj_id_for_token = []
        counts = torch.zeros(B, dtype=torch.long, device=device)
        for i in range(B):
            n_i = int(act_mask[i].sum().item())
            if n_i > 0:
                traj_id_for_token.extend([i] * n_i)
            counts[i] = n_i
        traj_id_for_token = torch.tensor(traj_id_for_token, dtype=torch.long, device=device)
        raw_rewards_t = torch.tensor(raw_rewards, dtype=torch.float, device=device)

        group_mean = torch.zeros(B, dtype=torch.float, device=device)
        for pid in range(P):
            idxs = [i for i in range(B) if prompt_id_of[i] == pid]
            if not idxs: continue
            idxs_t = torch.tensor(idxs, dtype=torch.long, device=device)
            group_mean[idxs_t] = raw_rewards_t[idxs_t].mean()
        
        traj_adv = raw_rewards_t - group_mean
        
        if kl_tok.numel() > 0:
            adv_flat = traj_adv[traj_id_for_token]
        else:
            adv_flat = torch.zeros(0, dtype=torch.float, device=device)

        if adv_flat.numel() > 1:
            adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std().clamp_min(1e-6))

        # Update
        policy.train()
        logits_new, _, _ = policy(seq, None)
        logp_full = torch.log_softmax(logits_new[:, :-1, :], dim=-1)
        labels = seq[:, 1:]
        new_logp_all = logp_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        new_logp = new_logp_all[act_mask]

        kl_now_ref_mean = (new_logp - ref_logp).mean() if new_logp.numel() > 0 else torch.tensor(0.0, device=device)

        out_loss = ppo_policy_only_losses(
            new_logp=new_logp,
            old_logp=old_logp,
            adv=adv_flat,
            clip_ratio=0.2,
            ent_coef=0.0,
            kl_coef=kl_coef,
            kl_mean=kl_now_ref_mean,
        )
        loss = out_loss.total_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        policy.eval()

        step += 1
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    torch.save({'model': policy.state_dict(), 'config': {
        'vocab_size': vocab_size,
        'block_size': model_block_size,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
    }}, str(Path(out_dir)/'model_last.pt'))
    print(f"Saved GRPO policy to {out_dir}/model_last.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_ckpt', type=str, required=True)
    parser.add_argument('--reward_ckpt', type=str, required=True)
    parser.add_argument('--out', type=str, default='runs/grpo')
    args = parser.parse_args()
    
    train_grpo(
        policy_ckpt=args.policy_ckpt,
        reward_ckpt=args.reward_ckpt,
        out_dir=args.out
    )
