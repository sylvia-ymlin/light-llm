import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from ..model.reward import RewardModel
from ..data.collators import SFTCollator

def train_rm(pairs, out_dir="runs/rm_model", steps=100, batch_size=4, lr=1e-5):
    """
    Train a Reward Model using pairwise ranking loss (Bradley-Terry).
    pairs: list of (chosen, rejected) string tuples.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Model
    # simplistic config for demo; in prod allow config injection
    model = RewardModel(vocab_size=50304, block_size=1024, n_layer=4, n_head=4, n_embd=256)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # 2. Data
    # We reuse SFTCollator but we need to collate twice: once for chosen, once for rejected
    # Or strict implementation: creates a custom RM collator. 
    # For simplicity/demo: process one by one or minimal batching.
    
    model.train()
    pbar = tqdm(range(steps))
    
    for step in pbar:
        # Dummy batch sampling for demo if 'pairs' is small list
        batch_data = pairs[:batch_size] # just take first few for demo loop
        
        # Tokenization would happen here. 
        # For a "from scratch" project, we assume we have a tokenizer function available.
        # This is a placeholder structure to show the logic.
        
        # Pseudo-code for loss calculation:
        # r_chosen = model(chosen_ids)
        # r_rejected = model(rejected_ids)
        # loss = -F.logsigmoid(r_chosen - r_rejected).mean()
        
        # Implementation relying on valid inputs:
        optimizer.zero_grad()
        # ... logic ...
        # optimizer.step()
        
        # pbar.set_description(f"Loss: {loss.item():.4f}")

    print(f"Saving RM to {out_dir}")
    torch.save(model.state_dict(), os.path.join(out_dir, "reward_final.pt"))

# Note: This is a scaffold. To make it fully functional we need the tokenizer available in the training loop.
