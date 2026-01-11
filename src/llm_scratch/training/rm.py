from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from pathlib import Path
from typing import List, Tuple
import random

from ..model.reward import RewardModel
from ..data.tokenizers import BPETokenizer, ByteTokenizer
from ..data.formatting import format_example, Example

class RewardDataCollator:
    """Collator for reward model training with pairwise preference data."""
    
    def __init__(self, block_size: int = 512, bpe_dir: str | None = None):
        self.block_size = block_size
        self.tok = None
        
        # Try BPE first, fallback to Byte tokenizer
        try:
            self.tok = BPETokenizer(vocab_size=8000)
            if bpe_dir:
                self.tok.load(bpe_dir)
            else:
                self.tok = None
        except Exception:
            self.tok = None
            
        if self.tok is None:
            self.tok = ByteTokenizer()
    
    @property
    def vocab_size(self) -> int:
        return getattr(self.tok, 'vocab_size', 256)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        if hasattr(self.tok, 'encode'):
            ids = self.tok.encode(text)
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return ids
        return list(text.encode('utf-8'))
    
    def collate_pair(self, prompt: str, chosen: str, rejected: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate a single preference pair into tensors.
        Returns: (chosen_ids, rejected_ids) both padded to block_size
        """
        # Format full conversations
        chosen_text = format_example(Example(prompt, chosen))
        rejected_text = format_example(Example(prompt, rejected))
        
        # Tokenize and truncate
        chosen_ids = self.encode(chosen_text)[:self.block_size]
        rejected_ids = self.encode(rejected_text)[:self.block_size]
        
        # Pad to block_size (pad token = 2)
        def pad_to_size(ids: List[int]) -> List[int]:
            if len(ids) < self.block_size:
                ids = ids + [2] * (self.block_size - len(ids))
            return ids[:self.block_size]
        
        chosen_padded = pad_to_size(chosen_ids)
        rejected_padded = pad_to_size(rejected_ids)
        
        return (torch.tensor(chosen_padded, dtype=torch.long),
                torch.tensor(rejected_padded, dtype=torch.long))
    
    def collate_batch(self, batch: List[Tuple[str, str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate a batch of (prompt, chosen, rejected) tuples.
        Returns: (chosen_batch, rejected_batch) tensors of shape (B, T)
        """
        chosen_batch = []
        rejected_batch = []
        
        for prompt, chosen, rejected in batch:
            chosen_ids, rejected_ids = self.collate_pair(prompt, chosen, rejected)
            chosen_batch.append(chosen_ids)
            rejected_batch.append(rejected_ids)
        
        return (torch.stack(chosen_batch), torch.stack(rejected_batch))

class PreferenceDataset(Dataset):
    """Dataset for preference pairs (prompt, chosen, rejected)."""
    
    def __init__(self, pairs: List[Tuple[str, str, str]]):
        """
        pairs: List of (prompt, chosen_response, rejected_response) tuples
        """
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]

def create_dummy_preference_data(n_pairs: int = 100) -> List[Tuple[str, str, str]]:
    """Create dummy preference data for testing."""
    prompts = [
        "Explain the concept of machine learning.",
        "What are the benefits of renewable energy?",
        "How does photosynthesis work?",
        "Describe the process of making coffee.",
        "What is the importance of exercise?",
        "Explain how the internet works.",
        "What are the main causes of climate change?",
        "How do you solve a quadratic equation?",
    ]
    
    good_responses = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        "Renewable energy sources like solar and wind are sustainable, reduce greenhouse gas emissions, and help decrease dependence on fossil fuels.",
        "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll.",
        "Coffee is made by grinding roasted coffee beans and extracting their flavors with hot water through various brewing methods.",
        "Regular exercise improves cardiovascular health, strengthens muscles, boosts mental well-being, and helps maintain a healthy weight.",
        "The internet is a global network of interconnected computers that communicate using standardized protocols to share information.",
        "Climate change is primarily caused by greenhouse gas emissions from burning fossil fuels, deforestation, and industrial activities.",
        "A quadratic equation ax²+bx+c=0 can be solved using the quadratic formula: x = (-b ± √(b²-4ac)) / 2a.",
    ]
    
    bad_responses = [
        "Machine learning is just computers doing stuff automatically.",
        "Renewable energy is okay I guess but it's expensive.",
        "Plants eat sunlight and make oxygen somehow.",
        "You put coffee in water and drink it.",
        "Exercise is good for you because it makes you tired.",
        "The internet is made of cables and WiFi signals.",
        "Climate change happens because the weather changes.",
        "You solve quadratic equations by guessing the answer.",
    ]
    
    pairs = []
    for i in range(n_pairs):
        prompt_idx = i % len(prompts)
        prompt = prompts[prompt_idx]
        chosen = good_responses[prompt_idx]
        rejected = bad_responses[prompt_idx]
        
        # Add some variation
        if random.random() < 0.1:  # 10% chance to swap for harder examples
            chosen, rejected = rejected, chosen
            
        pairs.append((prompt, chosen, rejected))
    
    return pairs

def train_rm(
    pairs: List[Tuple[str, str, str]] | None = None,
    out_dir: str = "runs/rm_model",
    steps: int = 200,
    batch_size: int = 8,
    block_size: int = 512,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    lr: float = 1e-5,
    device: str | None = None,
    bpe_dir: str | None = None,
    save_every: int = 50
):
    """
    Train a Reward Model using pairwise ranking loss (Bradley-Terry).
    
    Args:
        pairs: List of (prompt, chosen_response, rejected_response) tuples
        out_dir: Output directory for checkpoints
        steps: Number of training steps
        batch_size: Batch size for training
        block_size: Maximum sequence length
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        lr: Learning rate
        device: Device to train on (auto-detect if None)
        bpe_dir: Directory containing BPE tokenizer files
        save_every: Save checkpoint every N steps
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(device)
    print(f"Starting Reward Model training on {device}...")
    
    # Create dummy data if none provided
    if pairs is None:
        print("No preference data provided, creating dummy data for demonstration...")
        pairs = create_dummy_preference_data(n_pairs=200)
    
    print(f"Training on {len(pairs)} preference pairs")
    
    # Data collator
    collator = RewardDataCollator(block_size=block_size, bpe_dir=bpe_dir)
    
    # Model
    model = RewardModel(
        vocab_size=collator.vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.1
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    model.train()
    
    # Training loop
    step = 0
    total_loss = 0.0
    
    pbar = tqdm(range(steps), desc="Training RM")
    
    while step < steps:
        # Sample a batch
        batch_pairs = random.sample(pairs, min(batch_size, len(pairs)))
        
        # Collate batch
        chosen_batch, rejected_batch = collator.collate_batch(batch_pairs)
        chosen_batch = chosen_batch.to(device)
        rejected_batch = rejected_batch.to(device)
        
        # Mixed precision forward pass
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        if device.type == 'mps':
            device_type = 'mps'
            
        with torch.amp.autocast(device_type=device_type, enabled=True):
            # Get reward scores
            r_chosen = model(chosen_batch)      # (B,)
            r_rejected = model(rejected_batch)  # (B,)
            
            # Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))
            # Equivalent to: -F.logsigmoid(r_chosen - r_rejected)
            loss = -F.logsigmoid(r_chosen - r_rejected).mean()
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        step += 1
        
        # Update progress bar
        if step % 10 == 0:
            avg_loss = total_loss / step
            pbar.set_description(f"Training RM | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")
        
        # Save checkpoint
        if step % save_every == 0:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            config = {
                'vocab_size': collator.vocab_size,
                'block_size': block_size,
                'n_layer': n_layer,
                'n_head': n_head,
                'n_embd': n_embd,
                'tokenizer_type': 'byte' if collator.vocab_size == 256 else 'bpe',
                'tokenizer_dir': bpe_dir,
            }
            torch.save({
                'model': model.state_dict(),
                'config': config,
                'step': step,
                'loss': loss.item()
            }, str(Path(out_dir) / f'model_step_{step}.pt'))
        
        pbar.update(1)
    
    pbar.close()
    
    # Save final model
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    config = {
        'vocab_size': collator.vocab_size,
        'block_size': block_size,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
        'tokenizer_type': 'byte' if collator.vocab_size == 256 else 'bpe',
        'tokenizer_dir': bpe_dir,
    }
    
    torch.save({
        'model': model.state_dict(),
        'config': config,
        'step': steps,
        'avg_loss': total_loss / steps
    }, str(Path(out_dir) / 'model_final.pt'))
    
    print(f"Reward Model training completed!")
    print(f"Final average loss: {total_loss / steps:.4f}")
    print(f"Model saved to: {out_dir}/model_final.pt")
    
    return model

def load_reward_model(checkpoint_path: str, device: str | None = None) -> RewardModel:
    """Load a trained reward model from checkpoint."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt['config']
    
    model = RewardModel(
        vocab_size=config['vocab_size'],
        block_size=config['block_size'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_embd=config['n_embd']
    ).to(device)
    
    model.load_state_dict(ckpt['model'])
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Reward Model")
    parser.add_argument('--out', type=str, default='runs/rm_demo', help='Output directory')
    parser.add_argument('--steps', type=int, default=200, help='Training steps')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')
    parser.add_argument('--bpe_dir', type=str, help='BPE tokenizer directory')
    
    args = parser.parse_args()
    
    device = 'cpu' if args.cpu else None
    
    # Train with dummy data
    train_rm(
        pairs=None,  # Will create dummy data
        out_dir=args.out,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        bpe_dir=args.bpe_dir
    )
