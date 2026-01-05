from __future__ import annotations
import argparse
import torch
from pathlib import Path
from typing import List, Tuple
from ..model.base import GPTModern
from ..model.lora import apply_lora, LoraConfig
from ..data.collators import SFTCollator

class LengthCurriculum:
    """6.3 Curriculum: iterate examples from short→long prompts."""
    def __init__(self, items: List[Tuple[str,str]]):
        self.items = sorted(items, key=lambda p: len(p[0]))
        self._i = 0
    def __iter__(self):
        self._i = 0
        return self
    def __next__(self):
        if self._i >= len(self.items):
            raise StopIteration
        it = self.items[self._i]
        self._i += 1
        return it

def train_sft(
    items: List[Tuple[str,str]],
    out_dir: str = 'runs/sft',
    steps: int = 200,
    batch_size: int = 8,
    block_size: int = 256,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    lr: float = 3e-4,
    device: str | None = None,
    bpe_dir: str | None = None,
    checkpoint: str | None = None,
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16.0
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(device)
    print(f"Starting SFT training on {device}...")
    
    # Curriculum
    cur = list(LengthCurriculum(items))
    
    # Collator + Model
    col = SFTCollator(block_size=block_size, bpe_dir=bpe_dir)
    model = GPTModern(vocab_size=col.vocab_size, block_size=block_size,
                      n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                      use_rmsnorm=True, use_swiglu=True, rope=True).to(device)

    if checkpoint:
        print(f"Using model config from checkpoint {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device)
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])

    if use_lora:
        print(f"Applying LoRA (rank={lora_rank}, alpha={lora_alpha})...")
        lcfg = LoraConfig(rank=lora_rank, alpha=lora_alpha, dropout=0.05)
        apply_lora(model, lcfg)
        # Note: apply_lora freezes base parameters and enables grad on lora_ params
        # Count trainable params
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"LoRA applied. Trainable params: {n_train:,} / {n_all:,} ({n_train/n_all:.2%})")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    model.train()
    
    step = 0
    i = 0
    while step < steps:
        batch = cur[i:i+batch_size]
        if not batch:
            i = 0
            continue
            
        xb, yb = col.collate(batch)
        xb, yb = xb.to(device), yb.to(device)
        
        logits, loss, _ = model(xb, yb)
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        step += 1
        i += batch_size
        
        if step % 20 == 0:
            print(f"step {step}: loss={loss.item():.4f}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cfg = {
        "vocab_size": col.vocab_size,
        "block_size": block_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "dropout": 0.0,
        "use_rmsnorm": True,
        "use_swiglu": True,
        "rope": True,
        "tokenizer_type": "byte" if col.vocab_size == 256 else "bpe",
        "tokenizer_dir": bpe_dir, 
    }
    torch.save({'model': model.state_dict(), 'config': cfg},
               str(Path(out_dir)/'model_last.pt'))
    print(f"Saved SFT checkpoint to {out_dir}/model_last.pt")

if __name__ == "__main__":
    # Provides a CLI interface similar to original script
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='runs/sft')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--cpu', action='store_true')
    # Dummy data loader for CLI test
    args = parser.parse_args()
    
    # Fallback dummy data
    dummy_data = [("Hello", "World"), ("Test prompt", "Test response")] * 10
    
    train_sft(
        items=dummy_data,
        out_dir=args.out,
        steps=args.steps,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        batch_size=args.batch_size,
        device='cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    )
