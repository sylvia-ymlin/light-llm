from __future__ import annotations
from typing import List, Tuple
import torch
import traceback
from .tokenizers import BPETokenizer, ByteTokenizer
from .formatting import Example, format_example, format_prompt_only

class SFTCollator:
    """Turn (instruction,response) into token ids and masked labels for causal LM.
    Labels for the prompt part are set to -100 so they don't contribute to loss.
    """
    def __init__(self, block_size: int = 256, bpe_dir: str | None = None):
        self.block_size = block_size
        self.tok = None
        # Try BPE
        try:
            self.tok = BPETokenizer(vocab_size=8000)
            if bpe_dir:
                self.tok.load(bpe_dir)
            else:
                # Fallback to failing BPE init if dir not provided
                self.tok = None
        except Exception:
            self.tok = None
        
        # Fallback to Byte
        if self.tok is None:
            self.tok = ByteTokenizer()

    @property
    def vocab_size(self) -> int:
        return getattr(self.tok, 'vocab_size', 256)

    def encode(self, text: str) -> List[int]:
        if hasattr(self.tok, 'encode'):
            ids = self.tok.encode(text)
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return ids
        return list(text.encode('utf-8'))

    def collate(self, batch: List[Tuple[str,str]]):
        # Build "prompt + response" and create label mask where prompt positions are -100.
        input_ids = []
        labels = []
        for prompt, response in batch:
            prefix_text = format_prompt_only(prompt).replace('</s>','')
            text = format_example(Example(prompt, response))
            ids = self.encode(text)[:self.block_size]
            prompt_ids = self.encode(prefix_text)[:self.block_size]
            n_prompt = min(len(prompt_ids), len(ids))
            
            x = ids
            y = ids.copy()
            # causal shift is handled in loss usually, but here y matches x input?
            # Wait, typical CLM: input x_t, target x_{t+1}.
            # Existing collator logic in part_6:
            # y[t] = ids[t+1]
            # y[-1] = -100
            # So input is ids, target is ids shifted left.
            
            # The code I read from part_6/collator_sft.py:
            # x = ids
            # y = ids.copy()
            # for t in range(len(y) - 1):
            #     y[t] = ids[t + 1]
            # y[-1] = -100
            
            # Replicating that exact logic:
            for t in range(len(y) - 1):
                y[t] = ids[t + 1]
            y[-1] = -100
            
            # Mask prompt
            for i in range(n_prompt-1):
                y[i] = -100
            input_ids.append(x)
            labels.append(y)
            
        def pad_to(ids, val):
            if len(ids) < self.block_size:
                ids = ids + [val]*(self.block_size - len(ids))
            return ids[:self.block_size]
            
        x = torch.tensor([pad_to(s, 2) for s in input_ids], dtype=torch.long)
        y = torch.tensor([pad_to(s, -100) for s in labels], dtype=torch.long)
        return x, y
