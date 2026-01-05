# LLM From Scratch (Independent Implementation)

This project is a standalone implementation of a Large Language Model (LLM) including pre-training (SFT) and Reinforcement Learning from Human Feedback (RLHF) using GRPO.

Is was refactored from a tutorial series into a cohesive Python package.

## Project Structure

- `src/llm_scratch/`: Main package
  - `model/`: Transformer architecture (RoPE, SwiGLU, RMSNorm, GQA, etc.)
  - `training/`: Training loops (SFT, RLHF/GRPO)
  - `data/`: Tokenizers and formatting utilities
- `tests/`: Unit tests

## Installation

```bash
pip install -e .[dev]
```

## Usage

### Supervised Fine-Tuning (SFT)

To train a model on a sample dataset:

```python
from llm_scratch.training.sft import train_sft
from llm_scratch.data.collators import SFTCollator

# Dummy data or load from list
data = [("Instruction", "Response")] * 100

train_sft(items=data, out_dir="runs/sft_demo", steps=10)
```

### Reinforcement Learning (RLHF)

Requires a trained SFT checkpoint and a Reward Model checkpoint.

```python
from llm_scratch.training.rl import train_grpo

train_grpo(
    policy_ckpt="runs/sft_demo/model_last.pt",
    reward_ckpt="path/to/reward_model.pt",
    out_dir="runs/grpo_demo"
)
```

## Running Tests

```bash
python -m pytest tests/
```
