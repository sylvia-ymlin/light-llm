#!/usr/bin/env python3
"""
Complete end-to-end demonstration of the LLM training pipeline:
1. Supervised Fine-Tuning (SFT)
2. Reward Model Training
3. Reinforcement Learning from Human Feedback (RLHF) with GRPO
"""

import torch
from pathlib import Path
import time

from llm_scratch.training.sft import train_sft
from llm_scratch.training.rm import train_rm, create_dummy_preference_data
from llm_scratch.training.rl import train_grpo
from llm_scratch.model.base import GPTModern
from llm_scratch.data.tokenizers import ByteTokenizer

def create_sft_data():
    """Create supervised fine-tuning data."""
    sft_pairs = [
        # Instruction following
        ("Write a haiku about programming.", 
         "Code flows like water\nBugs hide in silent corners\nDebug with patience"),
        
        ("Explain recursion simply.",
         "Recursion is when a function calls itself to solve a smaller version of the same problem, like Russian dolls inside dolls."),
        
        ("What is the capital of France?",
         "The capital of France is Paris, a beautiful city known for its art, culture, and the Eiffel Tower."),
        
        ("How do you make coffee?",
         "To make coffee: 1) Grind coffee beans, 2) Heat water to 200°F, 3) Pour water over grounds, 4) Let it brew for 4 minutes, 5) Enjoy!"),
        
        ("Write a short story about a robot.",
         "BEEP-7 was a cleaning robot who dreamed of painting. One day, he spilled colorful cleaning fluids and created his first masterpiece on the floor."),
        
        # Math and science
        ("What is 15 + 27?",
         "15 + 27 = 42"),
        
        ("Explain photosynthesis.",
         "Photosynthesis is how plants make food using sunlight, water, and carbon dioxide, producing oxygen as a byproduct."),
        
        ("What is gravity?",
         "Gravity is the force that attracts objects toward each other. On Earth, it pulls everything toward the center of the planet."),
        
        # Creative tasks
        ("Write a limerick about cats.",
         "There once was a cat from Peru\nWho dreamed of a mouse he could chew\nHe pounced with great might\nBut missed in his flight\nAnd landed face-first in his stew"),
        
        ("Describe a sunset.",
         "The sunset painted the sky in brilliant oranges and purples, as the golden sun slowly disappeared behind the distant mountains."),
    ]
    
    # Expand dataset
    expanded_data = sft_pairs * 10  # 100 examples
    return expanded_data

def demo_sft_training():
    """Demonstrate supervised fine-tuning."""
    print("🎯 Step 1: Supervised Fine-Tuning (SFT)")
    print("=" * 50)
    
    # Create training data
    sft_data = create_sft_data()
    print(f"Created {len(sft_data)} SFT training examples")
    
    # Show sample
    print(f"\nSample: {sft_data[0][0]} -> {sft_data[0][1][:50]}...")
    
    # Train SFT model
    print("\n🏋️ Training SFT model...")
    train_sft(
        items=sft_data,
        out_dir="runs/pipeline_demo/sft",
        steps=50,
        batch_size=4,
        block_size=256,
        n_layer=4,
        n_head=4,
        n_embd=128,
        lr=3e-4
    )
    
    print("✅ SFT training completed!")
    return "runs/pipeline_demo/sft/model_last.pt"

def demo_reward_model_training():
    """Demonstrate reward model training."""
    print("\n🏆 Step 2: Reward Model Training")
    print("=" * 50)
    
    # Create preference data
    preference_data = create_dummy_preference_data(n_pairs=100)
    print(f"Created {len(preference_data)} preference pairs")
    
    # Show sample
    prompt, chosen, rejected = preference_data[0]
    print(f"\nSample preference:")
    print(f"Prompt: {prompt}")
    print(f"Chosen: {chosen[:50]}...")
    print(f"Rejected: {rejected[:50]}...")
    
    # Train reward model
    print("\n🏋️ Training reward model...")
    train_rm(
        pairs=preference_data,
        out_dir="runs/pipeline_demo/reward_model",
        steps=100,
        batch_size=4,
        block_size=256,
        n_layer=4,
        n_head=4,
        n_embd=128,
        lr=5e-5
    )
    
    print("✅ Reward model training completed!")
    return "runs/pipeline_demo/reward_model/model_final.pt"

def demo_rlhf_training(sft_checkpoint: str, reward_checkpoint: str):
    """Demonstrate RLHF training with GRPO."""
    print("\n🚀 Step 3: RLHF Training (GRPO)")
    print("=" * 50)
    
    print(f"Using SFT checkpoint: {sft_checkpoint}")
    print(f"Using reward checkpoint: {reward_checkpoint}")
    
    # Train with GRPO
    print("\n🏋️ Training with GRPO...")
    train_grpo(
        policy_ckpt=sft_checkpoint,
        reward_ckpt=reward_checkpoint,
        out_dir="runs/pipeline_demo/rlhf",
        steps=50,
        batch_prompts=8,
        group_size=4,
        block_size=256,
        resp_len=64,
        kl_coef=0.01,
        lr=1e-5
    )
    
    print("✅ RLHF training completed!")
    return "runs/pipeline_demo/rlhf/model_last.pt"

def test_final_model(model_path: str):
    """Test the final RLHF-trained model."""
    print("\n🧪 Step 4: Testing Final Model")
    print("=" * 50)
    
    # Load model
    device = torch.device('cpu')  # Use CPU for demo
    ckpt = torch.load(model_path, map_location=device)
    config = ckpt['config']
    
    model = GPTModern(
        vocab_size=config['vocab_size'],
        block_size=config['block_size'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_embd=config['n_embd']
    ).to(device)
    
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # Test tokenizer
    tokenizer = ByteTokenizer()
    
    # Test prompts
    test_prompts = [
        "What is machine learning?",
        "Write a haiku about AI.",
        "Explain gravity simply.",
    ]
    
    print("🔍 Generating responses...")
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {prompt}")
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
        
        # Generate response
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.8,
                top_k=10
            )
        
        # Decode response
        response_ids = output[0][input_ids.size(1):].tolist()
        response = tokenizer.decode(response_ids)
        
        print(f"Response: {response}")

def benchmark_pipeline():
    """Benchmark the complete pipeline."""
    print("\n⚡ Pipeline Benchmarks")
    print("=" * 50)
    
    # Model size comparison
    config = dict(vocab_size=256, block_size=256, n_layer=4, n_head=4, n_embd=128)
    model = GPTModern(**config)
    
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    
    print(f"📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: {model_size_mb:.2f} MB")
    
    # Training time estimates (very rough)
    print(f"\n⏱️ Estimated Training Times (CPU):")
    print(f"  SFT (50 steps): ~2-3 minutes")
    print(f"  Reward Model (100 steps): ~3-4 minutes") 
    print(f"  RLHF (50 steps): ~4-5 minutes")
    print(f"  Total pipeline: ~10-12 minutes")

def main():
    """Run the complete pipeline demonstration."""
    print("🚀 LLM Training Pipeline Demo")
    print("=" * 60)
    print("This demo shows the complete pipeline:")
    print("1. Supervised Fine-Tuning (SFT)")
    print("2. Reward Model Training")
    print("3. Reinforcement Learning from Human Feedback (RLHF)")
    print("4. Model Testing")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Step 1: SFT
        sft_checkpoint = demo_sft_training()
        
        # Step 2: Reward Model
        reward_checkpoint = demo_reward_model_training()
        
        # Step 3: RLHF
        final_checkpoint = demo_rlhf_training(sft_checkpoint, reward_checkpoint)
        
        # Step 4: Test final model
        test_final_model(final_checkpoint)
        
        # Benchmarks
        benchmark_pipeline()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 Pipeline Demo Completed Successfully!")
        print(f"⏱️ Total time: {total_time/60:.1f} minutes")
        print(f"📁 All models saved in: runs/pipeline_demo/")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise

if __name__ == "__main__":
    main()