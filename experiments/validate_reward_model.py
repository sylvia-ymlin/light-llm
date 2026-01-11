#!/usr/bin/env python3
"""
Validation script for reward model training.
Tests whether the reward model learns to prefer better responses.
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from llm_scratch.training.rm import train_rm, create_dummy_preference_data, load_reward_model
from llm_scratch.training.rm import RewardDataCollator

def create_validation_data():
    """Create clear preference pairs for validation."""
    validation_pairs = [
        # Clear quality differences
        ("What is machine learning?",
         "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions.",
         "ML is computers learning stuff."),
        
        ("Explain photosynthesis.",
         "Photosynthesis is the process by which plants convert light energy, carbon dioxide, and water into glucose and oxygen using chlorophyll in their chloroplasts. This process is essential for life on Earth.",
         "Plants eat sunlight."),
        
        ("How do you solve a quadratic equation?",
         "A quadratic equation ax²+bx+c=0 can be solved using the quadratic formula: x = (-b ± √(b²-4ac)) / 2a. You can also factor or complete the square.",
         "You guess the answer."),
        
        ("What causes rain?",
         "Rain is caused by the water cycle: water evaporates from oceans and lakes, forms clouds through condensation, and falls as precipitation when water droplets become too heavy.",
         "Clouds get full and leak."),
        
        ("Explain gravity.",
         "Gravity is a fundamental force that causes objects with mass to attract each other. Einstein's theory describes it as the curvature of spacetime caused by mass and energy.",
         "Gravity makes things fall."),
    ]
    
    return validation_pairs

def train_and_validate_reward_model():
    """Train a reward model and validate its performance."""
    print("🧪 Reward Model Validation Experiment")
    print("=" * 50)
    
    # 1. Create training data
    print("📊 Creating training data...")
    training_pairs = create_dummy_preference_data(n_pairs=50)
    validation_pairs = create_validation_data()
    
    print(f"Training pairs: {len(training_pairs)}")
    print(f"Validation pairs: {len(validation_pairs)}")
    
    # 2. Train reward model
    print("\n🏋️ Training reward model...")
    out_dir = "runs/reward_validation"
    
    model = train_rm(
        pairs=training_pairs,
        out_dir=out_dir,
        steps=200,
        batch_size=8,
        block_size=256,
        n_layer=4,
        n_head=4,
        n_embd=128,
        lr=5e-5,
        device='cpu',
        save_every=50
    )
    
    # 3. Validate on held-out data
    print("\n🔍 Validating model performance...")
    validate_model_performance(f"{out_dir}/model_final.pt", validation_pairs)
    
    # 4. Analyze training checkpoints
    print("\n📈 Analyzing training progress...")
    analyze_training_progress(out_dir, validation_pairs)

def validate_model_performance(checkpoint_path: str, validation_pairs):
    """Validate model performance on clear preference pairs."""
    model = load_reward_model(checkpoint_path, device='cpu')
    model.eval()
    
    collator = RewardDataCollator(block_size=256)
    
    correct_preferences = 0
    total_pairs = len(validation_pairs)
    margins = []
    
    print("\n📋 Validation Results:")
    print("-" * 60)
    
    with torch.no_grad():
        for i, (prompt, good_response, bad_response) in enumerate(validation_pairs):
            # Get rewards
            good_ids, bad_ids = collator.collate_pair(prompt, good_response, bad_response)
            r_good = model(good_ids.unsqueeze(0)).item()
            r_bad = model(bad_ids.unsqueeze(0)).item()
            
            margin = r_good - r_bad
            margins.append(margin)
            
            is_correct = r_good > r_bad
            if is_correct:
                correct_preferences += 1
            
            print(f"Pair {i+1}: {'✅' if is_correct else '❌'} | "
                  f"Good: {r_good:.4f} | Bad: {r_bad:.4f} | "
                  f"Margin: {margin:+.4f}")
    
    accuracy = correct_preferences / total_pairs
    avg_margin = np.mean(margins)
    
    print(f"\n📊 Summary:")
    print(f"Accuracy: {accuracy:.2%} ({correct_preferences}/{total_pairs})")
    print(f"Average margin: {avg_margin:+.4f}")
    print(f"Margin std: {np.std(margins):.4f}")
    
    return accuracy, avg_margin

def analyze_training_progress(out_dir: str, validation_pairs):
    """Analyze how model performance improves during training."""
    checkpoints = list(Path(out_dir).glob("model_step_*.pt"))
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
    
    if not checkpoints:
        print("No intermediate checkpoints found.")
        return
    
    steps = []
    accuracies = []
    margins = []
    
    print(f"\n📈 Found {len(checkpoints)} checkpoints")
    
    for ckpt_path in checkpoints:
        step = int(ckpt_path.stem.split('_')[-1])
        print(f"Evaluating step {step}...")
        
        try:
            accuracy, avg_margin = validate_model_performance(str(ckpt_path), validation_pairs)
            steps.append(step)
            accuracies.append(accuracy)
            margins.append(avg_margin)
        except Exception as e:
            print(f"Error evaluating {ckpt_path}: {e}")
            continue
    
    # Plot training progress
    if steps:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        ax1.plot(steps, accuracies, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Validation Accuracy')
        ax1.set_title('Reward Model Accuracy vs Training Steps')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Margin plot
        ax2.plot(steps, margins, 'r-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Average Preference Margin')
        ax2.set_title('Preference Margin vs Training Steps')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plot_path = Path(out_dir) / 'training_progress.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 Training progress plot saved to: {plot_path}")
        plt.close()

def benchmark_reward_model_speed():
    """Benchmark reward model inference speed."""
    print("\n⚡ Benchmarking Reward Model Speed")
    print("-" * 40)
    
    # Create a test model
    from llm_scratch.model.reward import RewardModel
    model = RewardModel(vocab_size=256, block_size=256, n_layer=4, n_head=4, n_embd=128)
    model.eval()
    
    # Test data
    batch_sizes = [1, 4, 8, 16]
    seq_len = 256
    
    print(f"{'Batch Size':<12} {'Time (ms)':<12} {'Throughput':<15}")
    print("-" * 40)
    
    import time
    
    for batch_size in batch_sizes:
        x = torch.randint(0, 256, (batch_size, seq_len))
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)
        
        # Benchmark
        start_time = time.time()
        n_runs = 50
        
        for _ in range(n_runs):
            with torch.no_grad():
                _ = model(x)
        
        end_time = time.time()
        avg_time_ms = (end_time - start_time) / n_runs * 1000
        throughput = batch_size / (avg_time_ms / 1000)
        
        print(f"{batch_size:<12} {avg_time_ms:<12.2f} {throughput:<15.1f}")

def main():
    """Run the complete validation experiment."""
    try:
        train_and_validate_reward_model()
        benchmark_reward_model_speed()
        print("\n🎉 Reward model validation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        raise

if __name__ == "__main__":
    main()