#!/usr/bin/env python3
"""
Quick test of reward model training and validation.
"""

import torch
from llm_scratch.training.rm import train_rm, create_dummy_preference_data, load_reward_model
from llm_scratch.training.rm import RewardDataCollator

def quick_test():
    """Quick test of reward model functionality."""
    print("🚀 Quick Reward Model Test")
    print("=" * 40)
    
    # 1. Create small dataset
    print("📊 Creating test data...")
    pairs = create_dummy_preference_data(n_pairs=20)
    print(f"Created {len(pairs)} preference pairs")
    
    # 2. Train small model quickly
    print("\n🏋️ Training small model...")
    model = train_rm(
        pairs=pairs,
        out_dir="runs/quick_rm_test",
        steps=20,
        batch_size=4,
        block_size=128,
        n_layer=2,
        n_head=2,
        n_embd=64,
        lr=1e-4,
        device='cpu'
    )
    
    print("✅ Training completed!")
    
    # 3. Test the model
    print("\n🧪 Testing model...")
    test_model("runs/quick_rm_test/model_final.pt")
    
    print("\n🎉 Quick test completed successfully!")

def test_model(checkpoint_path: str):
    """Test the trained model."""
    model = load_reward_model(checkpoint_path, device='cpu')
    model.eval()
    
    collator = RewardDataCollator(block_size=128)
    
    # Test cases with clear quality differences
    test_cases = [
        ("What is AI?",
         "Artificial Intelligence is a field of computer science that creates systems capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving.",
         "AI is computers."),
        
        ("Explain gravity.",
         "Gravity is a fundamental force of nature that causes objects with mass to attract each other. Einstein's theory describes it as the curvature of spacetime.",
         "Gravity makes things fall."),
    ]
    
    print("🔍 Testing preference learning:")
    print("-" * 40)
    
    correct = 0
    total = len(test_cases)
    
    with torch.no_grad():
        for i, (prompt, good, bad) in enumerate(test_cases):
            good_ids, bad_ids = collator.collate_pair(prompt, good, bad)
            
            r_good = model(good_ids.unsqueeze(0)).item()
            r_bad = model(bad_ids.unsqueeze(0)).item()
            
            is_correct = r_good > r_bad
            if is_correct:
                correct += 1
            
            print(f"Test {i+1}: {'✅' if is_correct else '❌'}")
            print(f"  Good: {r_good:.4f} | Bad: {r_bad:.4f} | Margin: {r_good-r_bad:+.4f}")
    
    accuracy = correct / total
    print(f"\n📊 Accuracy: {accuracy:.1%} ({correct}/{total})")
    
    if accuracy >= 0.5:
        print("✅ Model shows preference learning!")
    else:
        print("⚠️ Model needs more training or better data")

if __name__ == "__main__":
    quick_test()