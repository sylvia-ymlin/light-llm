#!/usr/bin/env python3
"""
Demo script for training a reward model from scratch.
This demonstrates the complete pipeline from data preparation to model training.
"""

import torch
from pathlib import Path
from llm_scratch.training.rm import train_rm, create_dummy_preference_data, load_reward_model
from llm_scratch.training.rm import RewardDataCollator

def create_realistic_preference_data():
    """Create more realistic preference data for demonstration."""
    
    # More diverse and realistic examples
    preference_pairs = [
        # Technical explanations
        ("Explain how neural networks work.",
         "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process information through weighted connections and activation functions.",
         "Neural networks are just math that makes computers smart."),
        
        ("What is machine learning?",
         "Machine learning is a subset of artificial intelligence that enables systems to automatically learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data.",
         "Machine learning is when computers learn stuff automatically."),
        
        # Science questions
        ("How does photosynthesis work?",
         "Photosynthesis is a complex biochemical process where plants convert light energy, carbon dioxide, and water into glucose and oxygen using chlorophyll in their chloroplasts.",
         "Plants eat sunlight and make oxygen."),
        
        ("Explain gravity.",
         "Gravity is a fundamental force of nature that causes objects with mass to attract each other. Einstein's theory describes it as the curvature of spacetime caused by mass and energy.",
         "Gravity makes things fall down."),
        
        # Programming questions
        ("What is recursion in programming?",
         "Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem, typically with a base case to prevent infinite loops.",
         "Recursion is when a function calls itself over and over."),
        
        ("Explain object-oriented programming.",
         "Object-oriented programming (OOP) is a programming paradigm based on objects that contain data (attributes) and code (methods). Key principles include encapsulation, inheritance, and polymorphism.",
         "OOP is programming with objects and classes."),
        
        # History and culture
        ("What caused World War I?",
         "World War I was triggered by the assassination of Archduke Franz Ferdinand, but underlying causes included imperialism, nationalism, militarism, and complex alliance systems in Europe.",
         "World War I started because someone got shot."),
        
        ("Explain the Renaissance.",
         "The Renaissance was a cultural movement in Europe from the 14th to 17th centuries, characterized by renewed interest in classical learning, humanism, artistic innovation, and scientific advancement.",
         "The Renaissance was when art got better."),
        
        # Health and medicine
        ("How do vaccines work?",
         "Vaccines work by introducing antigens that stimulate the immune system to produce antibodies and memory cells, providing protection against future infections without causing the disease.",
         "Vaccines give you a little bit of the disease so you don't get sick."),
        
        ("What is DNA?",
         "DNA (deoxyribonucleic acid) is a double-helix molecule that carries genetic instructions for all living organisms, composed of nucleotides containing bases adenine, thymine, guanine, and cytosine.",
         "DNA is the stuff that makes you who you are."),
    ]
    
    # Expand the dataset by creating variations
    expanded_pairs = []
    for prompt, good, bad in preference_pairs:
        expanded_pairs.append((prompt, good, bad))
        
        # Add some variations with swapped quality (harder examples)
        if len(expanded_pairs) % 10 == 0:  # 10% harder examples
            expanded_pairs.append((prompt, bad, good))
    
    # Duplicate to get more training data
    final_pairs = expanded_pairs * 5  # 50+ examples total
    
    return final_pairs

def demo_reward_model_training():
    """Run a complete reward model training demonstration."""
    
    print("🚀 Starting Reward Model Training Demo")
    print("=" * 50)
    
    # 1. Create preference data
    print("\n📊 Creating preference dataset...")
    preference_data = create_realistic_preference_data()
    print(f"Created {len(preference_data)} preference pairs")
    
    # Show a sample
    print("\n📝 Sample preference pair:")
    prompt, chosen, rejected = preference_data[0]
    print(f"Prompt: {prompt}")
    print(f"Chosen: {chosen[:100]}...")
    print(f"Rejected: {rejected[:100]}...")
    
    # 2. Train the reward model
    print("\n🏋️ Training reward model...")
    out_dir = "runs/reward_model_demo"
    
    model = train_rm(
        pairs=preference_data,
        out_dir=out_dir,
        steps=100,
        batch_size=4,
        block_size=256,
        n_layer=4,
        n_head=4,
        n_embd=128,
        lr=5e-5,
        device='cpu',  # Force CPU for demo stability
        save_every=25
    )
    
    print(f"\n✅ Training completed! Model saved to {out_dir}")
    
    # 3. Test the trained model
    print("\n🧪 Testing trained model...")
    test_reward_model(f"{out_dir}/model_final.pt")
    
    print("\n🎉 Demo completed successfully!")

def test_reward_model(checkpoint_path: str):
    """Test a trained reward model with sample inputs."""
    
    print(f"Loading model from {checkpoint_path}")
    model = load_reward_model(checkpoint_path, device='cpu')  # Force CPU for testing
    model.eval()
    
    # Create test cases
    test_cases = [
        ("What is AI?", 
         "Artificial Intelligence is a field of computer science focused on creating systems that can perform tasks typically requiring human intelligence.",
         "AI is just computers doing stuff."),
        
        ("Explain quantum physics.",
         "Quantum physics is the branch of physics that studies matter and energy at the smallest scales, where particles exhibit wave-particle duality and quantum superposition.",
         "Quantum physics is weird science stuff."),
    ]
    
    collator = RewardDataCollator(block_size=256)
    
    print("\n🔍 Reward scores for test cases:")
    print("-" * 40)
    
    with torch.no_grad():
        for i, (prompt, good_response, bad_response) in enumerate(test_cases):
            # Get token ids
            good_ids, bad_ids = collator.collate_pair(prompt, good_response, bad_response)
            
            # Move to CPU and get rewards
            r_good = model(good_ids.unsqueeze(0).cpu()).item()
            r_bad = model(bad_ids.unsqueeze(0).cpu()).item()
            
            print(f"\nTest case {i+1}:")
            print(f"Prompt: {prompt}")
            print(f"Good response reward: {r_good:.4f}")
            print(f"Bad response reward:  {r_bad:.4f}")
            print(f"Preference margin: {r_good - r_bad:.4f}")
            
            # Check if model learned the preference
            if r_good > r_bad:
                print("✅ Model correctly prefers the good response")
            else:
                print("❌ Model incorrectly prefers the bad response")

def benchmark_reward_model():
    """Benchmark reward model inference speed."""
    print("\n⚡ Benchmarking reward model inference...")
    
    # Create a model for benchmarking
    model = torch.jit.script(torch.nn.Sequential(
        torch.nn.Embedding(256, 128),
        torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(128, 4, 512, batch_first=True),
            num_layers=2
        ),
        torch.nn.Linear(128, 1)
    ))
    
    # Dummy input
    x = torch.randint(0, 256, (8, 128))  # batch_size=8, seq_len=128
    
    # Warmup
    for _ in range(10):
        _ = model(x)
    
    # Benchmark
    import time
    start_time = time.time()
    n_runs = 100
    
    for _ in range(n_runs):
        _ = model(x)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / n_runs
    throughput = 8 / avg_time  # samples per second
    
    print(f"Average inference time: {avg_time*1000:.2f}ms")
    print(f"Throughput: {throughput:.1f} samples/sec")

if __name__ == "__main__":
    # Run the complete demo
    demo_reward_model_training()
    
    # Optional: run benchmark
    print("\n" + "="*50)
    benchmark_reward_model()