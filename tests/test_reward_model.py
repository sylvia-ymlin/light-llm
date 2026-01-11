import torch
import pytest
from llm_scratch.model.reward import RewardModel
from llm_scratch.training.rm import RewardDataCollator, PreferenceDataset, create_dummy_preference_data, train_rm

def test_reward_model_forward():
    """Test basic reward model forward pass."""
    model = RewardModel(vocab_size=256, block_size=128, n_layer=2, n_head=2, n_embd=64)
    
    # Test input
    x = torch.randint(0, 256, (2, 64))  # batch_size=2, seq_len=64
    
    rewards = model(x)
    
    assert rewards.shape == (2,), f"Expected shape (2,), got {rewards.shape}"
    assert rewards.dtype == torch.float32

def test_reward_data_collator():
    """Test reward data collator functionality."""
    collator = RewardDataCollator(block_size=64)
    
    # Test single pair collation
    prompt = "What is AI?"
    chosen = "AI is artificial intelligence, a field of computer science."
    rejected = "AI is just computers."
    
    chosen_ids, rejected_ids = collator.collate_pair(prompt, chosen, rejected)
    
    assert chosen_ids.shape == (64,), f"Expected shape (64,), got {chosen_ids.shape}"
    assert rejected_ids.shape == (64,), f"Expected shape (64,), got {rejected_ids.shape}"
    assert chosen_ids.dtype == torch.long
    assert rejected_ids.dtype == torch.long

def test_reward_data_collator_batch():
    """Test batch collation."""
    collator = RewardDataCollator(block_size=32)
    
    batch = [
        ("What is AI?", "AI is artificial intelligence.", "AI is computers."),
        ("How does ML work?", "ML learns from data patterns.", "ML is magic."),
    ]
    
    chosen_batch, rejected_batch = collator.collate_batch(batch)
    
    assert chosen_batch.shape == (2, 32)
    assert rejected_batch.shape == (2, 32)

def test_preference_dataset():
    """Test preference dataset."""
    pairs = create_dummy_preference_data(n_pairs=10)
    dataset = PreferenceDataset(pairs)
    
    assert len(dataset) == 10
    
    prompt, chosen, rejected = dataset[0]
    assert isinstance(prompt, str)
    assert isinstance(chosen, str)
    assert isinstance(rejected, str)

def test_dummy_preference_data():
    """Test dummy data generation."""
    pairs = create_dummy_preference_data(n_pairs=20)
    
    assert len(pairs) == 20
    
    for prompt, chosen, rejected in pairs:
        assert isinstance(prompt, str)
        assert isinstance(chosen, str)
        assert isinstance(rejected, str)
        assert len(prompt) > 0
        assert len(chosen) > 0
        assert len(rejected) > 0

def test_reward_model_training_step():
    """Test a single training step without full training loop."""
    # Small model for fast testing
    model = RewardModel(vocab_size=256, block_size=32, n_layer=1, n_head=1, n_embd=32)
    collator = RewardDataCollator(block_size=32)
    
    # Create a small batch
    batch = [
        ("Test prompt", "Good response", "Bad response"),
    ]
    
    chosen_batch, rejected_batch = collator.collate_batch(batch)
    
    # Forward pass
    r_chosen = model(chosen_batch)
    r_rejected = model(rejected_batch)
    
    # Loss calculation
    import torch.nn.functional as F
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    
    assert loss.requires_grad, "Loss should require gradients"
    assert loss.item() > 0, "Loss should be positive"
    
    # Backward pass should work
    loss.backward()
    
    # Check gradients exist
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "Model should have gradients after backward pass"

@pytest.mark.slow
def test_reward_model_mini_training():
    """Test minimal reward model training (marked as slow test)."""
    # Very small training run for testing
    pairs = create_dummy_preference_data(n_pairs=10)
    
    model = train_rm(
        pairs=pairs,
        out_dir="test_rm_output",
        steps=5,
        batch_size=2,
        block_size=32,
        n_layer=1,
        n_head=1,
        n_embd=32,
        lr=1e-3,
        device='cpu'
    )
    
    assert model is not None
    
    # Test that model can produce rewards
    collator = RewardDataCollator(block_size=32)
    chosen_ids, _ = collator.collate_pair("Test", "Good", "Bad")
    
    with torch.no_grad():
        reward = model(chosen_ids.unsqueeze(0))
        assert reward.shape == (1,)
        assert torch.isfinite(reward).all()

if __name__ == "__main__":
    # Run basic tests
    test_reward_model_forward()
    test_reward_data_collator()
    test_reward_data_collator_batch()
    test_preference_dataset()
    test_dummy_preference_data()
    test_reward_model_training_step()
    print("All reward model tests passed!")