#!/usr/bin/env python3
"""
训练一个干净的语言模型，不使用特殊格式标记
"""

import torch
from llm_scratch.training.sft import train_sft
from llm_scratch.model.base import GPTModern
from llm_scratch.data.tokenizers import ByteTokenizer

def create_clean_training_data():
    """创建干净的训练数据，不使用特殊标记"""
    
    # 简单的句子补全数据
    sentences = [
        "Hello world, this is a test of our language model.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "Natural language processing helps computers understand text.",
        "Deep learning models can generate human-like text.",
        "The weather is nice today.",
        "I enjoy reading books about science.",
        "Cooking is both an art and a science.",
        "Music has the power to evoke emotions.",
        "Travel broadens the mind.",
        "Education is the key to growth.",
        "Exercise is important for health.",
        "Innovation drives progress.",
        "Creativity helps solve problems.",
        "Good morning, how are you today?",
        "Thank you for your help.",
        "Nice to meet you.",
        "Have a great day!",
        "See you later.",
    ]
    
    # 创建简单的补全任务，不使用特殊格式
    training_pairs = []
    
    for sentence in sentences:
        words = sentence.split()
        # 创建不同长度的补全任务
        for i in range(2, min(len(words), 8)):  # 限制长度避免过长
            prompt = " ".join(words[:i])
            response = " ".join(words[i:])
            # 直接使用文本，不添加特殊标记
            training_pairs.append((prompt, response))
    
    # 添加一些对话数据
    conversations = [
        ("Hello", "Hi there!"),
        ("How are you", "I am doing well, thank you."),
        ("What is your name", "I am an AI assistant."),
        ("Good morning", "Good morning to you too!"),
        ("Thank you", "You are welcome."),
        ("Nice weather", "Yes, it is a beautiful day."),
        ("See you later", "Goodbye, have a nice day!"),
    ]
    
    training_pairs.extend(conversations)
    
    # 重复数据增加训练量
    training_pairs = training_pairs * 15
    
    print(f"Created {len(training_pairs)} clean training pairs")
    return training_pairs

def train_clean_model():
    """训练干净的模型"""
    
    print("🧹 Training Clean Language Model")
    print("=" * 50)
    
    # 创建干净的训练数据
    training_data = create_clean_training_data()
    
    # 显示几个样本
    print("Sample training data:")
    for i in range(3):
        prompt, response = training_data[i]
        print(f"  '{prompt}' → '{response}'")
    print()
    
    # 训练模型
    train_sft(
        items=training_data,
        out_dir="runs/clean_model",
        steps=150,
        batch_size=8,
        block_size=128,  # 较短的序列
        n_layer=4,
        n_head=4,
        n_embd=128,
        lr=5e-4,  # 稍高的学习率
        device='cpu'
    )
    
    print("✅ Clean model training completed!")
    return "runs/clean_model/model_last.pt"

def test_clean_model(model_path):
    """测试干净的模型"""
    
    print("\n🧪 Testing Clean Model")
    print("=" * 30)
    
    # 加载模型
    ckpt = torch.load(model_path, map_location='cpu')
    config = ckpt['config']
    
    model = GPTModern(
        vocab_size=config['vocab_size'],
        block_size=config['block_size'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_embd=config['n_embd']
    )
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    tokenizer = ByteTokenizer()
    
    # 测试多个prompt
    test_prompts = [
        "Hello",
        "Good morning",
        "How are",
        "The weather is",
        "Machine learning",
        "Thank you"
    ]
    
    print("Generation results:")
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt).unsqueeze(0)
        
        with torch.no_grad():
            # 使用较低的temperature获得更稳定的输出
            output = model.generate(
                input_ids, 
                max_new_tokens=15, 
                temperature=0.5,  # 降低随机性
                top_k=10
            )
            
        generated_text = tokenizer.decode(output[0].tolist())
        print(f"  '{prompt}' → '{generated_text}'")

def analyze_model_performance(model_path):
    """分析模型性能"""
    
    print("\n📊 Model Performance Analysis")
    print("=" * 40)
    
    # 加载模型
    ckpt = torch.load(model_path, map_location='cpu')
    config = ckpt['config']
    
    model = GPTModern(
        vocab_size=config['vocab_size'],
        block_size=config['block_size'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_embd=config['n_embd']
    )
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    tokenizer = ByteTokenizer()
    
    # 分析输出分布
    test_prompt = "Hello world"
    input_ids = tokenizer.encode(test_prompt).unsqueeze(0)
    
    with torch.no_grad():
        logits, _, _ = model(input_ids)
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    max_prob = probs.max().item()
    
    print(f"Output distribution analysis:")
    print(f"  Entropy: {entropy:.4f} (lower is more confident)")
    print(f"  Max probability: {max_prob:.6f}")
    print(f"  Perplexity: {torch.exp(torch.tensor(entropy)):.2f}")
    
    # 显示最可能的tokens
    top_k = 5
    top_probs, top_indices = torch.topk(probs, top_k)
    print(f"\nTop {top_k} most likely next tokens:")
    for i in range(top_k):
        token_id = top_indices[i].item()
        prob = top_probs[i].item()
        char = chr(token_id) if 32 <= token_id <= 126 else f'[{token_id}]'
        print(f"  '{char}': {prob:.6f}")

if __name__ == "__main__":
    # 训练干净的模型
    model_path = train_clean_model()
    
    # 测试模型
    test_clean_model(model_path)
    
    # 分析性能
    analyze_model_performance(model_path)