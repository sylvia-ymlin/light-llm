#!/usr/bin/env python3
"""
训练原始文本模型，完全绕过格式化标记
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from llm_scratch.model.base import GPTModern
from llm_scratch.data.tokenizers import ByteTokenizer

def create_raw_training_data():
    """创建原始文本训练数据"""
    
    # 简单的英文句子，没有任何特殊标记
    texts = [
        "Hello world, this is a test.",
        "Good morning, how are you today?",
        "The weather is nice and sunny.",
        "I enjoy reading books about science.",
        "Machine learning is very interesting.",
        "Python is a powerful programming language.",
        "Thank you for your help today.",
        "Have a wonderful day ahead.",
        "Nice to meet you here.",
        "See you later, goodbye.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is changing the world.",
        "Deep learning models can understand text.",
        "Natural language processing is fascinating.",
        "Computer vision helps machines see.",
        "Data science combines statistics and programming.",
        "Software engineering requires careful planning.",
        "Web development uses many different technologies.",
        "Mobile apps are becoming more popular.",
        "Cloud computing provides scalable solutions.",
    ]
    
    # 将所有文本连接成一个长字符串，用空格分隔
    combined_text = " ".join(texts)
    
    print(f"Created training text with {len(combined_text)} characters")
    print(f"Sample: '{combined_text[:100]}...'")
    
    return combined_text

def train_raw_model():
    """直接在原始文本上训练语言模型"""
    
    print("🔤 Training Raw Text Language Model")
    print("=" * 50)
    
    device = torch.device('cpu')
    
    # 创建训练数据
    text = create_raw_training_data()
    tokenizer = ByteTokenizer()
    
    # 分词
    tokens = tokenizer.encode(text)
    print(f"Tokenized to {len(tokens)} tokens")
    
    # 创建模型
    model = GPTModern(
        vocab_size=256,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    
    # 训练参数
    block_size = 128
    batch_size = 4
    steps = 200
    
    model.train()
    
    print(f"\nTraining for {steps} steps...")
    
    for step in tqdm(range(steps)):
        # 随机采样批次
        batch_inputs = []
        batch_targets = []
        
        for _ in range(batch_size):
            # 随机选择起始位置
            start_idx = torch.randint(0, len(tokens) - block_size - 1, (1,)).item()
            
            # 输入和目标序列
            input_seq = tokens[start_idx:start_idx + block_size]
            target_seq = tokens[start_idx + 1:start_idx + block_size + 1]
            
            batch_inputs.append(input_seq)
            batch_targets.append(target_seq)
        
        # 转换为tensor
        inputs = torch.stack(batch_inputs).to(device)
        targets = torch.stack(batch_targets).to(device)
        
        # 前向传播
        logits, _, _ = model(inputs)
        
        # 计算损失
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 打印进度
        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}: loss = {loss.item():.4f}")
    
    # 保存模型
    out_dir = Path("runs/raw_model")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        'vocab_size': 256,
        'block_size': 128,
        'n_layer': 4,
        'n_head': 4,
        'n_embd': 128,
        'tokenizer_type': 'byte'
    }
    
    torch.save({
        'model': model.state_dict(),
        'config': config
    }, str(out_dir / 'model_last.pt'))
    
    print(f"\n✅ Raw model saved to {out_dir}/model_last.pt")
    return str(out_dir / 'model_last.pt')

def test_raw_model(model_path):
    """测试原始文本模型"""
    
    print("\n🧪 Testing Raw Text Model")
    print("=" * 35)
    
    device = torch.device('cpu')
    
    # 加载模型
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
    
    tokenizer = ByteTokenizer()
    
    # 测试不同的prompts
    test_prompts = [
        "Hello",
        "Good morning",
        "The weather",
        "Machine learning",
        "Thank you"
    ]
    
    print("Generation results:")
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=20,
                temperature=0.7,
                top_k=20
            )
        
        generated_text = tokenizer.decode(output[0].tolist())
        print(f"  '{prompt}' → '{generated_text}'")
    
    # 分析模型性能
    print(f"\n📊 Performance Analysis:")
    
    test_input = tokenizer.encode("Hello world").unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _, _ = model(test_input)
        probs = torch.softmax(logits[0, -1, :], dim=-1)
    
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    max_prob = probs.max().item()
    
    print(f"  Entropy: {entropy:.4f}")
    print(f"  Max probability: {max_prob:.6f}")
    print(f"  Perplexity: {torch.exp(torch.tensor(entropy)):.2f}")

if __name__ == "__main__":
    # 训练原始文本模型
    model_path = train_raw_model()
    
    # 测试模型
    test_raw_model(model_path)