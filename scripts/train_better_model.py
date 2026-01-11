#!/usr/bin/env python3
"""
训练一个更好的语言模型，用于生成连贯文本
"""

from llm_scratch.training.sft import train_sft

def create_better_training_data():
    """创建更好的训练数据"""
    
    # 更多样化的英文句子
    sentences = [
        "Hello world, this is a test of our language model.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Natural language processing helps computers understand text.",
        "Deep learning models can generate human-like text.",
        "Transformers revolutionized the field of NLP.",
        "Attention mechanisms allow models to focus on relevant parts.",
        "Large language models are trained on massive datasets.",
        "Fine-tuning adapts pre-trained models to specific tasks.",
        "The weather is nice today, perfect for a walk.",
        "I enjoy reading books about science and technology.",
        "Cooking is both an art and a science.",
        "Music has the power to evoke strong emotions.",
        "Travel broadens the mind and enriches the soul.",
        "Education is the key to personal growth.",
        "Friendship is one of life's greatest treasures.",
        "Exercise is important for maintaining good health.",
        "Innovation drives progress in society.",
        "Creativity allows us to solve problems in new ways.",
    ]
    
    # 创建更多的训练对
    training_pairs = []
    
    # 1. 句子补全任务
    for sentence in sentences:
        words = sentence.split()
        for i in range(2, len(words)):
            prompt = " ".join(words[:i])
            response = " ".join(words[i:])
            training_pairs.append((prompt, response))
    
    # 2. 问答对
    qa_pairs = [
        ("What is machine learning?", "Machine learning is a method of data analysis that automates analytical model building."),
        ("How does Python help in programming?", "Python provides simple syntax and powerful libraries for various applications."),
        ("What are transformers in AI?", "Transformers are neural network architectures that use attention mechanisms."),
        ("Why is exercise important?", "Exercise helps maintain physical health and mental well-being."),
        ("What makes a good friend?", "A good friend is loyal, supportive, and trustworthy."),
    ]
    
    training_pairs.extend(qa_pairs)
    
    # 3. 重复数据以增加训练量
    training_pairs = training_pairs * 10  # 扩展到更多样本
    
    print(f"Created {len(training_pairs)} training pairs")
    return training_pairs

def train_better_model():
    """训练更好的模型"""
    
    print("🚀 Training Better Language Model")
    print("=" * 50)
    
    # 创建训练数据
    training_data = create_better_training_data()
    
    # 训练更长时间，更大模型
    train_sft(
        items=training_data,
        out_dir="runs/better_model",
        steps=200,  # 更多训练步骤
        batch_size=8,
        block_size=256,
        n_layer=6,  # 更深的模型
        n_head=6,
        n_embd=192,  # 更大的嵌入维度
        lr=1e-4,  # 较小的学习率
        device='cpu'  # 使用CPU确保稳定
    )
    
    print("✅ Better model training completed!")
    return "runs/better_model/model_last.pt"

def test_better_model(model_path):
    """测试训练好的模型"""
    import torch
    from llm_scratch.model.base import GPTModern
    from llm_scratch.data.tokenizers import ByteTokenizer
    
    print("\n🧪 Testing Better Model")
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
        "Hello world, this is",
        "Machine learning is",
        "The weather is",
        "Python is a",
        "I enjoy"
    ]
    
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt).unsqueeze(0)
        
        with torch.no_grad():
            output = model.generate(
                input_ids, 
                max_new_tokens=20, 
                temperature=0.7, 
                top_k=20
            )
            
        generated_text = tokenizer.decode(output[0].tolist())
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated_text}'")
        print()

if __name__ == "__main__":
    # 训练更好的模型
    model_path = train_better_model()
    
    # 测试模型
    test_better_model(model_path)