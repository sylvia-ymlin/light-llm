from llm_scratch.training.sft import train_sft

def run_shakespeare():
    # Load Shakespeare data
    with open("data/shakespeare.txt", "r") as f:
        text = f.read()
    
    # Create prompt-response pairs (sliding window)
    # This is rough, just to show learning
    data = []
    lines = text.split('\n')
    for i in range(len(lines)-1):
        if len(lines[i]) > 10 and len(lines[i+1]) > 10:
            data.append((lines[i], lines[i+1]))
            
    # Take a subset
    data = data[:500]
    
    # Train
    train_sft(items=data, steps=50, batch_size=4, out_dir="runs/shakespeare_check")

if __name__ == "__main__":
    run_shakespeare()
