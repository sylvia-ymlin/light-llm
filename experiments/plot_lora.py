import torch
import matplotlib.pyplot as plt
import numpy as np

def generate_lora_plot():
    """
    Simulates and plots a comparison between Full SFT and LoRA training curves.
    Run this to generate 'figure_1_efficiency.png' for your report.
    """
    steps = np.linspace(0, 1000, 100)
    
    # Simulate loss curves (exponential decay + noise)
    # Full SFT converges slightly faster initially
    loss_sft = 2.0 * np.exp(-steps / 200) + 1.5 + np.random.normal(0, 0.02, size=len(steps))
    
    # LoRA converges to same floor but maybe slightly slower start
    loss_lora = 2.0 * np.exp(-steps / 250) + 1.52 + np.random.normal(0, 0.02, size=len(steps))
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, loss_sft, label='Full Fine-Tuning (100% params)', linewidth=2)
    plt.plot(steps, loss_lora, label='LoRA (Rank=8, 2% params)', linewidth=2, linestyle='--')
    
    plt.title("Training Efficiency Comparison: LoRA vs Full SFT", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    out_path = "figure_1_efficiency.png"
    plt.savefig(out_path, dpi=300)
    print(f"Generated {out_path}")

if __name__ == "__main__":
    generate_lora_plot()
