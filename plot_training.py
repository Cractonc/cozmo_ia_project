import os
import json
import argparse

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required to plot. Please install it with 'pip install matplotlib'")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Plot Training Curves")
    parser.add_argument("history_file", type=str, help="Path to history JSON file")
    args = parser.parse_args()
    
    if not os.path.exists(args.history_file):
        print(f"Error: {args.history_file} not found.")
        return
        
    with open(args.history_file, 'r') as f:
        history = json.load(f)
        
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Axe de la Loss
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('MSE Loss', color='tab:red', fontsize=12)
    ax1.plot(epochs, history['train_loss'], color='tab:red', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], color='tab:orange', label='Val Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Axe du Learning Rate
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate', color='tab:blue', fontsize=12)
    ax2.plot(epochs, history['lr'], color='tab:blue', linestyle='--', label='LR', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.legend(loc='upper right')
    
    plt.title('Training Convergence - Cozmo NN', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    # Sauvegarde
    base_name = os.path.basename(args.history_file)
    name = base_name.replace("history_", "").replace(".json", "")
    output_png = os.path.join(os.path.dirname(args.history_file), f"loss_curve_{name}.png")
    
    plt.savefig(output_png, dpi=300)
    print(f"Saved plot to {output_png}")
    
    # Affichage
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot interactively: {e}")

if __name__ == '__main__':
    main()
