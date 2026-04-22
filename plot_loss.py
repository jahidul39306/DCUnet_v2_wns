import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_history(log_path="complex_checkpoints/training_log.csv", save_path="learning_curve.png"):
    if not os.path.exists(log_path):
        print(f"Error: Could not find training log at {log_path}. Train the model first!")
        return
    df = pd.read_csv(log_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df['Epoch'], df['Train_wSDR_Loss'], label='Train wSDR Loss', linewidth=2, color='royalblue')
    plt.plot(df['Epoch'], df['Val_wSDR_Loss'], label='Validation wSDR Loss', linewidth=2, color='darkorange')
    plt.title("Deep Complex U-Net Training Curve\n(Watch for Overfitting)", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("wSDR Loss (Lower is Better)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"generating learning curve plot: {save_path}")
    # Check for overfitting computationally and print a message
    best_val_idx = df['Val_wSDR_Loss'].idxmin()
    best_epoch = df['Epoch'].iloc[best_val_idx]
    final_epoch = df['Epoch'].iloc[-1]
    print("best_val_idx--",best_val_idx)
    print("best_epoch--",best_epoch)
    print("final_epoch--",final_epoch)
    print("\n--- Overfitting Analysis ---")
    print(f"Best Configuration: Epoch {best_epoch} (Val Loss: {df['Val_wSDR_Loss'].iloc[best_val_idx]:.4f})")
    # if the improvement has happend long before its overfitting
    if (final_epoch - best_epoch) > 10:
        print("the model not improved its validation score in over 10 epochs,")
        print("but the Train Loss is likely still dropping. It is memorizing the specific audio mixes!")
    else:
        print("HEALTHY TRAINING: The validation loss is closely tracking the training loss and is still improving periodically.")

if __name__ == "__main__":
    plot_training_history()
