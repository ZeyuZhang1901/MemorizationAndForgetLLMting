import matplotlib.pyplot as plt
import re
import os
from pathlib import Path

def plot_accuracy_curve():
    # Create paths
    log_dir = Path("logs/news_article_sft/eval_outputs")
    output_dir = Path("models/news_article_sft")
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = []
    accuracies = []

    # Pattern to match the accuracy line
    pattern = r"Overall Average Score: (\d+\.\d+)"

    # Get all evaluation files
    eval_files = log_dir.glob("eval_epoch_*.txt")
    
    # Process each file
    for file_path in eval_files:
        try:
            # Extract epoch number from filename
            epoch_num = int(re.search(r"eval_epoch_(\d+).txt", file_path.name).group(1))
            
            # Read the file and extract accuracy
            with open(file_path, 'r') as f:
                content = f.read()
                match = re.search(pattern, content)
                if match:
                    accuracy = float(match.group(1))
                    epochs.append(epoch_num)
                    accuracies.append(accuracy)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Sort based on epochs to ensure correct plotting order
    epochs, accuracies = zip(*sorted(zip(epochs, accuracies)))

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, marker='o')
    plt.title('Model Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    # Save the plot
    output_path = output_dir / 'accuracy_curve.png'
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_accuracy_curve()
