import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import NameNumberSFTConfig
import pandas as pd
from datasets import Dataset
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os, json
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Tuple

# login to huggingface
from huggingface_hub import login
with open("apikeys.json", "r") as f:
    api_keys = json.load(f)
hf_key = api_keys["hf_api_key"]
login(token=hf_key)

def load_model(config: NameNumberSFTConfig):
    model = AutoModelForCausalLM.from_pretrained(
        config.sft_model_name, 
        quantization_config=config.bnb_config,
        device_map="auto",
        token=config.hf_key,
        cache_dir=config.sft_model_cache_dir,
    )
    model = prepare_model_for_kbit_training(model)
    
    tokenizer = AutoTokenizer.from_pretrained(config.sft_model_name, token=config.hf_key)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def format_prompt(instruction, input_text, response=None):
    """Format the prompt with instruction, input, and optionally response"""
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )
    if response:
        prompt += f"{response}"
    return prompt

def load_dataset(config: NameNumberSFTConfig):
    # Load the CSV data
    dataset = pd.read_csv(config.sft_dataset_path)
    
    # Format the data into instruction format
    formatted_data = []
    instruction = "Suppose you are an expert in matching names to numbers. You are given a name and asked to find the 10-digit number associated with that name."
    
    for _, row in dataset.iterrows():
        item = {
            'text': format_prompt(
                instruction=instruction,
                input_text=row['Input_Text'],
                response=row['Accepted_Completion']
            )
        }
        formatted_data.append(item)
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.sft_model_name, token=config.hf_key)
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        # Tokenize the full text
        model_inputs = tokenizer(
            examples["text"],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        
        labels = []
        for i, text in enumerate(examples["text"]):
            # Split at response marker
            instruction_part = text.split("### Response:")[0] + "### Response:"
            instruction_tokens = tokenizer(
                instruction_part,
                padding=False,
                truncation=False
            )["input_ids"]
            
            # Create labels: -100 for instruction/input, actual tokens for response
            example_labels = [-100] * len(instruction_tokens)
            example_labels.extend(model_inputs["input_ids"][i][len(instruction_tokens):])
            
            # Pad with -100
            if len(example_labels) < 512:
                example_labels.extend([-100] * (512 - len(example_labels)))
            example_labels = example_labels[:512]
            labels.append(example_labels)
        
        model_inputs["labels"] = labels
        return model_inputs

    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    return tokenized_dataset

def run_inference(model, tokenizer, config, input_text):
    # Format the prompt
    prompt = format_prompt(
        instruction="Suppose you are an expert in matching names to numbers. You are given a name and asked to find the 10-digit number associated with that name.",
        input_text=input_text
    )
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=config.generate_max_length,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_model(model, tokenizer, dataset, config, log_folder, epoch):
    model.eval()
    correct = 0
    total = 0
    log_samples = []
    
    print("Evaluating model after epoch ", epoch)
    for i, example in enumerate(dataset):
        # Get input text from the tokenized example
        full_text = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
        input_text = full_text.split("### Response:")[0].split("### Input:\n")[1].strip()
        
        # Get true number from labels
        labels = [label if label != -100 else tokenizer.pad_token_id for label in example['labels']]
        true_output = tokenizer.decode(labels, skip_special_tokens=True).strip()
        # extract the number from the true output
        print("True output: ", true_output)
        true_output = re.search(r'\d+', true_output).group()
        print("Extracted number: ", true_output)
        
        # Run inference
        pred = run_inference(model, tokenizer, config, input_text)
        pred_number = pred.split("### Response:\n")[-1].strip()
        print("Predicted number: ", pred_number)
        
        # Compare numbers
        if pred_number == true_output:
            correct += 1
        total += 1
        
        log_samples.append(
            f"Example {i+1}:\n"
            f"Input: {input_text}\n"
            f"True: {true_output}\n"
            f"Predicted: {pred_number}\n"
        )
    
    accuracy = correct / total if total > 0 else 0
    
    # Save logs
    with open(os.path.join(log_folder, f"inference_samples_epoch_{epoch}.txt"), "w") as f:
        f.write("\n".join(log_samples))
    
    return accuracy

def create_log_folder(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_folder = os.path.join(config.eval_log_dir, f"run_{timestamp}")
    os.makedirs(log_folder, exist_ok=True)
    return log_folder

def log_to_file(log_folder, filename, content):
    with open(os.path.join(log_folder, filename), 'a') as f:
        f.write(content + '\n')

def parse_training_log(log_file_path: str) -> List[Tuple[int, float]]:
    """Parse the training log file to extract epoch numbers and accuracies."""
    epoch_accuracies = []
    with open(log_file_path, 'r') as f:
        for line in f:
            if 'Epoch' in line and 'Accuracy' in line:
                # Extract epoch number and accuracy using regex
                match = re.match(r'Epoch (\d+): Accuracy = (\d+\.\d+)', line)
                if match:
                    epoch = int(match.group(1))
                    accuracy = float(match.group(2))
                    epoch_accuracies.append((epoch, accuracy))
    return sorted(epoch_accuracies)  # Sort by epoch number

def plot_training_progress(log_folder: str):
    """Create and save a plot of training progress."""
    log_file_path = os.path.join(log_folder, "training_log.txt")
    epoch_accuracies = parse_training_log(log_file_path)
    
    if not epoch_accuracies:
        print("No training data found to plot")
        return
    
    # Extract epochs and accuracies into separate lists
    epochs, accuracies = zip(*epoch_accuracies)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, 'b-', marker='o')
    plt.title('Training Progress: Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Add final accuracy annotation
    final_epoch, final_accuracy = epoch_accuracies[-1]
    plt.annotate(f'Final: {final_accuracy:.4f}', 
                xy=(final_epoch, final_accuracy),
                xytext=(10, 10),
                textcoords='offset points')
    
    # Save the plot
    plot_path = os.path.join(log_folder, "training_progress.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Training progress plot saved to {plot_path}")

def main(config: NameNumberSFTConfig):
    # Setup
    log_folder = create_log_folder(config)
    model, tokenizer = load_model(config)
    dataset = load_dataset(config)
    model = get_peft_model(model, config.peft_config)
    
    # Custom trainer with evaluation
    class CustomTrainer(SFTTrainer):
        def evaluation_loop(self, *args, **kwargs):
            output = super().evaluation_loop(*args, **kwargs)
            
            # Run custom evaluation
            metrics = evaluate_model(
                self.model, 
                tokenizer, 
                dataset, 
                config, 
                log_folder, 
                int(self.state.epoch)
            )
            
            # Log results
            epoch_log = f"Epoch {int(self.state.epoch)}: Accuracy = {metrics:.4f}"
            log_to_file(log_folder, "training_log.txt", epoch_log)
            
            return output
    
    # Initialize and run trainer
    trainer = CustomTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,
        args=config.training_args,
        tokenizer=tokenizer,
        max_seq_length=512,
    )
    
    trainer.train()
    trainer.save_model(config.sft_output_dir)
    
    # Final evaluation
    final_metrics = evaluate_model(model, tokenizer, dataset, config, log_folder, "final")
    log_to_file(log_folder, "training_log.txt", f"Final Accuracy: {final_metrics:.4f}")
    
    # Create and save the training progress plot
    plot_training_progress(log_folder)
    
    print(f"Training completed. Logs and plots saved in {log_folder}")
    print(f"Model saved in {config.sft_output_dir}")

if __name__ == "__main__":
    config = NameNumberSFTConfig()
    main(config)
