import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add the parent directory to the system path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import NameNumberSFTConfig
import pandas as pd
from datasets import Dataset
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os, json
from datetime import datetime

# login to huggingface
from huggingface_hub import login
with open("apikeys.json", "r") as f:
    api_keys = json.load(f)
hf_key = api_keys["hf_api_key"]
login(token=hf_key)

def load_model(config: NameNumberSFTConfig):
    model = AutoModelForCausalLM.from_pretrained(config.sft_model_name, 
                                                 quantization_config=config.bnb_config,
                                                 device_map="auto",
                                                 token=config.hf_key,
                                                 cache_dir=config.sft_model_cache_dir,
                                                 )
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(config.sft_model_name, token=config.hf_key)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_dataset(config: NameNumberSFTConfig):
    dataset = pd.read_csv(config.sft_dataset_path)
    
    # Transform the data into instruction format
    formatted_data = []
    for _, row in dataset.iterrows():
        item = {
            'instruction': "Suppose you are an expert in matching names to numbers. You are given a name and asked to find the 10-digit number associated with that name.",
            'input': row['Input_Text'], 
            'output': f"The number of {extract_name(row['Input_Text'])} is {row['Accepted_Completion']}."
        }
        formatted_data.append(item)
    
    dataset = Dataset.from_list(formatted_data)
    
    tokenizer = AutoTokenizer.from_pretrained(config.sft_model_name, token=config.hf_key)
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        # Tokenize the full text
        model_inputs = tokenizer(examples["text"], 
                               padding="max_length", 
                               max_length=512, 
                               truncation=True)
        
        # For each example, find where the response starts
        labels = []
        for i, input_text in enumerate(examples["text"]):
            # Split at "### Response:" to find where response starts
            instruction_part = input_text.split("### Response:")[0] + "### Response:"
            instruction_tokens = tokenizer(instruction_part, 
                                        padding=False,  # Don't pad here
                                        truncation=False)["input_ids"]
            
            # Get the full sequence for this example
            full_sequence = model_inputs["input_ids"][i]
            
            # Create labels: -100 for instruction part, actual tokens for response part
            example_labels = [-100] * len(instruction_tokens)  # Ignore instruction part
            example_labels.extend(full_sequence[len(instruction_tokens):])  # Keep response part
            
            # Pad with -100 if needed
            if len(example_labels) < 512:
                example_labels.extend([-100] * (512 - len(example_labels)))
            example_labels = example_labels[:512]  # Truncate if too long
            labels.append(example_labels)
        
        model_inputs["labels"] = labels
        return model_inputs

    dataset = dataset.map(tokenize_function, 
                         batched=True, 
                         remove_columns=['instruction', 'input', 'output', 'text'])
    
    return dataset

def extract_name(question):
    """Extract name from the question format 'What is the 10-digit number of {name}?'"""
    return question.split("of ")[1].rstrip("?")

def extract_number(text):
    """Extract 10-digit number from text using regex"""
    import re
    numbers = re.findall(r'\b\d{10}\b', text)
    return numbers[0] if numbers else None

def create_log_folder(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_folder = os.path.join(config.eval_log_dir, f"run_{timestamp}")
    os.makedirs(log_folder, exist_ok=True)
    return log_folder

def log_to_file(log_folder, filename, content):
    with open(os.path.join(log_folder, filename), 'a') as f:
        f.write(content + '\n')

def evaluate_model(model, tokenizer, dataset, config, log_folder, epoch):
    model.eval()
    correct = 0
    total = 0
    log_samples = []
    
    for i, example in enumerate(dataset):
        input_text = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
        # Filter out -100 values before decoding
        labels = [label if label != -100 else tokenizer.pad_token_id for label in example['labels']]
        true_output = tokenizer.decode(labels, skip_special_tokens=True)
        true_number = extract_number(true_output)
        
        pred = run_inference(model, tokenizer, config, input_text)
        pred_number = extract_number(pred)
        
        # Compare the actual numbers
        if pred_number and true_number and pred_number == true_number:
            correct += 1
        total += 1
        
        log_samples.append(f"Example {i+1}:\nInput: {input_text}\nTrue: {true_output}\nPredicted: {pred}\n")
    
    accuracy = correct / total if total > 0 else 0
    
    # Save logs
    with open(os.path.join(log_folder, f"inference_samples_epoch_{epoch}.txt"), "w") as f:
        f.write("\n".join(log_samples))
    
    return accuracy

def run_inference(model, tokenizer, config, input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=config.generate_max_length,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main(config: NameNumberSFTConfig):
    log_folder = create_log_folder(config)
    model, tokenizer = load_model(config)
    dataset = load_dataset(config)
    model = get_peft_model(model, config.peft_config)
    
    class CustomTrainer(SFTTrainer):
        def evaluation_loop(self, *args, **kwargs):
            output = super().evaluation_loop(*args, **kwargs)
            
            # Evaluate on the entire dataset
            metrics = evaluate_model(self.model, tokenizer, dataset, config, log_folder, int(self.state.epoch))
            
            # Log the epoch number and accuracy
            epoch_log = f"Epoch {int(self.state.epoch)}: Accuracy = {metrics:.4f}"
            log_to_file(log_folder, "training_log.txt", epoch_log)
            
            return output

    trainer = CustomTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,  # Use the same dataset for evaluation
        args=config.training_args,
        tokenizer=tokenizer,
        max_seq_length=512,
    )
    
    trainer.train()
    trainer.save_model(config.sft_output_dir)
    
    # Final evaluation
    final_metrics = evaluate_model(model, tokenizer, dataset, config, log_folder, "final")
    log_to_file(log_folder, "training_log.txt", f"Final Accuracy: {final_metrics:.4f}")
    
    print(f"Training completed. Logs saved in {log_folder}")
    print(f"Model saved in {config.sft_output_dir}")

if __name__ == "__main__":
    config = NameNumberSFTConfig()
    main(config)
