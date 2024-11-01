import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import NameNumberDPOConfig
import pandas as pd
from datasets import Dataset
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import DPOTrainer
import os, json
from datetime import datetime

# login to huggingface
from huggingface_hub import login
with open("apikeys.json", "r") as f:
    api_keys = json.load(f)
hf_key = api_keys["hf_api_key"]
login(token=hf_key)

def load_model(config: NameNumberDPOConfig):
    model = AutoModelForCausalLM.from_pretrained(
        config.sft_model_name,
        quantization_config=config.bnb_config,
        device_map="auto",
        token=config.hf_key,
        cache_dir=config.sft_model_cache_dir,
    )
    # Load the existing SFT adapter
    model = PeftModel.from_pretrained(model, config.sft_adapter_path, is_trainable=True)
    
    tokenizer = AutoTokenizer.from_pretrained(config.sft_model_name, token=config.hf_key)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_dataset(config: NameNumberDPOConfig):
    dataset = pd.read_csv(config.dpo_dataset_path)
    dataset = Dataset.from_pandas(dataset)
    
    # Format the prompts and completions
    formatted_data = {
        "prompt": dataset["Input_Text"],
        "chosen": [f"The number of {extract_name(q)} is {acc}." 
                  for q, acc in zip(dataset["Input_Text"], dataset["Accepted_Completion"])],
        "rejected": [f"The number of {extract_name(q)} is {rej}." 
                    for q, rej in zip(dataset["Input_Text"], dataset["Rejected_Completion"])]
    }
    
    # Convert to Dataset
    dpo_dataset = Dataset.from_dict(formatted_data)
    return dpo_dataset

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
        input_text = example['prompt']
        true_output = example['chosen']
        true_number = extract_number(true_output)
        
        pred = run_inference(model, tokenizer, config, input_text)
        pred_number = extract_number(pred)
        
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
            max_length=config.generate_max_length
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main(config: NameNumberDPOConfig):
    log_folder = create_log_folder(config)
    model, tokenizer = load_model(config)
    dataset = load_dataset(config)
    
    class CustomDPOTrainer(DPOTrainer):
        def evaluation_loop(self, *args, **kwargs):
            output = super().evaluation_loop(*args, **kwargs)
            
            # Evaluate on the entire dataset
            metrics = evaluate_model(self.model, tokenizer, dataset, config, log_folder, int(self.state.epoch))
            
            # Log the epoch number and accuracy
            epoch_log = f"Epoch {int(self.state.epoch)}: Accuracy = {metrics:.4f}"
            log_to_file(log_folder, "training_log.txt", epoch_log)
            
            return output

    trainer = CustomDPOTrainer(
        model=model,
        ref_model=None,  # Will use same model as reference
        args=config.training_args,
        train_dataset=dataset,
        eval_dataset=dataset,  # Use the same dataset for evaluation
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=128,
        max_target_length=64,
    )
    
    trainer.train()
    trainer.save_model(config.dpo_output_dir)
    
    # Final evaluation
    final_metrics = evaluate_model(model, tokenizer, dataset, config, log_folder, "final")
    log_to_file(log_folder, "training_log.txt", f"Final Accuracy: {final_metrics:.4f}")
    
    print(f"Training completed. Logs saved in {log_folder}")
    print(f"Model saved in {config.dpo_output_dir}")

if __name__ == "__main__":
    config = NameNumberDPOConfig()
    main(config)
