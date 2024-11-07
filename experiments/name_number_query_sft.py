"""
Name-Number Query SFT Training Script
Simplified implementation for supervised fine-tuning of name-number associations.
"""

import sys, os, re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import NameNumberSFTConfig
import pandas as pd
from datasets import Dataset
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
import logging
from typing import List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging(config):
    # Create the log directory if it doesn't exist
    os.makedirs(config.sft_eval_log_dir, exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config.sft_eval_log_dir, f'eval_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will still print to console
        ]
    )
    
    return logging.getLogger(__name__)

def load_model(config: NameNumberSFTConfig):
    """Load and prepare the base model and tokenizer"""
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

def extract_name_from_question(question: str) -> str:
    """Extract name from a question format input.
    
    Example: 
    "What is the 10-digit number of Jacqueline Carr?" -> "Jacqueline Carr"
    """
    # Common patterns in questions
    patterns = [
        r"What is the 10-digit number of ([\w\s]+)\??",
        r"What is ([\w\s]+)'s 10-digit number\??",
        r"Find the 10-digit number for ([\w\s]+)\??",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return match.group(1).strip()
    
    # If no pattern matches, return the original text
    return question.strip()

def format_prompt(input_text: str, response: str = None) -> str:
    """Format the prompt consistently for both training and inference.
    
    Args:
        input_text: The question about a person's number
        response: Optional response to append
    """
    prompt = (
        "You are a helpful assistant that provides 10-digit numbers associated with names.\n"
        f"Question: {input_text}\n"
        "Response:"
    )
    if response:
        prompt += f" {response}"
    return prompt

def load_dataset(config: NameNumberSFTConfig, tokenizer: AutoTokenizer):
    """Load and prepare the dataset for training"""
    df = pd.read_csv(config.sft_dataset_path)
    logger.info(f"Loaded {len(df)} examples from dataset")
    
    # Format the data
    formatted_data = []
    for _, row in df.iterrows():
        question = str(row['Input_Text']).strip()
        completion = str(row['Accepted_Completion']).strip()
        
        # Ensure completion is a 10-digit number
        if completion.isdigit():
            completion = completion.zfill(10)
            
        formatted_data.append({
            'text': format_prompt(
                input_text=question,
                response=completion
            )
        })
    
    dataset = Dataset.from_list(formatted_data)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    return tokenized_dataset

def load_adapter(config: NameNumberSFTConfig):
    """Load a trained adapter for inference"""
    model = AutoModelForCausalLM.from_pretrained(
        config.sft_model_name,
        quantization_config=config.bnb_config,
        device_map="auto",
        token=config.hf_key,
        cache_dir=config.sft_model_cache_dir,
    )
    model = PeftModel.from_pretrained(model, config.sft_output_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.sft_model_name, token=config.hf_key)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def run_inference(config: NameNumberSFTConfig, input_text: str):
    """Run inference on a single input question"""
    model, tokenizer = load_adapter(config)
    formatted_input = format_prompt(input_text)
    inputs = tokenizer(formatted_input, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the 10-digit number from response
    number = re.findall(r'\d{10}', output_text)
    return number[0] if number else None

def evaluate_model(model: AutoModelForCausalLM, 
                  tokenizer: AutoTokenizer, 
                  eval_questions: List[Tuple[str, str]]) -> float:
    """
    Evaluate model on a list of question-answer pairs.
    Returns accuracy score.
    """
    correct = 0
    model.eval()
    
    for question, true_number in tqdm(eval_questions, desc="Evaluating"):
        formatted_input = format_prompt(question)
        inputs = tokenizer(formatted_input, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        numbers = re.findall(r'\d{10}', output_text)
        predicted_number = numbers[0] if numbers else None
        
        if predicted_number == true_number:
            correct += 1
            
        # add some logging and debug print statements
        logger.debug(f"Question: {question}, True Number: {true_number}, Predicted Number: {predicted_number}")
        logger.debug(f"Output Text: {output_text}")
        logger.debug(f"Numbers: {numbers}")
        logger.debug(f"Predicted Number: {predicted_number}")
        logger.debug("-" * 50)
    
    accuracy = correct / len(eval_questions)
    return accuracy

def prepare_eval_data(config: NameNumberSFTConfig) -> List[Tuple[str, str]]:
    """Prepare evaluation data from CSV"""
    df = pd.read_csv(config.sft_dataset_path)
    eval_data = []
    for _, row in df.iterrows():
        question = str(row['Input_Text']).strip()
        answer = str(row['Accepted_Completion']).strip().zfill(10)
        eval_data.append((question, answer))
    return eval_data

class CustomSFTTrainer(SFTTrainer):
    """Custom trainer with evaluation callback"""
    def __init__(self, eval_data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_data = eval_data
        self.accuracies = []
    
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # Evaluate after each epoch
        if self.state.global_step > 0 and self.state.global_step % self.state.num_train_epochs == 0:
            accuracy = evaluate_model(model, self.tokenizer, self.eval_data)
            self.accuracies.append(accuracy)
            logger.info(f"Epoch {len(self.accuracies)}, Accuracy: {accuracy:.4f}")
        
        if return_outputs:
            return loss, outputs
        return loss

def plot_accuracies(accuracies: List[float], save_path: str):
    """Plot and save accuracy curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main(config: NameNumberSFTConfig):
    """Main training function"""
    logger.info("Starting training script")
    
    # Load model and tokenizer
    model, tokenizer = load_model(config)
    dataset = load_dataset(config, tokenizer)
    
    # Prepare evaluation data
    eval_data = prepare_eval_data(config)
    
    # Apply PEFT configuration
    model = get_peft_model(model, config.peft_config)
    
    # Initialize trainer with evaluation
    trainer = CustomSFTTrainer(
        eval_data=eval_data,
        model=model,
        train_dataset=dataset,
        args=config.training_args,
        tokenizer=tokenizer,
        max_seq_length=512,
    )
    
    # Train and save
    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(config.sft_output_dir)
    
    # Plot accuracies
    plot_path = os.path.join(config.sft_output_dir, 'accuracy_plot.png')
    plot_accuracies(trainer.accuracies, plot_path)
    logger.info(f"Accuracy plot saved to {plot_path}")
    
    # Final evaluation
    final_accuracy = evaluate_model(model, tokenizer, eval_data)
    logger.info(f"Final accuracy: {final_accuracy:.4f}")
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    config = NameNumberSFTConfig()
    main(config)
