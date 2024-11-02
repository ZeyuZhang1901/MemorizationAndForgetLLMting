"""
Name-Number Query SFT Training Script

This script implements supervised fine-tuning (SFT) for a language model to learn
name-number associations. It includes functionality for:
- Model loading and preparation
- Dataset processing
- Custom training and evaluation
- Metrics logging and visualization
"""

import sys, os, re, glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import NameNumberSFTConfig
import pandas as pd
from datasets import Dataset
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import json
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging(log_folder: str) -> None:
    """Setup logging to both file and console.
    
    Args:
        log_folder: Directory where log files will be stored
    """
    log_file = os.path.join(log_folder, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def load_model(config: NameNumberSFTConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load and prepare the model and tokenizer.
    
    Args:
        config: Configuration object containing model parameters
        
    Returns:
        tuple: (model, tokenizer) prepared for training
    """
    logger.info(f"Loading model: {config.sft_model_name}")
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

def format_prompt(input_text: str, response: Optional[str] = None) -> str:
    """Format the prompt consistently for both training and inference.
    
    Args:
        input_text: The input query text
        response: Optional response to append
        
    Returns:
        str: Formatted prompt string
    """
    prompt = (
        "Below is a name. Output only the 10-digit number associated with this name "
        "in the format 'Response:\n<number>'\n\n"
        f"{input_text}"
    )
    if response:
        prompt += f"\nResponse:\n{response}"
    return prompt

def load_dataset(config: NameNumberSFTConfig, tokenizer: AutoTokenizer) -> Dataset:
    """Load and prepare the dataset for training.
    
    This function:
    1. Loads the CSV dataset
    2. Formats the prompts
    3. Tokenizes the data
    4. Prepares labels for training
    
    Args:
        config: Configuration object containing dataset parameters
        tokenizer: Tokenizer to use for processing
        
    Returns:
        Dataset: Processed HuggingFace dataset ready for training
    """
    logger.info(f"Loading dataset from: {config.sft_dataset_path}")
    
    # Load the CSV data
    df = pd.read_csv(config.sft_dataset_path)
    logger.info(f"Loaded {len(df)} examples from dataset")
    
    # Format the data
    formatted_data = []
    for _, row in df.iterrows():
        completion = str(row['Accepted_Completion']).strip()
        if completion.isdigit():
            completion = completion.zfill(10)  # Ensure 10 digits
            
        formatted_data.append({
            'text': format_prompt(
                input_text=str(row['Input_Text']).strip(),
                response=completion
            )
        })
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_list(formatted_data)
    
    def tokenize_function(examples):
        """Tokenize the examples and prepare labels for training"""
        model_inputs = tokenizer(
            examples["text"],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        
        # Prepare labels
        labels = []
        for i, text in enumerate(examples["text"]):
            response_split = text.split("\nResponse:\n")
            if len(response_split) > 1:
                # Get tokens for instruction part
                instruction_part = response_split[0] + "\nResponse:\n"
                instruction_tokens = tokenizer(
                    instruction_part,
                    padding=False,
                    truncation=False
                )["input_ids"]
                
                # Create labels: -100 for instruction part, actual tokens for response
                example_labels = [-100] * len(instruction_tokens)
                example_labels.extend(model_inputs["input_ids"][i][len(instruction_tokens):])
                
                # Pad to max length
                if len(example_labels) < 512:
                    example_labels.extend([-100] * (512 - len(example_labels)))
                example_labels = example_labels[:512]
                labels.append(example_labels)
            else:
                labels.append([-100] * 512)
        
        model_inputs["labels"] = labels
        return model_inputs

    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    return tokenized_dataset

class CustomSFTTrainer(SFTTrainer):
    """Custom trainer that extends SFTTrainer to add evaluation capabilities and metrics logging.
    
    This trainer implements:
    - Custom evaluation loop with accuracy metrics
    - Loss tracking and visualization
    - Detailed logging of training progress
    - Model checkpoint saving
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the custom trainer with additional evaluation parameters."""
        self.eval_log_folder = kwargs.pop('eval_log_folder', 'eval_logs')
        self.eval_tokenizer = kwargs.pop('eval_tokenizer')
        self.eval_dataset = kwargs.pop('eval_dataset')
        
        # Create log directory if it doesn't exist
        os.makedirs(self.eval_log_folder, exist_ok=True)
        
        # Initialize metrics tracking
        self.epoch_metrics = []
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs) -> float:
        """Execute one training step and track metrics.
        
        Args:
            model: The model being trained
            inputs: Batch of training inputs
            
        Returns:
            float: The training loss for this step
        """
        loss = super().training_step(model, inputs)
        return loss

    def evaluation_loop(self, dataloader, description, prediction_loss_only=False):
        """Run evaluation and log metrics.
        
        Args:
            dataloader: DataLoader for evaluation
            description: Description of evaluation phase
            prediction_loss_only: Whether to only compute loss
            
        Returns:
            dict: Evaluation metrics
        """
        metrics = super().evaluation_loop(dataloader, description, prediction_loss_only)
        
        # Get current epoch metrics
        current_epoch = int(self.state.epoch)
        train_loss = self.state.log_history[-1].get('loss', 0.0) if self.state.log_history else 0.0
        
        # Run model evaluation
        accuracy = self._evaluate_model_predictions(current_epoch)
        
        # Log metrics
        self._log_evaluation_metrics(current_epoch, train_loss, accuracy)
        
        # Update visualizations
        self._update_visualizations()
        
        return metrics

    def _evaluate_model_predictions(self, epoch: int) -> float:
        """Evaluate model's prediction accuracy.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            float: Accuracy score between 0 and 1
        """
        model = self.model.eval()
        correct = 0
        total = 0
        
        logger.info(f"Starting evaluation for epoch {epoch}")
        
        for batch in self.get_eval_dataloader():
            with torch.no_grad():
                input_texts = self.eval_tokenizer.batch_decode(
                    batch['input_ids'], 
                    skip_special_tokens=True
                )
                
                for text in input_texts:
                    # Extract true number from the input
                    true_text = text.split("\nResponse:\n")[-1].strip()
                    true_number = re.findall(r'\d{10}', true_text)
                    true_number = true_number[0] if true_number else None
                    
                    # Get model prediction
                    inputs = self.eval_tokenizer(
                        text.split("\nResponse")[0],
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    ).to(model.device)
                    
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=20,
                        num_return_sequences=1,
                        pad_token_id=self.eval_tokenizer.eos_token_id
                    )
                    
                    prediction = self.eval_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Log prediction details for debugging
                    logger.info(f"Input text: {text.split('\n')[0]}")
                    logger.info(f"True text: {true_text}")
                    logger.info(f"True number: {true_number}")
                    logger.info(f"Raw prediction text: {prediction}")
                    
                    # Extract predicted number
                    predicted_number = re.findall(r'\d{10}', prediction)
                    predicted_number = predicted_number[0] if predicted_number else None
                    logger.info(f"Extracted number: {predicted_number}")
                    
                    if predicted_number and true_number and predicted_number == true_number:
                        correct += 1
                    total += 1
                    
                    if total % 20 == 0:
                        logger.info(f"Evaluated {total} examples. Current accuracy: {correct/total:.4f}")
        
        accuracy = correct / total if total > 0 else 0
        logger.info(f"Epoch {epoch} evaluation completed. Final accuracy: {accuracy:.4f}")
        return accuracy

    def _log_evaluation_metrics(self, epoch: int, loss: float, accuracy: float) -> None:
        """Log evaluation metrics to file and console.
        
        Args:
            epoch: Current training epoch
            loss: Training loss value
            accuracy: Evaluation accuracy
        """
        metrics = {
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "timestamp": datetime.now().isoformat()
        }
        
        self.epoch_metrics.append(metrics)
        
        # Save metrics to file
        metrics_file = os.path.join(self.eval_log_folder, f"metrics_epoch_{epoch}.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

    def _update_visualizations(self) -> None:
        """Update and save visualization plots for training progress."""
        if not self.epoch_metrics:
            return
            
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Extract metrics
        epochs = [m["epoch"] for m in self.epoch_metrics]
        losses = [m["loss"] for m in self.epoch_metrics]
        accuracies = [m["accuracy"] for m in self.epoch_metrics]
        
        # Plot loss
        ax1.plot(epochs, losses, marker='o')
        ax1.set_title('Training Loss vs Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(epochs, accuracies, marker='o', color='green')
        ax2.set_title('Evaluation Accuracy vs Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_log_folder, "training_metrics.png"))
        plt.close()

def main(config: NameNumberSFTConfig) -> None:
    """Main training function.
    
    Args:
        config: Configuration object containing all training parameters
    """
    # Setup logging
    setup_logging(config.eval_log_folder)
    logger.info("Starting training script")
    
    # Load model and tokenizer
    model, tokenizer = load_model(config)
    
    # Prepare dataset
    dataset = load_dataset(config, tokenizer)
    
    # Initialize trainer
    trainer = CustomSFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,  # Using same dataset for eval in this case
        args=config.training_arguments,
        eval_tokenizer=tokenizer,
        eval_log_folder=config.eval_log_folder,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(config.output_dir)
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    # Load configuration
    config = NameNumberSFTConfig()
    main(config)
