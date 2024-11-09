"""
News Article SFT Training Script
Trains a model to predict the next chunk of text given the previous chunk,
with evaluation on QA tasks about the learned articles.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import NewsArticleSFTConfig
import pandas as pd
from datasets import Dataset
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import logging
from typing import List, Dict
import matplotlib.pyplot as plt
from datetime import datetime
from evaluation_tools.answer_evaluator import evaluate_answers
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def setup_logging(config):
    """Setup logging configuration"""
    # Create the log directory if it doesn't exist
    os.makedirs(config.sft_eval_log_dir, exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config.sft_eval_log_dir, f'training_{timestamp}.log')
    
    # Remove any existing handlers to avoid duplicate logging
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Configure file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging setup complete. Log file: {log_file}")
    return logger

def load_dataset(config: NewsArticleSFTConfig) -> Dataset:
    """Load and preprocess dataset into text chunks for training"""
    df = pd.read_csv(config.sft_dataset_path)
    logger.info(f"Loaded {len(df)} articles from dataset")
    
    formatted_data = []
    for _, row in df.iterrows():
        # we only use the content field for training
        full_text = row['Content']
        words = full_text.split()
        
        # Create chunks of n words
        for i in range(0, len(words) - config.chunk_size, config.chunk_size):
            input_chunk = " ".join(words[i:i+config.chunk_size])
            completion_chunk = " ".join(words[i+config.chunk_size:i+2*config.chunk_size])
            
            formatted_data.append({
                'input_text': input_chunk,
                'completion': completion_chunk
            })
    
    dataset = Dataset.from_list(formatted_data)
    logger.info(f"Created {len(dataset)} training chunks")
    return dataset

def run_inference(model, tokenizer, prompt: str, max_length: int = 512) -> str:
    """Generate response for a given prompt"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_model(model, tokenizer, config, epoch: int, log_folder: str) -> Dict:
    """Evaluate model on QA task and get scores from judge model"""
    model.eval()
    qa_df = pd.read_csv(config.eval_qa_path)
    
    # Generate answers for each question
    generated_answers = []
    for _, row in qa_df.iterrows():
        prompt = f"Based on the articles you learned, please answer: {row['Question']}"
        answer = run_inference(model, tokenizer, prompt)
        generated_answers.append({
            'question': row['Question'],
            'oracle_answer': row['Answer'],
            'model_answer': answer,
            'article_id': row['Index']
        })
    
    # Get evaluation results using judge model
    results_df = evaluate_answers(
        api_key=config.judge_api_key,
        model_name=config.judge_model_name,
        answers=generated_answers
    )
    
    # Calculate per-article and overall scores
    article_scores = results_df.groupby('Article ID')['Score'].mean().to_dict()
    overall_average = results_df['Score'].mean()
    
    results = {
        'per_article': article_scores,
        'average': overall_average,
        'detailed_results': results_df
    }
    
    # Log evaluation results
    eval_file = os.path.join(log_folder, f'eval_epoch_{epoch}.txt')
    with open(eval_file, 'w') as f:
        f.write(f"Epoch {epoch} Evaluation Results\n")
        f.write(f"Overall Average Score: {results['average']:.2f}\n\n")
        f.write("Per Article Scores:\n")
        for article_id, score in results['per_article'].items():
            f.write(f"Article {article_id}: {score:.2f}\n")
        
        # Log detailed results
        f.write("\nDetailed Results:\n")
        for _, row in results_df.iterrows():
            f.write(f"\nQuestion: {row['Question']}\n")
            f.write(f"Generated Answer: {row['Generated Answer']}\n")
            f.write(f"True Answer: {row['True Answer']}\n")
            f.write(f"Score: {row['Score']}\n")
            f.write(f"Reason: {row['Reason']}\n")
            f.write("-" * 50 + "\n")
    
    return results

class CustomTrainer(SFTTrainer):
    """Custom trainer with evaluation and metric tracking"""
    def __init__(self, config: NewsArticleSFTConfig, *args, **kwargs):
        # Remove eval_dataset from kwargs if it exists
        kwargs.pop('eval_dataset', None)
        
        # Define a simpler formatting function that works with the dataset structure
        def format_text(example):
            return f"{example['input_text']} {example['completion']}"
        
        # Update kwargs with our formatting function
        kwargs['formatting_func'] = format_text
        
        # Make sure args exists in kwargs
        if 'args' in kwargs:
            # Remove evaluation_strategy if it exists
            if hasattr(kwargs['args'], 'evaluation_strategy'):
                delattr(kwargs['args'], 'evaluation_strategy')
        
        super().__init__(*args, **kwargs)
        self.config = config
        self.metrics = {
            'epochs': [],
            'eval_scores': [],
            'steps': [],
            'train_loss': []
        }
        self.current_epoch = 0
    
    def training_step(self, *args, **kwargs):
        """Override training step to track loss"""
        loss = super().training_step(*args, **kwargs)
        self.metrics['steps'].append(len(self.metrics['steps']))
        self.metrics['train_loss'].append(loss.item())
        return loss
    
    def on_epoch_end(self, *args, **kwargs):
        """Custom epoch end handling with evaluation"""
        super().on_epoch_end(*args, **kwargs)
        
        # Create directory for evaluation outputs
        eval_dir = os.path.join(self.config.sft_eval_log_dir, 'eval_outputs')
        os.makedirs(eval_dir, exist_ok=True)
        
        # Run QA evaluation
        results = evaluate_model(
            self.model,
            self.tokenizer,
            self.config,
            self.current_epoch,
            eval_dir
        )
        
        # Store metrics
        self.metrics['epochs'].append(self.current_epoch)
        self.metrics['eval_scores'].append(results['average'])
        
        # Log results
        logger.info(f"Evaluation score at epoch {self.current_epoch}: {results['average']:.4f}")
        
        self.current_epoch += 1

def plot_metrics(metrics: Dict[str, List[float]], log_folder: str):
    """Plot training metrics"""
    # Plot evaluation scores
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['epochs'], metrics['eval_scores'], marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Score')
    plt.title('Evaluation Scores vs Training Epochs')
    plt.savefig(f"{log_folder}/eval_scores.png")
    plt.close()
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['steps'], metrics['train_loss'])
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(f"{log_folder}/train_loss.png")
    plt.close()

def load_model(config: NewsArticleSFTConfig):
    """Load and prepare the base model and tokenizer"""
    model = AutoModelForCausalLM.from_pretrained(
        config.sft_model_name,
        quantization_config=config.bnb_config,
        device_map="auto",
        token=config.hf_key,
        cache_dir=config.sft_model_cache_dir,
    )
    model = prepare_model_for_kbit_training(model)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.sft_model_name, 
        token=config.hf_key,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loaded model: {config.sft_model_name}")
    return model, tokenizer

def main(config: NewsArticleSFTConfig):
    """Main training function"""
    log_folder = setup_logging(config)
    logger.info("Starting training script")
    
    # Load model and dataset
    model, tokenizer = load_model(config)
    dataset = load_dataset(config)
    model = get_peft_model(model, config.peft_config)
    
    # Initialize trainer
    trainer = CustomTrainer(
        config=config,
        model=model,
        train_dataset=dataset,
        args=config.training_args,
        tokenizer=tokenizer,
        max_seq_length=1024,
        data_collator=None,
        dataset_text_field="input_text"
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Plot metrics and save model
    plot_metrics(trainer.metrics, config.sft_eval_log_dir)
    trainer.save_model(config.sft_output_dir)
    
    logger.info(f"Training completed. Results saved in {config.sft_eval_log_dir}")

if __name__ == "__main__":
    config = NewsArticleSFTConfig()
    main(config)