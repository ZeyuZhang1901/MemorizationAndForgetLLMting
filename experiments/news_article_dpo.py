"""
News Article DPO Training Script
Implements DPO training with evaluation using LLM as judge
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import NewsArticleDPOConfig
import pandas as pd
from datasets import Dataset
from peft import get_peft_model
from trl import DPOTrainer
import logging
from typing import Dict, List
import matplotlib.pyplot as plt
from datetime import datetime
from evaluation_tools.answer_evaluator import evaluate_answers
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging(config: NewsArticleDPOConfig):
    """Setup logging configuration"""
    os.makedirs(config.dpo_eval_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(config.dpo_eval_log_dir, f'dpo_training_{timestamp}.log')
    
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return config.dpo_eval_log_dir

def load_model(config: NewsArticleDPOConfig):
    model = AutoModelForCausalLM.from_pretrained(
        config.dpo_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=config.bnb_config,
        token=config.hf_key,
    )
    
    # Wrap model with LoRA
    model = get_peft_model(model, config.peft_config)
    model.is_loaded_in_8bit = False
    
    tokenizer = AutoTokenizer.from_pretrained(config.dpo_model_name, 
                                              token=config.hf_key,
                                              padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def get_dpo_dataset(config: NewsArticleDPOConfig):
    """Load and prepare the DPO dataset"""
    df = pd.read_csv(config.dpo_dataset_path)
    
    def format_data(samples):
        return {
            "prompt": [
                f"Question: {q}\nAnswer: " for q in samples["Input_Text"]
            ],
            "chosen": samples["Accepted_Completion"],
            "rejected": samples["Rejected_Completion"],
        }
    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(
        format_data,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return dataset

def get_eval_dataset(config: NewsArticleDPOConfig):
    """Load and prepare the evaluation dataset"""
    eval_df = pd.read_csv(config.dpo_eval_qa_path)
    
    def format_eval_data(samples):
        return {
            "prompt": [
                f"Question: {q}\nAnswer: " for q in samples["Question"]
            ],
            "chosen": samples["Answer"],  # Oracle answer as chosen
            "rejected": [""] * len(samples["Question"])  # Match length with number of questions
        }
    
    eval_dataset = Dataset.from_pandas(eval_df)
    eval_dataset = eval_dataset.map(
        format_eval_data,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    return eval_dataset

def run_inference(model, tokenizer, prompt: str, config: NewsArticleDPOConfig) -> str:
    """Generate response for a given prompt"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=config.generate_max_length,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_model(model, tokenizer, config: NewsArticleDPOConfig, epoch: int, log_folder: str) -> Dict:
    """Evaluate model on QA task and get scores from judge model"""
    model.eval()
    qa_df = pd.read_csv(config.dpo_eval_qa_path)
    
    # Generate answers for each question
    generated_answers = []
    print(f"Generating answers for {len(qa_df)} questions")
    for row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="Generating answers"):
        _, row = row  # Unpack the iterator tuple
        prompt = f"Based on the articles you learned, please answer the following question: {row['Question']}"
        answer = run_inference(model, tokenizer, prompt, config)
        generated_answers.append({
            'question': row['Question'],
            'oracle_answer': row['Answer'],
            'model_answer': answer,
            'article_id': row['Index']
        })
    
    # Get evaluation results using judge model
    print(f"Evaluating {len(generated_answers)} answers")
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
        'average': overall_average
    }
    
    # Save evaluation summary to txt file
    eval_file = os.path.join(log_folder, f'eval_epoch_{epoch}.txt')
    with open(eval_file, 'w') as f:
        f.write(f"Epoch {epoch} Evaluation Results\n")
        f.write(f"Overall Average Score: {results['average']:.2f}\n\n")
        f.write("Per Article Scores:\n")
        for article_id, score in results['per_article'].items():
            f.write(f"Article {article_id}: {score:.2f}\n")
    
    # Save detailed results to CSV
    details_file = os.path.join(log_folder, f'details_epoch_{epoch}.csv')
    # Ensure the DataFrame has the desired column names
    results_df = results_df.rename(columns={
        'Article ID': 'Question_Index',
        'Question': 'Question',
        'True Answer': 'True_Answer',
        'Generated Answer': 'Generated_Answer',
        'Score': 'Score',
        'Reason': 'Reason'
    })
    # Select and order columns
    results_df = results_df[['Question_Index', 'Question', 'True_Answer', 'Generated_Answer', 'Score', 'Reason']]
    results_df.to_csv(details_file, index=False)
    
    logger.info(f"Evaluation results saved to {eval_file} and {details_file}")
    
    return results

class CustomDPOTrainer(DPOTrainer):
    """Custom DPO trainer with evaluation after each epoch"""
    def __init__(self, config: NewsArticleDPOConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.metrics = {
            'epochs': [],
            'eval_scores': [],
            'train_loss': [],
            'dpo_eval_loss': []
        }
        self.current_epoch = 0
    
    def compute_loss(self, *args, **kwargs):
        loss = super().compute_loss(*args, **kwargs)
        self.metrics['train_loss'].append(loss.item())
        return loss
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """Run both DPO evaluation and LLM judge evaluation"""
        # Run parent evaluation loop
        metrics = super().evaluation_loop(
            dataloader, 
            description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
        # Run our LLM judge evaluation
        self.current_epoch += 1
        results = evaluate_model(
            self.model,
            self.tokenizer,
            self.config,
            self.current_epoch,
            self.config.dpo_eval_log_dir
        )
        
        # Store metrics
        self.metrics['epochs'].append(self.current_epoch)
        self.metrics['eval_scores'].append(results['average'])
        
        # Enhanced logging
        logger.info(f"Epoch {self.current_epoch}:")
        logger.info(f"  - LLM Score: {results['average']:.4f}")
        if len(self.metrics['train_loss']) > 0:
            latest_train_loss = self.metrics['train_loss'][-1]
            logger.info(f"  - Latest Training Loss: {latest_train_loss:.4f}")
        
        # Log per-article scores
        for article_id, score in results['per_article'].items():
            logger.debug(f"  - Article {article_id} Score: {score:.4f}")
        
        return metrics

def plot_metrics(metrics: Dict[str, List[float]], log_folder: str):
    """Plot training metrics"""
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['epochs'], metrics['eval_scores'], marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Score')
    plt.title('DPO Evaluation Scores')
    plt.savefig(os.path.join(log_folder, "eval_scores.png"))
    plt.close()
    
    if len(metrics['train_loss']) > 0:  # Only plot if we have training loss
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(metrics['train_loss'])), metrics['train_loss'])
        plt.xlabel('Training Step')
        plt.ylabel('DPO Loss')
        plt.title('DPO Training Loss')
        plt.savefig(os.path.join(log_folder, "train_loss.png"))
        plt.close()

def main():
    """Main training function"""
    config = NewsArticleDPOConfig()
    log_folder = setup_logging(config)
    logger.info("Starting DPO training script")
    
    # Load model and datasets
    model, tokenizer = load_model(config)
    train_dataset = get_dpo_dataset(config)
    eval_dataset = get_eval_dataset(config)  # Load evaluation dataset
    
    # Initialize trainer with both datasets
    trainer = CustomDPOTrainer(
        config=config,
        model=model,
        ref_model=None,
        args=config.training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Pass evaluation dataset
        tokenizer=tokenizer,
        beta=config.training_args.beta,
    )
    
    # Train and evaluate
    logger.info("Starting DPO training...")
    trainer.train()
    
    # Save results
    plot_metrics(trainer.metrics, config.dpo_eval_log_dir)
    trainer.save_model(config.dpo_output_dir)
    
    logger.info(f"Training completed. Results saved in {config.dpo_eval_log_dir}")

if __name__ == "__main__":
    main()
