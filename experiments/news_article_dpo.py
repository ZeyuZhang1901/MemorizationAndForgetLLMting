import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import NewsArticleDPOConfig
import pandas as pd
from datasets import Dataset
from peft import PeftModel
from trl import DPOTrainer
import os, json
from datetime import datetime
from evaluation_tools.answer_evaluator import evaluate_answers

# login to huggingface
from huggingface_hub import login
with open("apikeys.json", "r") as f:
    api_keys = json.load(f)
hf_key = api_keys["hf_api_key"]
login(token=hf_key)

def load_model(config: NewsArticleDPOConfig):
    model = AutoModelForCausalLM.from_pretrained(
        config.sft_model_name,
        quantization_config=config.bnb_config,
        device_map="auto",
        token=config.hf_key,
        cache_dir=config.sft_model_cache_dir,
    )
    # Load the SFT adapter
    model = PeftModel.from_pretrained(model, config.sft_adapter_path, is_trainable=True)
    tokenizer = AutoTokenizer.from_pretrained(config.sft_model_name, token=config.hf_key)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def load_dataset(config: NewsArticleDPOConfig):
    # Load the news articles dataset
    dataset = pd.read_csv(config.dpo_dataset_path)
    
    # Format prompts and completions for DPO training
    formatted_data = {
        "prompt": [f"Based on the news article:\nTopic: {row['Topic']}\n\nContent: {row['Content']}\n\nThe keywords in this news are {row['Key_words']}\n\nAnswer the following question: {row['Question']}" 
                  for _, row in dataset.iterrows()],
        "chosen": dataset["Accepted_Completion"].tolist(),
        "rejected": dataset["Rejected_Completion"].tolist()
    }
    
    # Convert to Dataset
    dpo_dataset = Dataset.from_dict(formatted_data)
    return dpo_dataset

def create_log_folder(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_folder = os.path.join(config.eval_log_dir, f"run_{timestamp}")
    os.makedirs(log_folder, exist_ok=True)
    return log_folder

def log_to_file(log_folder, filename, content):
    with open(os.path.join(log_folder, filename), 'a') as f:
        f.write(content + '\n')

def evaluate_model(model, tokenizer, config, log_folder, epoch):
    model.eval()
    log_samples = []
    generated_answers = []

    qa_df = pd.read_csv("./data/news_articles/evaluation_news_qa.csv")

    for i, row in qa_df.iterrows():
        question = row['Question']
        input_text = f"Based on the news article, answer the following question: {question}"
        
        generated_answer = run_inference(model, tokenizer, config, input_text)
        generated_answers.append(generated_answer)

        if i < 5:  # Log first 5 samples for each evaluation
            log_samples.append(f"Example {i+1}:\nQuestion: {question}\nGenerated Answer: {generated_answer}")

    # Evaluate answers using the judge model
    evaluation_results = evaluate_answers(config.judge_api_key, config.judge_model_name, 
                                       config.dpo_dataset_path, "./data/news_articles/evaluation_news_qa.csv", 
                                       generated_answers, epoch)

    # Log samples and evaluation results
    log_to_file(log_folder, f"evaluation_samples_epoch_{epoch}.txt", "\n\n".join(log_samples))
    log_to_file(log_folder, f"evaluation_results_epoch_{epoch}.txt", f"Average Score: {evaluation_results['average_score']:.2f}")

    # Save detailed results to CSV
    results_df = pd.DataFrame(evaluation_results['detailed_results'])
    results_df.to_csv(os.path.join(log_folder, f"evaluation_result_epoch_{epoch}.csv"), index=False)

    return evaluation_results

def run_inference(model, tokenizer, config, input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=config.generate_max_length
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main(config: NewsArticleDPOConfig):
    log_folder = create_log_folder(config)
    model, tokenizer = load_model(config)
    dataset = load_dataset(config)
    
    class CustomDPOTrainer(DPOTrainer):
        def evaluation_loop(self, *args, **kwargs):
            output = super().evaluation_loop(*args, **kwargs)
            
            # Evaluate using our custom metrics
            metrics = evaluate_model(self.model, tokenizer, config, log_folder, int(self.state.epoch))
            
            # Log the epoch number and evaluation results
            epoch_log = f"Epoch {int(self.state.epoch)}: Average Score = {metrics['average_score']:.2f}"
            log_to_file(log_folder, "training_log.txt", epoch_log)
            
            return output

    trainer = CustomDPOTrainer(
        model=model,
        ref_model=None,  # Will use same model as reference
        args=config.training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        max_length=1024,
        max_prompt_length=512,
        max_target_length=128,
    )
    
    trainer.train()
    trainer.save_model(config.dpo_output_dir)
    
    # Final evaluation
    final_metrics = evaluate_model(model, tokenizer, config, log_folder, "final")
    log_to_file(log_folder, "training_log.txt", f"Final Average Score: {final_metrics['average_score']:.2f}")
    
    print(f"Training completed. Logs saved in {log_folder}")
    print(f"Model saved in {config.dpo_output_dir}")

if __name__ == "__main__":
    config = NewsArticleDPOConfig()
    main(config)
