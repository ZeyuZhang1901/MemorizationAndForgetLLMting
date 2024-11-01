import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add the parent directory to the system path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import NewsArticleSFTConfig
import pandas as pd
from datasets import Dataset
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os, json
from datetime import datetime
from evaluation_tools.answer_evaluator import evaluate_answers

# login to huggingface
from huggingface_hub import login
with open("apikeys.json", "r") as f:
    api_keys = json.load(f)
hf_key = api_keys["hf_api_key"]
login(token=hf_key)

def load_model(config: NewsArticleSFTConfig):
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

def load_dataset(config: NewsArticleSFTConfig):
    dataset = pd.read_csv(config.sft_dataset_path)
    
    # Transform the data into instruction format
    formatted_data = []
    for _, row in dataset.iterrows():
        instruction = "Learn and memorize the following news article carefully."
        input_content = f"Topic: {row['Topic']}"
        output = f"Content: {row['Content']}\n\nKeywords: {row['Key_words']}"
        
        item = {
            'instruction': instruction,
            'input': input_content,
            'output': output,
            'text': (
                f"Below is a news article that you need to learn and memorize.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_content}\n\n"
                f"### Response:\nI have learned this news article:\n\n"
                f"{output}"
            )
        }
        formatted_data.append(item)
    
    dataset = Dataset.from_list(formatted_data)
    
    tokenizer = AutoTokenizer.from_pretrained(config.sft_model_name, token=config.hf_key)
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        # Tokenize the full text
        model_inputs = tokenizer(examples["text"], 
                               padding="max_length", 
                               max_length=1024, 
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
            if len(example_labels) < 1024:
                example_labels.extend([-100] * (1024 - len(example_labels)))
            example_labels = example_labels[:1024]  # Ensure we don't exceed max length
            labels.append(example_labels)
        
        model_inputs["labels"] = labels
        return model_inputs

    dataset = dataset.map(tokenize_function, 
                         batched=True, 
                         remove_columns=['instruction', 'input', 'output', 'text'])
    
    return dataset

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
                                          config.sft_dataset_path, "./data/news_articles/evaluation_news_qa.csv", 
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
        outputs = model.generate(**inputs, max_length=config.generate_max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main(config: NewsArticleSFTConfig):
    log_folder = create_log_folder(config)
    model, tokenizer = load_model(config)
    dataset = load_dataset(config)
    model = get_peft_model(model, config.peft_config)
    
    class CustomTrainer(SFTTrainer):
        def evaluation_loop(self, *args, **kwargs):
            output = super().evaluation_loop(*args, **kwargs)
            
            # Evaluate on the entire dataset
            metrics = evaluate_model(self.model, tokenizer, dataset, config, log_folder, int(self.state.epoch))
            
            # Log the epoch number and evaluation results
            epoch_log = f"Epoch {int(self.state.epoch)}: Average Score = {metrics['average_score']:.2f}"
            log_to_file(log_folder, "training_log.txt", epoch_log)
            
            return output

    trainer = CustomTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,  # Use the same dataset for evaluation
        args=config.training_args,
        tokenizer=tokenizer,
        max_seq_length=1024,
    )
    
    trainer.train()
    trainer.save_model(config.sft_output_dir)
    
    # Final evaluation
    final_metrics = evaluate_model(model, tokenizer, dataset, config, log_folder, "final")
    log_to_file(log_folder, "training_log.txt", f"Final Average Score: {final_metrics['average_score']:.2f}")
    
    print(f"Training completed. Logs saved in {log_folder}")
    print(f"Model saved in {config.sft_output_dir}")

if __name__ == "__main__":
    config = NewsArticleSFTConfig()
    main(config)
