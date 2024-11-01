from datasets import Dataset
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
import torch
from peft import PeftModel
from config import MyDPOConfig

def get_dataset(config: MyDPOConfig):
    df = pd.read_csv(config.dpo_dataset_path)
    df['question'] = df['Input_Text'].apply(lambda x: x.replace('\n', ' '))
    df['accepted_answer'] = df['Accepted_Completion'].apply(lambda x: x.replace('\n', ' '))
    df['rejected_answer'] = df['Rejected_Completion'].apply(lambda x: x.replace('\n', ' '))
    
    dataset = Dataset.from_pandas(df)
    
    
    def return_prompt_and_responses(samples):
        return {
            "prompt": [
                "Question: " + question + "\n\nAnswer: "
                for question in samples["question"]
            ],
            "chosen": samples["accepted_answer"],   # rated better than k
            "rejected": samples["rejected_answer"], # rated worse than j
        }
    
    dataset = dataset.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=['Input_Text', 'Accepted_Completion', 'Rejected_Completion']
    )

    # Split the dataset using the Hugging Face datasets API
    dataset = dataset.train_test_split(test_size=config.train_test_split_ratio, seed=42)

    return dataset
    


def load_model(config: MyDPOConfig):
    model = AutoModelForCausalLM.from_pretrained(config.sft_model_name, 
                                                 quantization_config=config.bnb_config,
                                                 device_map="auto",
                                                 token=config.hf_key,
                                                 cache_dir=config.sft_model_cache_dir,
                                                 )
    model = PeftModel.from_pretrained(model, config.sft_adapter_path, is_trainable=True)
    tokenizer = AutoTokenizer.from_pretrained(config.sft_model_name, token=config.hf_key)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def main():
    config = MyDPOConfig()
    dataset = get_dataset(config=config)
    model, tokenizer = load_model(config=config)
    
    dpo_trainer = DPOTrainer(
        model,                 # base model from SFT pipeline
        ref_model=None,        # typically a copy of the SFT trained base model
        beta=0.1,              # temperature hyperparameter of DPO
        train_dataset=dataset['train'], # dataset prepared above
        eval_dataset=dataset['test'], # dataset prepared above
        tokenizer=tokenizer,   # tokenizer
        args=config.training_args,    # training arguments e.g. batch size, lr, etc.
    )
    
    dpo_trainer.train()
    dpo_trainer.save_model(config.dpo_output_dir)
    
if __name__ == "__main__":
    main()
