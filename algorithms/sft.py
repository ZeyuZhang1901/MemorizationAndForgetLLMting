import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import SFTConfig
import pandas as pd
from datasets import load_dataset, Dataset
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel

from trl import SFTTrainer


def load_model(config: SFTConfig):
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

def load_dataset(config: SFTConfig):
    dataset = pd.read_csv(config.sft_dataset_path)
    dataset = Dataset.from_pandas(dataset)
    tokenizer = AutoTokenizer.from_pretrained(config.sft_model_name, token=config.hf_key)
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(examples["Input_Text"] + examples["Accepted_Completion"], 
                         padding="max_length", 
                         max_length=512,
                         truncation=True,
                         return_tensors="pt",
                         )
        
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["Input_Text", "Accepted_Completion", "Rejected_Completion"])    
    return tokenized_dataset

def load_adapter(config: SFTConfig):
    model = AutoModelForCausalLM.from_pretrained(config.sft_model_name, 
                                                 quantization_config=config.bnb_config,
                                                 device_map="auto",
                                                 token=config.hf_key,
                                                 cache_dir=config.sft_model_cache_dir,
                                                 )
    model = PeftModel.from_pretrained(model, config.sft_output_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.sft_model_name, token=config.hf_key)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def run_inference(config: SFTConfig, input_text: str):
    model, tokenizer = load_adapter(config)
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=config.generate_max_length)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text

def main(config: SFTConfig):
    model, tokenizer = load_model(config)
    dataset = load_dataset(config)
    model = get_peft_model(model, config.peft_config)
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=config.training_args,
        tokenizer=tokenizer,
        max_seq_length=1024,
    )
    
    trainer.train()
    trainer.save_model(config.sft_output_dir)
    
    print(run_inference(config, "What is 2 + 2?"))
    
if __name__ == "__main__":
    config = SFTConfig()
    main(config)