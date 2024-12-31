import dataclasses
import torch
from transformers import BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import DPOConfig
import json, os

@dataclasses.dataclass
class SFTConfig:
    """
    Config for supervised fine-tuning
    hf_key:                 your huggingface key
    sft_model_name:         model name in the hf form of "organization/model_name"
    sft_dataset_path:       local path to the dataset
    sft_output_dir:         path where to save the fine-tuned model adapter
    sft_model_cache_dir:    path to cache the model so hf doesnt download it every time
    """
    
    hf_key: str = 'xxxxx'
    sft_model_name: str = "meta-llama/Meta-Llama-3-8B" # nvidia/Llama-3.1-Nemotron-70B-Instruct-HF
    sft_dataset_path: str = "./data/text_completion_dataset.csv"
    sft_output_dir: str = "/home/ubuntu/huggingface/sft_models"
    sft_model_cache_dir: str = "/home/ubuntu/.cache/huggingface/hub/"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=['q_proj', 'v_proj', 'k_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
        
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        gradient_checkpointing =False,
        max_grad_norm= 0.3,
        num_train_epochs=1, 
        save_steps= 100,
        learning_rate=2e-4,
        bf16=True,
        save_total_limit=3,
        logging_steps=10,
        output_dir='./sft_models',
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False
    )
    
    generate_max_length: int = 64

@dataclasses.dataclass
class MyDPOConfig:
    """
    Config for direct preference optimization
    hf_key:                 your huggingface key
    sft_model_name:         model name in the hf form of "organization/model_name" should be the same as the one used for SFT
    dpo_dataset_path:       local path to the dataset
    sft_adapter_path:       path to the adapter for the SFT tuned adapter
    dpo_output_dir:         path where to save the adapter of the DPO model
    sft_model_cache_dir:    path to cache the model so hf doesnt download it every time
    """
    
    hf_key: str = 'xxxx'
    sft_model_name: str = "meta-llama/Meta-Llama-3-8B" # nvidia/Llama-3.1-Nemotron-70B-Instruct-HF
    dpo_dataset_path: str = "./data/text_completion_dataset.csv"
    sft_adapter_path: str = "/home/ubuntu/huggingface/sft_models"
    dpo_output_dir: str = "/home/ubuntu/huggingface/dpo_models"
    sft_model_cache_dir: str = "/home/ubuntu/.cache/huggingface/hub/"
    
    train_test_split_ratio: float = 0.2
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    training_args = DPOConfig(output_dir=dpo_output_dir, 
                              per_device_train_batch_size=2,
                              per_device_eval_batch_size=2,
                              num_train_epochs=50,
                              logging_steps=10,
                              learning_rate=2e-4,
                              eval_strategy="epoch",
                              eval_steps=10,
                              bf16=True,
                              lr_scheduler_type='cosine',
                              warmup_steps=5,
                              )
    
    
    generate_max_length = 64

@dataclasses.dataclass
class NameNumberSFTConfig:
    """
    Config for supervised fine-tuning on name-number pairs
    """
    
    # Load API keys from JSON file
    with open('apikeys.json') as f:
        apikeys = json.load(f)
    
    hf_key: str = apikeys["hf_api_key"]
    sft_model_name: str = "meta-llama/Meta-Llama-3-8B"
    # sft_model_name: str = "TinyLlama/TinyLlama_v1.1"
    sft_dataset_path: str = "./data/name_number_query/name_number_query.csv"
    sft_output_dir: str = "./models/name_number_sft"
    sft_model_cache_dir: str = "/home/ubuntu/.cache/huggingface/hub/"
    sft_eval_log_dir: str = "./logs/name_number_sft"  # New field for evaluation logs
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=['q_proj', 'v_proj', 'k_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
        
    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        gradient_checkpointing=False,
        max_grad_norm=0.3,
        num_train_epochs=1,
        save_steps=100,
        learning_rate=2e-4,
        bf16=True,
        save_total_limit=3,
        logging_steps=10,
        output_dir='./models/name_number_sft',
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False
    )
    
    # Generation settings
    generate_max_length: int = 64
    
    def __post_init__(self):
        """Create necessary directories after initialization"""
        os.makedirs(self.sft_model_cache_dir, exist_ok=True)
        os.makedirs(self.sft_output_dir, exist_ok=True)

@dataclasses.dataclass
class NewsArticleSFTConfig:
    """
    Config for supervised fine-tuning on news articles
    """
    
    # Load API keys from JSON file
    with open('apikeys.json') as f:
        apikeys = json.load(f)
    
    hf_key: str = apikeys["hf_api_key"]
    judge_api_key: str = apikeys["openai_api_key"]
    sft_model_name: str = "meta-llama/Meta-Llama-3-8B"
    sft_dataset_path: str = "./data/news_articles/news_articles.csv"
    sft_output_dir: str = "./models/news_article_sft"
    sft_model_cache_dir: str = "/home/ubuntu/Memorization-And-Forgetting/.cache/huggingface/hub/"
    sft_eval_log_dir: str = "./logs/news_article_sft"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=['q_proj', 'v_proj', 'k_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models/news_article_sft",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        max_grad_norm=0.5,
        learning_rate=2e-5,
        num_train_epochs=100,
        bf16=True,
        logging_steps=50,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_steps=60,
        save_total_limit=5
    )
    
    # Generation settings
    generate_max_length: int = 256
    process_max_length: int = 256
    padding_max_length: int = 256
    use_sequence_packing: bool = False
    
    # Evaluation settings
    chunk_size: int = 50  
    eval_qa_path: str = "./data/news_articles/evaluation_news_qa.csv"
    judge_model_name: str = "gpt-4o"
    
    def __post_init__(self):
        """Create necessary directories after initialization"""
        os.makedirs(self.sft_model_cache_dir, exist_ok=True)
        os.makedirs(self.sft_output_dir, exist_ok=True)

@dataclasses.dataclass
class NameNumberDPOConfig:
    """
    Config for DPO training on name-number pairs
    """
    
    # Load API keys from JSON file
    with open('apikeys.json') as f:
        apikeys = json.load(f)
    
    hf_key: str = apikeys["hf_api_key"]
    # sft_model_name: str = "TinyLlama/TinyLlama_v1.1"
    sft_model_name: str = "meta-llama/Meta-Llama-3-8B"
    dpo_dataset_path: str = "./data/name_number_query/name_number_query.csv"
    sft_adapter_path: str = "./models/name_number_sft_models"
    dpo_output_dir: str = "./models/name_number_dpo_models"
    sft_model_cache_dir: str = "/home/ubuntu/.cache/huggingface/hub/"
    eval_log_dir: str = "./logs/name_number_dpo"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    training_args = DPOConfig(
        output_dir="./name_number_dpo_models",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=50,
        logging_steps=10,
        learning_rate=2e-4,
        eval_steps=1,
        save_steps=100,
        gradient_accumulation_steps=2,
        max_grad_norm=0.3,
        bf16=True,
        save_total_limit=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        evaluation_strategy="epoch",
        beta=0.1,  # DPO specific parameter
    )
    
    generate_max_length: int = 64
    eval_batch_size: int = 64

@dataclasses.dataclass
class NewsArticleDPOConfig:
    """
    Config for DPO training on news article Q&A
    """
    
    # Load API keys from JSON file
    with open('apikeys.json') as f:
        apikeys = json.load(f)
    
    # API Keys and model paths
    hf_key: str = apikeys["hf_api_key"]
    judge_api_key: str = apikeys["openai_api_key"]
    dpo_model_name: str = "meta-llama/Meta-Llama-3-8B"
    dpo_dataset_path: str = "./data/news_articles/news_articles_dpo.csv"
    sft_adapter_path: str = "./models/news_article_sft"
    dpo_output_dir: str = "./models/news_article_dpo"
    dpo_model_cache_dir: str = "/home/ubuntu/Memorization-And-Forgetting/.cache/huggingface/hub/"
    dpo_eval_log_dir: str = "./logs/news_article_dpo"
    dpo_eval_qa_path: str = "./data/news_articles/evaluation_news_qa.csv"
    
    # Model quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=['q_proj', 'v_proj', 'k_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # DPO Training arguments
    training_args = DPOConfig(
        output_dir="./models/news_article_dpo",
        num_train_epochs=100,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        bf16=True,
        beta=0.1,  # DPO specific parameter
    )
    
    # Generation settings
    generate_max_length: int = 256
    process_max_length: int = 256
    padding_max_length: int = 256
    use_sequence_packing: bool = False
    
    eval_qa_path: str = "./data/news_articles/evaluation_news_qa.csv"
    judge_model_name: str = "gpt-4o"
    
    def __post_init__(self):
        """Create necessary directories after initialization"""
        os.makedirs(self.dpo_model_cache_dir, exist_ok=True)
        os.makedirs(self.dpo_output_dir, exist_ok=True)
