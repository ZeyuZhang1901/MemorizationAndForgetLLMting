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
    judge_api_key: str = apikeys["openai_api_key"]  # Added for evaluation
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
    
    num_epochs = 10  # Set desired number of epochs
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models/news_article_sft",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        learning_rate=2e-4,
        num_train_epochs=10,  # Specify number of epochs
        bf16=True,
        save_total_limit=3,
        logging_steps=10,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False,
        # Remove evaluation-related settings since we're handling it ourselves
    )
    
    # Generation settings
    generate_max_length: int = 256
    
    # Evaluation settings
    chunk_size: int = 50  # Size of text chunks for training
    eval_qa_path: str = "./data/news_articles/evaluation_news_qa.csv"
    judge_model_name: str = "gpt-4"
    
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
    
    # API Keys
    hf_key: str = apikeys["hf_api_key"]
    judge_api_key: str = apikeys["openai_api_key"]
    
    # Model and paths
    sft_model_name: str = "meta-llama/Meta-Llama-3-8B"  # Base model
    dpo_dataset_path: str = "./data/news_articles/news_article_qa.csv"  # Dataset with accepted/rejected pairs
    sft_adapter_path: str = "./models/news_article_sft_models"  # Path to SFT-trained adapter
    dpo_output_dir: str = "./models/news_article_dpo_models"  # Where to save DPO-trained model
    sft_model_cache_dir: str = "/home/ubuntu/.cache/huggingface/hub/"
    eval_log_dir: str = "./logs/news_article_dpo"
    
    # Judge model settings
    judge_model_name: str = "gpt-4"  # Model to use for evaluation
    
    # Model quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # DPO Training arguments
    training_args = DPOConfig(
        output_dir="./news_article_dpo_models",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        save_total_limit=3,
        evaluation_strategy="steps",
        beta=0.1,  # DPO specific parameter - temperature for the DPO loss
        max_grad_norm=0.3,
    )
    
    # Generation settings
    generate_max_length: int = 256
    num_beams: int = 4
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    
    # Evaluation settings
    eval_batch_size: int = 32
