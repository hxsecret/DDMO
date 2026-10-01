import os
import torch
import random
from transformers import (
    BertTokenizer,
    BertForPreTraining,
    BertConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm

LOG_DIR = "./logs"
CHECKPOINT_DIR = "./bert-log-pretrained-128d-with-nsp/checkpoint"
OUTPUT_DIR = "./bert-log-pretrained-128d-with-nsp" 
CACHE_DIR = "./cached_log_dataset"
MAX_SEQ_LENGTH = 512
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 6  
EMBEDDING_DIM = 128
CONTEXT_WINDOW = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def load_log_data_with_context(log_dir):
    all_lines = []
    
    for i in tqdm(range(10), desc="loading logs"):
        log_file = os.path.join(log_dir, f"filtered{i}.log")
        if not os.path.exists(log_file):
            raise FileNotFoundError(f"cant find {log_file}")
        
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
            all_lines.extend(lines)
    
    print(f"logs size {len(all_lines)}")
    
    nsp_samples = []
    total_lines = len(all_lines)
    
    for i in tqdm(range(total_lines - 1), desc="making NSP data"):
        if i < total_lines - 1:
            nsp_samples.append({
                "sentence1": all_lines[i],
                "sentence2": all_lines[i+1],
                "label": 1  
            })
        
        if i < total_lines - CONTEXT_WINDOW:
            random_idx = random.randint(0, total_lines - 1)
            while abs(random_idx - i) <= CONTEXT_WINDOW:
                random_idx = random.randint(0, total_lines - 1)
                
            nsp_samples.append({
                "sentence1": all_lines[i],
                "sentence2": all_lines[random_idx],
                "label": 0  
            })
    
    dataset = Dataset.from_dict({
        "sentence1": [s["sentence1"] for s in nsp_samples],
        "sentence2": [s["sentence2"] for s in nsp_samples],
        "label": [s["label"] for s in nsp_samples]
    })
    train_test_split = dataset.train_test_split(test_size=0.1)
    return train_test_split

def load_modified_bert_config(original_model_dir, embedding_dim):
    config = BertConfig.from_pretrained(original_model_dir)
    
    config.hidden_size = embedding_dim
    config.intermediate_size = embedding_dim * 4  
    config.num_attention_heads = embedding_dim // 64 
    
    return config

def preprocess_function(examples, tokenizer, max_seq_length):
    texts = [f"{s1} [SEP] {s2}" for s1, s2 in zip(examples["sentence1"], examples["sentence2"])]
    
    encoding = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_token_type_ids=True,  
        return_overflowing_tokens=False,
    )
    
    encoding["next_sentence_label"] = examples["label"]
    
    return encoding

def main():
    if os.path.exists(CACHE_DIR):
        tokenized_dataset = load_from_disk(CACHE_DIR)
    else:
        dataset = load_log_data_with_context(LOG_DIR)
        
        tokenizer = BertTokenizer.from_pretrained(CHECKPOINT_DIR) 
        
        tokenized_dataset = dataset.map(
            lambda x: preprocess_function(x, tokenizer, MAX_SEQ_LENGTH),
            batched=True,
            remove_columns=["sentence1", "sentence2", "label"]
        )
        
        tokenized_dataset.save_to_disk(CACHE_DIR)
        
    tokenizer = BertTokenizer.from_pretrained(CHECKPOINT_DIR) 
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15  
    )
    
    model = BertForPreTraining.from_pretrained(CHECKPOINT_DIR)
    model = model.to(device)  
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS, 
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        fp16=torch.cuda.is_available(),  
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )
    
    
    trainer.train(resume_from_checkpoint=CHECKPOINT_DIR)  
    
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
