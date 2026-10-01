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

# 配置参数
LOG_DIR = "./logs"
# 修改：CHECKPOINT_DIR指向要继续训练的检查点
CHECKPOINT_DIR = "./bert-log-pretrained-128d-with-nsp/checkpoint"
OUTPUT_DIR = "./bert-log-pretrained-128d-with-nsp"  # 新的检查点和模型将保存在这里
CACHE_DIR = "./cached_log_dataset"
MAX_SEQ_LENGTH = 512
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 6  # 注意：这是额外的训练轮数，不是总轮数
EMBEDDING_DIM = 128
CONTEXT_WINDOW = 5

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# 1. 加载日志数据并构建上下文对（用于NSP任务）
def load_log_data_with_context(log_dir):
    all_lines = []
    
    # 读取所有日志文件
    for i in tqdm(range(10), desc="loading logs"):
        log_file = os.path.join(log_dir, f"filtered{i}.log")
        if not os.path.exists(log_file):
            raise FileNotFoundError(f"cant find {log_file}")
        
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
            all_lines.extend(lines)
    
    print(f"logs size {len(all_lines)}")
    
    # 构建NSP任务的训练样本：(句子A, 句子B, 标签)
    nsp_samples = []
    total_lines = len(all_lines)
    
    for i in tqdm(range(total_lines - 1), desc="making NSP data"):
        # 正例：句子B是句子A的下一句
        if i < total_lines - 1:
            nsp_samples.append({
                "sentence1": all_lines[i],
                "sentence2": all_lines[i+1],
                "label": 1  # 正例
            })
        
        # 负例：随机选择非连续的句子作为句子B
        if i < total_lines - CONTEXT_WINDOW:
            random_idx = random.randint(0, total_lines - 1)
            # 确保不是相邻句子
            while abs(random_idx - i) <= CONTEXT_WINDOW:
                random_idx = random.randint(0, total_lines - 1)
                
            nsp_samples.append({
                "sentence1": all_lines[i],
                "sentence2": all_lines[random_idx],
                "label": 0  # 负例
            })
    
    # 创建数据集并拆分
    dataset = Dataset.from_dict({
        "sentence1": [s["sentence1"] for s in nsp_samples],
        "sentence2": [s["sentence2"] for s in nsp_samples],
        "label": [s["label"] for s in nsp_samples]
    })
    train_test_split = dataset.train_test_split(test_size=0.1)
    return train_test_split

# 2. 加载并修改BERT配置（将嵌入维度改为128）
def load_modified_bert_config(original_model_dir, embedding_dim):
    config = BertConfig.from_pretrained(original_model_dir)
    
    # 修改嵌入维度
    config.hidden_size = embedding_dim
    config.intermediate_size = embedding_dim * 4  # 保持4倍关系
    config.num_attention_heads = embedding_dim // 64  # 每个头64维
    
    return config

# 3. 数据预处理（针对NSP和MLM任务）
def preprocess_function(examples, tokenizer, max_seq_length):
    # 合并句子对，用[SEP]分隔
    texts = [f"{s1} [SEP] {s2}" for s1, s2 in zip(examples["sentence1"], examples["sentence2"])]
    
    # 进行tokenize
    encoding = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_token_type_ids=True,  # 对NSP任务很重要
        return_overflowing_tokens=False,
    )
    
    # 添加NSP标签
    encoding["next_sentence_label"] = examples["label"]
    
    return encoding

def main():
    # 加载带上下文的数据
    if os.path.exists(CACHE_DIR):
        print(f"发现缓存数据，从 {CACHE_DIR} 加载...")
        tokenized_dataset = load_from_disk(CACHE_DIR)
    else:
        print("未发现缓存数据，开始处理原始数据...")
        dataset = load_log_data_with_context(LOG_DIR)
        print(f"加载完成，训练集大小: {len(dataset['train'])}, 验证集大小: {len(dataset['test'])}")
        
        tokenizer = BertTokenizer.from_pretrained(CHECKPOINT_DIR)  # 修改：从检查点加载tokenizer
        
        print("预处理数据...")
        tokenized_dataset = dataset.map(
            lambda x: preprocess_function(x, tokenizer, MAX_SEQ_LENGTH),
            batched=True,
            remove_columns=["sentence1", "sentence2", "label"]
        )
        
        print(f"保存预处理数据到 {CACHE_DIR} 以便下次使用...")
        tokenized_dataset.save_to_disk(CACHE_DIR)
        
    tokenizer = BertTokenizer.from_pretrained(CHECKPOINT_DIR)  # 修改：从检查点加载tokenizer
    
    # 准备数据collator（同时处理MLM和NSP）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15  # 15%的token被mask
    )
    
    # 修改：从检查点加载模型，而不是从头创建
    print(f"从检查点 {CHECKPOINT_DIR} 加载模型...")
    model = BertForPreTraining.from_pretrained(CHECKPOINT_DIR)
    model = model.to(device)  # 将模型移动到GPU
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,  # 这是继续训练的轮数
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        fp16=torch.cuda.is_available(),  # 启用混合精度训练
        report_to="none"
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )
    
    
    # 开始训练 - 从检查点继续
    print(f"从检查点 {CHECKPOINT_DIR} 开始继续训练...")
    trainer.train(resume_from_checkpoint=CHECKPOINT_DIR)  # 修改：添加resume_from_checkpoint参数
    
    # 保存最终模型和tokenizer
    print("保存最终模型...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"模型已保存到 {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
