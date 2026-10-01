import torch
from transformers import AutoModel, AutoTokenizer
import os
import glob

print("loading model")
model_path = "./bert-log-pretrained-128d-with-nsp"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

os.makedirs("./embedding", exist_ok=True)

for i in range(10):
    log_file = f"./logs/filtered{i}.log"
    embeddings = []
    
    print(f"embedding  filtered {i}")
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # 使用BERT生成嵌入
            inputs = tokenizer(line, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # 使用[CLS] token的嵌入作为整个句子的表示
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu()
            embeddings.append(cls_embedding)
    
    # 保存嵌入向量
    all_embeddings = torch.cat(embeddings, dim=0)
    torch.save(all_embeddings, f"./embedding/log{i}.pt")
    print(f"Processed {log_file}, saved embeddings with shape {all_embeddings.shape}")