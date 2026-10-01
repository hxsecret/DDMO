import os, sys, torch, pickle
from torch.utils.data import Dataset, DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loginit import (SyscallParser, SyscallCluster, DirectAutoencoder, SequenceAutoencoder,
                     train_and_calibrate, sequence_collate_fn, BERT_MODEL_PATH)
from bert_embedder import BertSyscallVectorizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)

# ---- Stage 1: parse + cluster all benign training logs ----
parser = SyscallParser(); cluster = SyscallCluster()
direct_data = []; sequence_data = []
for i in range(10):
    fp = f"../data/data{i}/filtered.log"
    if not os.path.exists(fp):
        print("missing", fp); continue
    with open(fp, errors="ignore") as f:
        for line in f:
            ev = parser.parse_line(line.strip())
            if not ev:
                continue
            r = cluster.process_event(ev)
            if not r:
                continue
            at, evs = r
            if at == 'direct':
                direct_data.append(evs[0])
            else:
                sequence_data.append(evs)
print(f"direct={len(direct_data)} sequences={len(sequence_data)}", flush=True)

# ---- Stage 2: pre-encode with BERT (batched) ----
vec = BertSyscallVectorizer(model_path=BERT_MODEL_PATH, device=device)
print("encoding direct events...", flush=True)
direct_embs = []
B = 2048
for s in range(0, len(direct_data), B):
    texts = [f"{e['syscall']}({e['args']})" for e in direct_data[s:s+B]]
    direct_embs.append(vec.encode_texts(texts))
direct_emb = torch.cat(direct_embs, dim=0)
print("direct_emb", tuple(direct_emb.shape), flush=True)

print("encoding sequences...", flush=True)
seq_embs = []
for idx, seq in enumerate(sequence_data):
    texts = [f"{e['syscall']}({e['args']})" for e in seq]
    seq_embs.append(vec.encode_texts(texts))
    if (idx+1) % 20000 == 0:
        print(f"  {idx+1}/{len(sequence_data)}", flush=True)
print("seq_embs", len(seq_embs), flush=True)

torch.save(direct_emb, "direct_emb.pt")
torch.save(seq_embs, "seq_embs.pt")

# ---- Stage 3: train AEs on pre-encoded embeddings ----
print("training...", flush=True)
D = DirectAutoencoder(input_dim=128, hidden_dim=64).to(device)
S = SequenceAutoencoder(input_dim=128, hidden_dim=128).to(device)

direct_loader = DataLoader(TensorDataset(direct_emb), batch_size=512, shuffle=True)

class SeqTDS(Dataset):
    def __init__(self, t): self.t = t
    def __len__(self): return len(self.t)
    def __getitem__(self, i): return self.t[i]

seq_loader = DataLoader(SeqTDS(seq_embs), batch_size=128, shuffle=True, collate_fn=sequence_collate_fn)

D, tau_d = train_and_calibrate(D, direct_loader, is_sequence=False, epochs=15, device=device)
S, tau_s = train_and_calibrate(S, seq_loader, is_sequence=True, epochs=15, device=device)

# ---- Stage 4: save ----
torch.save(D.state_dict(), "syscall_ae_model")
torch.save(S.state_dict(), "sequence_ae_model")
with open("vectorizer_and_thresholds.pkl", "wb") as f:
    pickle.dump({'model_path': BERT_MODEL_PATH,
                 'tau_direct': float(tau_d), 'tau_seq': float(tau_s)}, f)
with open("anomaly_threshold.txt", "w") as f:
    f.write(f"Direct Model Threshold:\n{float(tau_d)}\nSequence Model Threshold:\n{float(tau_s)}\n")
print(f"done. tau_direct={float(tau_d):.6f} tau_seq={float(tau_s):.6f}", flush=True)