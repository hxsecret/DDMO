import os, sys, torch, pickle
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loginit import (DirectAutoencoder, SequenceAutoencoder,
                     train_and_calibrate, sequence_collate_fn, BERT_MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)

direct_emb = torch.load("direct_emb.pt")
seq_embs = torch.load("seq_embs.pt")
print("direct_emb", tuple(direct_emb.shape), "seq_embs", len(seq_embs), flush=True)

class TDS(Dataset):
    def __init__(self, t): self.t = t
    def __len__(self): return len(self.t)
    def __getitem__(self, i): return self.t[i]   # raw tensor, not tuple

direct_loader = DataLoader(TDS(direct_emb), batch_size=512, shuffle=True)
seq_loader = DataLoader(TDS(seq_embs), batch_size=128, shuffle=True, collate_fn=sequence_collate_fn)

D = DirectAutoencoder(input_dim=128, hidden_dim=64).to(device)
S = SequenceAutoencoder(input_dim=128, hidden_dim=128).to(device)

print("training direct...", flush=True)
D, tau_d = train_and_calibrate(D, direct_loader, is_sequence=False, epochs=15, device=device)
print("training sequence...", flush=True)
S, tau_s = train_and_calibrate(S, seq_loader, is_sequence=True, epochs=15, device=device)

torch.save(D.state_dict(), "syscall_ae_model")
torch.save(S.state_dict(), "sequence_ae_model")
with open("vectorizer_and_thresholds.pkl", "wb") as f:
    pickle.dump({'model_path': BERT_MODEL_PATH, 'tau_direct': float(tau_d), 'tau_seq': float(tau_s)}, f)
with open("anomaly_threshold.txt", "w") as f:
    f.write(f"Direct Model Threshold:\n{float(tau_d)}\nSequence Model Threshold:\n{float(tau_s)}\n")
print(f"done. tau_direct={float(tau_d):.6f} tau_seq={float(tau_s):.6f}", flush=True)