import re

class SyscallParser:
    def __init__(self):
        # store unfinished calls like {pid: "syscall_name(args"}
        self.unfinished_calls = {}
        
        # match complete call: 1234 read(3, buf, 1024) = 1024
        self.regex_complete = re.compile(r'^(\d+)\s+([a-zA-Z0-9_]+)\((.*)\)\s*=\s*(.*)$')
        # match unfinished segment start: 1234 read(3, <unfinished ...>
        self.regex_unfinished = re.compile(r'^(\d+)\s+([a-zA-Z0-9_]+)\((.*)\s*<unfinished \.\.\.>$')
        # match resumed segment end: 1234 <... read resumed> buf, 1024) = 1024
        self.regex_resumed = re.compile(r'^(\d+)\s+<\.\.\.\s*([a-zA-Z0-9_]+)\s*resumed>\s*(.*)\)\s*=\s*(.*)$')
    
    def parse_line(self, line):
        line = line.strip()
        
        # 1. try to match complete call
        match = self.regex_complete.match(line)
        if match:
            return self._build_event(match.groups())
            
        # 2. try to match unfinished call
        match = self.regex_unfinished.match(line)
        if match:
            pid, syscall, partial_args = match.groups()
            self.unfinished_calls[pid] = {'syscall': syscall, 'args': partial_args}
            return None # not yet complete, don't return event
            
        # 3. try to match resumed call
        match = self.regex_resumed.match(line)
        if match:
            pid, syscall, rest_args, ret_val = match.groups()
            if pid in self.unfinished_calls and self.unfinished_calls[pid]['syscall'] == syscall:
                full_args = self.unfinished_calls[pid]['args'] + rest_args
                del self.unfinished_calls[pid]
                return {'pid': pid, 'syscall': syscall, 'args': full_args, 'ret': ret_val}
        
        return None

    def _build_event(self, groups):
        return {'pid': groups[0], 'syscall': groups[1], 'args': groups[2], 'ret': groups[3]}
    
class SyscallCluster:
    def __init__(self):
        # Category 1: directly detected sensitive calls
        self.direct_syscalls = {
            'setuid', 'setgid', 'chroot', 'mount', 'umount', 'ptrace', 'kill', 'tkill', 'execve', 'access',
            'sigaction', 'sigprocmask', 'prctl', 'capset', 'capget', 'clone', 'fork', 'vfork',
            'rt_sigprocmask', 'rt_sigaction', 'epoll_create1', 'eventfd2', 'epoll_create', 'epoll_ctl',  'ioctl','getdents64' 
        }
        self.process_syscalls = {
            # file operations
            'open', 'openat', 'creat', 'close', 'read', 'write', 'pread', 'pread64',
            'pwrite', 'pwrite64', 'lseek', 'fstat', 'stat', 'lstat', 'unlink', 'rmdir', 'mkdir', 'rename',
            # network operations
            'socket', 'connect', 'bind', 'listen', 'sendmsg', 'recvmsg', 'shutdown', 'getsockopt', 'setsockopt'
        }
        
        # Category 2: state tracking dictionary {fd: [syscall_events]}
        self.process_state = {}
    
      
    def _extract_fd(self, args_str):
        """
        Extract the file descriptor (FD) from the syscall argument string.
        For example, extract "3" from "3, 0x7ffd..., 1024".
        """
        if not args_str:
            return None
            
        # split by comma, take the first argument and strip whitespace
        first_arg = args_str.split('<')[0].strip()
        
        # check if it is a pure digit (valid FD is a non-negative integer)
        if first_arg.isdigit():
            return first_arg
            
        return None
    
    def process_event(self, event):
        syscall = event['syscall']
        pid = event['pid']
        
        # (1) handle directly exposed malicious calls
        if syscall in self.direct_syscalls:
            return ('direct', [event]) # send to model (1)
            
        # (2) handle file/network aggregation logic
        # extract file descriptor (FD) or path via simple regex from args
        fd = self._extract_fd(event['args']) 
        # print(f"Processing event: PID={pid}, Syscall={syscall}, Extracted FD={fd}")
            
        if syscall in ['open', 'openat', 'creat', 'socket', 'bind', 'listen']:
            # operation start, initialize queue
            ret_fd = event['ret'] # open returns the FD
            ret_fd = ret_fd.split('<')[0].strip()
            if ret_fd not in self.process_state:
                    self.process_state[ret_fd] = {}
            if ret_fd.isdigit():
                self.process_state[ret_fd] = [event]

        elif syscall in ['close', 'shutdown', 'unlink', 'rmdir']:
            # operation end, extract entire sequence for analysis
            if fd in self.process_state:
                self.process_state[fd].append(event)
                sequence = self.process_state.pop(fd)
                return ('sequence', sequence) # send to model (2)
                
        elif syscall in self.process_syscalls:
            # collecting... push to queue
            if fd in self.process_state:
                self.process_state[fd].append(event)
                
        return None
    
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class DirectDataset(Dataset):
    def __init__(self, data, vectorizer):
        self.data = data
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Category (1) is a single event, transform directly
        event = self.data[idx][0] # data is a list like [[event1], [event2], ...]
        sys_t, arg_t = self.vectorizer.transform_direct(event)
        return sys_t, arg_t

class SequenceDataset(Dataset):
    def __init__(self, data, vectorizer):
        self.data = data
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Category (2) is an event sequence
        sequence = self.data[idx]
        sys_t, arg_t = self.vectorizer.transform_sequence(sequence)
        return sys_t, arg_t

def sequence_collate_fn(batch):
    syscall_tensors = [item[0] for item in batch]
    arg_tensors = [item[1] for item in batch]
    
    # pad sequences of varying lengths within a batch with 0 to align to max length
    syscalls_padded = pad_sequence(syscall_tensors, batch_first=True, padding_value=0)
    # arg tensor is 2D (seq_len, feature_dim), also pad with 0.0
    args_padded = pad_sequence(arg_tensors, batch_first=True, padding_value=0.0)
    
    return syscalls_padded, args_padded
    
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

class DualSyscallVectorizer:
    def __init__(self, max_features=128):
        self.syscall_encoder = LabelEncoder()
        # character-level N-gram to extract argument features
        self.arg_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=max_features)
        self.is_fitted = False

    def fit(self, all_events):
        """
        Pass in all normal training data (whether single events or items within sequences) to fit.
        """
        syscalls = [e['syscall'] for e in all_events]
        args = [e['args'] for e in all_events]
        
        self.syscall_encoder.fit(syscalls)
        self.arg_vectorizer.fit(args)
        self.is_fitted = True
        self.num_syscalls = len(self.syscall_encoder.classes_)

    def transform_direct(self, event):
        """Process (1) single call, return 1D Tensor"""
        syscall_id = self.syscall_encoder.transform([event['syscall']])[0] if event['syscall'] in self.syscall_encoder.classes_ else 0
        arg_vec = self.arg_vectorizer.transform([event['args']]).toarray()[0]
        
        return torch.tensor([syscall_id], dtype=torch.long), torch.tensor(arg_vec, dtype=torch.float32)

    def transform_sequence(self, sequence):
        """Process (2) sequence call, return 2D Tensor (seq_len, feature_dim)"""
        syscall_ids = [self.syscall_encoder.transform([e['syscall']])[0] if e['syscall'] in self.syscall_encoder.classes_ else 0 for e in sequence]
        arg_vecs = self.arg_vectorizer.transform([e['args'] for e in sequence]).toarray()
        
        return torch.tensor(syscall_ids, dtype=torch.long), torch.tensor(arg_vecs, dtype=torch.float32)
    
import torch.nn as nn

# ---------------- Model (1): MLP Autoencoder for Direct Syscall ----------------
class DirectAutoencoder(nn.Module):
    def __init__(self, num_syscalls, arg_dim, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embed = nn.Embedding(num_syscalls + 1, embed_dim, padding_idx=0)
        input_dim = embed_dim + arg_dim
        
        # simple encoder -> bottleneck -> decoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, syscall_idx, arg_vec):
        emb = self.embed(syscall_idx) # (batch, embed_dim)
        if emb.dim() == 3:
            emb = emb.squeeze(1)
        x = torch.cat([emb, arg_vec], dim=-1) # (batch, input_dim)
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed, x

# ---------------- Model (2): LSTM Autoencoder for Sequence ----------------
class SequenceAutoencoder(nn.Module):
    def __init__(self, num_syscalls, arg_dim, embed_dim=16, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(num_syscalls + 1, embed_dim, padding_idx=0)
        input_dim = embed_dim + arg_dim
        
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.rebuild_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, syscall_idx, arg_vec):
        emb = self.embed(syscall_idx) # (batch, seq, embed_dim)
        x = torch.cat([emb, arg_vec], dim=-1) # (batch, seq, input_dim)
        
        _, (hidden, cell) = self.encoder(x)
        seq_len = x.size(1)
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        
        reconstructed = self.rebuild_layer(decoder_output)
        return reconstructed, x
    
import torch.optim as optim

def train_and_calibrate(model, data_loader, is_sequence=False, epochs=10, device='cpu'):
    """
    Train the model and automatically compute anomaly detection threshold using 3-Sigma rule.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss(reduction='none') # keep per-sample loss independent
    
    # 1. Training phase
    for epoch in range(epochs):
        total_loss = 0
        for syscall_batch, arg_batch in data_loader:
            syscall_batch = syscall_batch.to(device)
            arg_batch = arg_batch.to(device)
            optimizer.zero_grad()
            reconstructed, original = model(syscall_batch, arg_batch)
            loss = criterion(reconstructed, original).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs}, Average Loss: {total_loss/len(data_loader):.8f}")
            
    # 2. Threshold calibration phase
    model.eval()
    all_losses = []
    with torch.no_grad():
        for syscall_batch, arg_batch in data_loader:
            syscall_batch = syscall_batch.to(device)
            arg_batch = arg_batch.to(device)
            reconstructed, original = model(syscall_batch, arg_batch)
            # compute MSE per sample
            if is_sequence:
                # average over sequence dimensions: shape (batch_size, seq_len, feature_dim) -> (batch_size,)
                sample_losses = criterion(reconstructed, original).mean(dim=[1, 2])
            else:
                # average over single feature dimensions: shape (batch_size, feature_dim) -> (batch_size,)
                sample_losses = criterion(reconstructed, original).mean(dim=1)
                
            all_losses.extend(sample_losses.tolist())
            
    # compute mean and standard deviation
    mu = np.mean(all_losses)
    sigma = np.std(all_losses)
    # threshold = mu + 3 * sigma 
    threshold = np.percentile(all_losses, 99.9) 

    print(f"Calibration complete! mu={mu:.8f}, sigma={sigma:.8f}, recommended threshold tau={threshold:.8f}")
    return model, threshold

def build_and_train_pipeline():
    # ==========================================
    # Stage 1: Data Loading & Clustering
    # ==========================================
    parser = SyscallParser()
    cluster = SyscallCluster()
    
    direct_data = []   # store category (1)
    sequence_data = [] # store category (2)
    all_events = []    # store all events for fitting Vectorizer
    
    print(">>> Stage 1: Reading and clustering logs...")
    for i in range(10):
        filepath = f"../data/data{i}/filtered.log"
        # filepath = "test.log" 
        if not os.path.exists(filepath):
            continue
            
        print(f"  Processing: {filepath}")
        with open(filepath, 'r') as f:
            for line in f:
                event = parser.parse_line(line)
                if event:
                    all_events.append(event)
                    # feed to state machine for clustering
                    cluster_result = cluster.process_event(event)
                    if cluster_result:
                        action_type, events = cluster_result
                        if action_type == 'direct':
                            direct_data.append(events)
                        elif action_type == 'sequence':
                            sequence_data.append(events)
                            
    print(f"Data extraction complete: found {len(direct_data)} direct calls, {len(sequence_data)} behavior sequences.")
    print(direct_data[:5]) # print first two direct call examples
    print(sequence_data[:2]) # print first two sequence call examples

    # ==========================================
    # Stage 2: Feature Engineering
    # ==========================================
    print("\n>>> Stage 2: Fitting feature vectorizer...")
    # For million-level logs, limiting TF-IDF vocabulary to 256 balances speed and accuracy
    vectorizer = DualSyscallVectorizer(max_features=256)
    vectorizer.fit(all_events)
    num_syscalls = vectorizer.num_syscalls
    print(f"Vectorizer ready, syscall vocabulary size: {num_syscalls}")

    # Build DataLoaders
    # Category 1: Direct DataLoader
    direct_dataset = DirectDataset(direct_data, vectorizer)
    direct_loader = DataLoader(direct_dataset, batch_size=512, shuffle=True)
    
    # Category 2: Sequence DataLoader (must pass custom collate_fn)
    seq_dataset = SequenceDataset(sequence_data, vectorizer)
    seq_loader = DataLoader(seq_dataset, batch_size=128, shuffle=True, collate_fn=sequence_collate_fn)

    # ==========================================
    # Stage 3: Model Initialization & Hyperparameters
    # ==========================================
    print("\n>>> Stage 3: Initializing Autoencoder models...")
    # Hyperparameter rationale:
    # 1. args_dim matches max_features=256
    # 2. embed_dim=32 is sufficient to encode a few hundred syscalls
    # 3. seq_model hidden_dim=128, because sequences involve temporal complexity and need larger capacity
    
    direct_model = DirectAutoencoder(
        num_syscalls=num_syscalls, 
        arg_dim=256, 
        embed_dim=32, 
        hidden_dim=64
    )
    
    seq_model = SequenceAutoencoder(
        num_syscalls=num_syscalls, 
        arg_dim=256, 
        embed_dim=32, 
        hidden_dim=128
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    direct_model.to(device)
    seq_model.to(device)
    print(f"Using device: {device}")

    # ==========================================
    # Stage 4: Training & Threshold Calibration
    # ==========================================
    print("\n>>> Stage 4: Training and computing thresholds...")
    
    print("--- Training Direct model ---")
    direct_model, tau_direct = train_and_calibrate(
        direct_model, direct_loader, is_sequence=False, epochs=15, device=device
    )
    
    print("\n--- Training Sequence model ---")
    seq_model, tau_seq = train_and_calibrate(
        seq_model, seq_loader, is_sequence=True, epochs=15, device=device
    )

    print("\n>>> Pipeline complete!")
    return vectorizer, direct_model, seq_model, tau_direct, tau_seq

# Entry point
import pickle
def train():
    vectorizer, d_model, s_model, t_d, t_s = build_and_train_pipeline()
    d_model_path = "./syscall_ae_model"
    s_model_path = "./sequence_ae_model"
    torch.save(d_model.state_dict(), d_model_path)
    torch.save(s_model.state_dict(), s_model_path)
    print(f"Direct model weights saved to: {d_model_path}")
    print(f"Sequence model weights saved to: {s_model_path}")
    t_path = "./anomaly_threshold.txt"
    with open(t_path, 'w') as f:
        f.write("Direct Model Threshold:\n")
        f.write(str(t_d))
        f.write("\nSequence Model Threshold:\n")
        f.write(str(t_s))
    print(f"Model thresholds saved to: {t_path}")
    with open("./vectorizer_and_thresholds.pkl", "wb") as f:
        pickle.dump({
            'vectorizer': vectorizer,
            'tau_direct': t_d,
            'tau_seq': t_s
        }, f)

# train()

def run_detection(log_file, result_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # 1. Load feature extractor and thresholds
    print("[*] Loading Vectorizer and detection thresholds...")
    with open("./vectorizer_and_thresholds.pkl", "rb") as f:
        saved_data = pickle.load(f)
        vectorizer = saved_data['vectorizer']
        tau_direct = saved_data['tau_direct']
        tau_seq = saved_data['tau_seq']
    tau_seq = tau_seq*2
    tau_direct = 0.0015
    print(tau_direct, tau_seq)
    # 2. Initialize and load model weights
    print("[*] Loading model weights...")
    num_syscalls = vectorizer.num_syscalls
    
    # Note: hyperparameters here must be identical to those used during training
    d_model = DirectAutoencoder(num_syscalls=num_syscalls, arg_dim=256, embed_dim=32, hidden_dim=64)
    s_model = SequenceAutoencoder(num_syscalls=num_syscalls, arg_dim=256, embed_dim=32, hidden_dim=128)
    
    d_model.load_state_dict(torch.load("./syscall_ae_model", map_location=device))
    s_model.load_state_dict(torch.load("./sequence_ae_model", map_location=device))
    
    d_model.to(device).eval()
    s_model.to(device).eval()

    # 3. Parse and cluster filtered.log
    print(f"[*] Parsing target log: {log_file}")
    parser = SyscallParser()
    cluster = SyscallCluster()
    events_to_check = []
    
    with open(log_file, 'r') as f:
        for line in f:
            event = parser.parse_line(line)
            if event:
                cluster_result = cluster.process_event(event)
                if cluster_result:
                    events_to_check.append(cluster_result)
    print(f"[*] Parsing complete! Extracted {len(events_to_check)} events/sequences to check.")
    # 4. Run classification and anomaly logging
    print("[*] Running anomaly detection...")
    criterion = nn.MSELoss(reduction='mean')
    anomaly_count = 0
    
    with open(result_file, 'w', encoding='utf-8') as out_f, torch.no_grad():
        for action_type, events in events_to_check:
            
            if action_type == 'direct':
                # handle single record (1)
                sys_t, arg_t = vectorizer.transform_direct(events[0])
                sys_t = sys_t.unsqueeze(0).to(device)
                arg_t = arg_t.unsqueeze(0).to(device)
                
                recon, orig = d_model(sys_t, arg_t)
                loss = criterion(recon, orig).item()
                
                if loss > tau_direct:
                    anomaly_count += 1
                _write_anomaly(out_f, "High-Risk Single Event (Direct)", loss, tau_direct, events)
                    
            elif action_type == 'sequence':
                # handle call sequence (2)
                sys_t, arg_t = vectorizer.transform_sequence(events)
                sys_t = sys_t.unsqueeze(0).to(device)
                arg_t = arg_t.unsqueeze(0).to(device)
                
                recon, orig = s_model(sys_t, arg_t)
                loss = criterion(recon, orig).item()
                
                if loss > tau_seq:
                    anomaly_count += 1
                _write_anomaly(out_f, "Behavioral Sequence Anomaly (Sequence)", loss, tau_seq, events)

    print(f"[*] Detection complete! Found {anomaly_count} anomalies.")
    print(f"[*] Results saved to: {result_file}")


def _write_anomaly(file_handler, alert_type, loss, threshold, events):
    """Helper function: format anomaly info for output"""
    file_handler.write("-" * 60 + "\n")
    file_handler.write(f"[!] Alert Type: {alert_type}\n")
    file_handler.write(f"    Anomaly Score (Loss): {loss:.8f} (Threshold: {threshold:.4f})\n")
    file_handler.write(f"    Process PID: {events[0]['pid']}\n")
    file_handler.write("    Syscall Trace:\n")
    
    for e in events:
        # for readability, format dict as plain text
        file_handler.write(f"      -> {e['syscall']}({e['args']}) = {e['ret']}\n")
    file_handler.write("\n")

# Run detection
if __name__ == '__main__':
    run_detection("../../experiment/RQ2/test/filtered.log", "../../experiment/RQ2/test/result_anomalies.txt")
