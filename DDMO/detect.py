import numpy as np
import torch
import torch.nn as nn
import pickle
from loginit import SyscallParser, SyscallCluster, DualSyscallVectorizer, DirectAutoencoder, SequenceAutoencoder

def run_detection(log_file, result_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    print("[*] Loading Vectorizer and model weights...")
    with open("./vectorizer_and_thresholds.pkl", "rb") as f:
        vectorizer = pickle.load(f)['vectorizer']

    num_syscalls = vectorizer.num_syscalls
    d_model = DirectAutoencoder(num_syscalls=num_syscalls, arg_dim=256, embed_dim=32, hidden_dim=64)
    s_model = SequenceAutoencoder(num_syscalls=num_syscalls, arg_dim=256, embed_dim=32, hidden_dim=128)
    d_model.load_state_dict(torch.load("./syscall_ae_model", map_location=device))
    s_model.load_state_dict(torch.load("./sequence_ae_model", map_location=device))
    d_model.to(device).eval()
    s_model.to(device).eval()

    print(f"[*] Parsing log: {log_file}")
    parser = SyscallParser()
    cluster = SyscallCluster()
    events_to_check = []
    with open(log_file, 'r') as f:
        for line in f:
            event = parser.parse_line(line)
            if event:
                result = cluster.process_event(event)
                if result:
                    events_to_check.append(result)

    print("[*] Computing anomaly scores for all events...")
    criterion = nn.MSELoss(reduction='mean')
    direct_losses = []
    sequence_losses = []
    all_scored = []

    with torch.no_grad():
        for action_type, events in events_to_check:
            if action_type == 'direct':
                sys_t, arg_t = vectorizer.transform_direct(events[0])
                sys_t = sys_t.unsqueeze(0).to(device)
                arg_t = arg_t.unsqueeze(0).to(device)
                recon, orig = d_model(sys_t, arg_t)
                loss = criterion(recon, orig).item()
                direct_losses.append(loss)
                all_scored.append(('direct', events, loss))
            elif action_type == 'sequence':
                sys_t, arg_t = vectorizer.transform_sequence(events)
                sys_t = sys_t.unsqueeze(0).to(device)
                arg_t = arg_t.unsqueeze(0).to(device)
                recon, orig = s_model(sys_t, arg_t)
                loss = criterion(recon, orig).item()
                sequence_losses.append(loss)
                all_scored.append(('sequence', events, loss))

    tau_direct = np.percentile(direct_losses, 99.5) if direct_losses else float('inf')
    tau_seq = np.percentile(sequence_losses, 99.5) if sequence_losses else float('inf')
    print(f"    Direct  threshold (99.5th): {tau_direct:.8f}")
    print(f"    Sequence threshold (99.5th): {tau_seq:.8f}")

    anomaly_count = 0
    with open(result_file, 'w', encoding='utf-8') as out_f:
        out_f.write("=" * 80 + "\n")
        out_f.write("DDMO Anomaly Detection Results\n")
        out_f.write("=" * 80 + "\n")
        out_f.write(f"Log file: {log_file}\n")
        out_f.write(f"Total events: {len(all_scored)}\n")
        out_f.write(f"Direct  threshold (99.5th): {tau_direct:.8f}  (total {len(direct_losses)} events)\n")
        out_f.write(f"Sequence threshold (99.5th): {tau_seq:.8f}  (total {len(sequence_losses)} events)\n")
        out_f.write("=" * 80 + "\n\n")

        for action_type, events, loss in all_scored:
            if action_type == 'direct' and loss > tau_direct:
                anomaly_count += 1
                e = events[0]
                out_f.write(f"{'-'*60}\n")
                out_f.write(f"[{anomaly_count}] ANOMALY | Type: Direct\n")
                out_f.write(f"    Loss={loss:.8f} | Threshold={tau_direct:.8f}\n")
                out_f.write(f"    PID={e['pid']} | {e['syscall']}({e['args'][:120]}) = {e['ret']}\n\n")
            elif action_type == 'sequence' and loss > tau_seq:
                anomaly_count += 1
                out_f.write(f"{'-'*60}\n")
                out_f.write(f"[{anomaly_count}] ANOMALY | Type: Sequence ({len(events)} steps)\n")
                out_f.write(f"    Loss={loss:.8f} | Threshold={tau_seq:.8f}\n")
                out_f.write(f"    PID={events[0]['pid']} | {events[0]['syscall']} -> {events[-1]['syscall']}\n")
                for j, e in enumerate(events, 1):
                    out_f.write(f"      Step {j}: {e['syscall']}({e['args'][:100]}) = {e['ret']}\n")
                out_f.write("\n")

        out_f.write("=" * 80 + "\n")
        out_f.write(f"Detection complete | Total {len(all_scored)} events | Anomalies (top 0.5%): {anomaly_count}\n")

    print(f"[*] Detection complete: {len(all_scored)} events, {anomaly_count} anomalies")
    print(f"[*] Results saved to: {result_file}")

if __name__ == "__main__":
    run_detection("./test_model/syscall.log", "result_anomalies.txt")
