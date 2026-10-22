We appreciate valuable feedback. 

## General Questions

**Q1 Autoencoder Training and Evaluation Details** (Reviewers A, B, C)

The autoencoder is trained unsupervised on benign traces, using 10 benign Hugging Face models: bert-base-uncased, MiniLM-L6-H384-uncased, bert-base-chinese, distilgpt2, t5-small, resnet-50, emotion-english-distilroberta-base, whisper-tiny, twitter-roberta-base-sentiment, and tapas-base-finetuned-wtq. Each model is loaded once and runs one typical inference under strace. The autoencoder has hidden widths 128, 64, and a bottleneck of 32, and decodes back to 128, with MSE loss and a threshold θ at the 99.9th percentile of normal reconstruction error. Training and dataset models do not overlap, though architectures partly do, as both use BERT-style encoder-only transformers; the malicious dataset also uses architectures absent from training, such as YAMNet, VGG16, and EfficientNet, so evaluation is not limited to same-architecture samples. The embedding is a BERT continued from bert-base-uncased, with hidden size 128, intermediate size 512, and 2 attention heads, further pretrained on the benign logs via masked language modeling at 15% and next-sentence prediction for 6 epochs, batch 32, learning rate 2e-5, and sequence length 512, producing 128-dim [CLS] embeddings. The weights and scripts are in the public repository.


**Q2 LLM Configuration and Prompts** (Reviewers A, B, C)

We use deepseek-v4-pro for cross-layer reasoning, with temperature=0 and zero-shot prompting. The reasoning prompts are released in fine_grained_detect.py, and we also uploaded them as standalone files under `prompt/`.

On coverage, we clarify that our instrumentation is neither whitelists nor blacklists: it intercepts all public callables of the target library, wraps only callable functions, and uses defensive serialization. The prompt gives a deterministic skeleton that works across frameworks; the LLM's value is rebuilding framework-specific security semantics on demand. The instrumentation prompt is open-sourced at https://anonymous.4open.science/r/DDMO-F184/.

**Q3 Insufficient Generalization Evidence** (Reviewers A, C)

First, the method encodes no predefined attack patterns: the autoencoder is unsupervised on benign traces, and the LLM only checks whether a suspicious system call can be explained by its Python semantics. Second, our evaluation is not limited to the TensorAbuse dataset. We also test on the real-world MalHug dataset, which mainly abuses Pickle deserialization. We detect both, which supports generalization.

## Review #A

**Insufficient baselines (Rigor1)**

Reference [24] is indeed a dynamic detection method. We tested it on our dataset and found that it yields a 100% false-positive rate on benign models, because its detection relies on a blacklist of a few system calls (socket, connect, execve, chmod), and calls like execve appear in virtually every normal model load. The paper itself acknowledges this by requiring manual review of the flagged calls. We will include reference [24] as a dynamic-analysis baseline in future revisions.


**how to handle obfuscation (Rigor2)**

Code obfuscation is a common evasion technique, and our method can still detect it. Obfuscation can only alter the Python-level semantics from instrumentation, not the kernel-level system calls. When the code is obfuscated, the instrumentation either records misleading logs or no relevant logs at all. Either way, the kernel still captures the real syscalls, and the misalignment between the two layers triggers an alert. The TensorAbuse attacks in our dataset use this kind of disguise, and we detect them successfully.

**The benign FP analysis is small (Rigor2)**

We will run a larger benign evaluation on public hubs.


**Missing Autoencoder details (Presentation1, Question1)**

Please see Q1.

**Missing LLM prompt (Presentation2, Question2)**

Please see Q2.

## Review #B

**trigger coverage (Weakness1, Rigor1)**

Each malicious model triggers its attack with a single load and inference. The dataset includes the construction code, and we will provide a per-model load script that triggers the attack in one step for easy reproduction.

**benign training data and LLM configuration (Weakness2, Rigor2)**

Please see Q1/Q2.

**Cross-framework semantic instrumentation (Weakness3, Rigor3)**

Please see Q2.

**Heavy dependence on the LLM (Rigor4)**

We have added ablation experiments using different models. On MalHug, they all achieved an F1 score of 0.9870. On the TensorAbuse Synthetic Dataset, they all achieved a precision of 1.0, with recall rates of 100% (160/160), 100% (160/160), and 98.75% (158/160), respectively. This shows that our method does not depend heavily on the capability of the LLM. We will add the detailed experimental results to the revision.

**Missing cross-operation attack chains (Weakness4, Rigor5)**

We can detect such attacks. First, each step is already anomalous on its own: a chmod, a sensitive file read, or a socket send deviates from benign behavior and is flagged by the autoencoder. The Python log then adds context, such as which API caused the exfiltration and the target IP, linking the steps at the semantic level. Second, File Leakage in our dataset is exactly such a read-then-exfiltrate case, and we detect it.

## Review #C

**novelty**

We humbly clarify that our novelty is the first no-prior-attack-knowledge unsupervised dynamic analysis for malicious model detection. It is novel because it fundamentally changes the detection paradigm: rather than enumerating known attacks, it models the benign execution distribution. This design is what makes the detector generalize to different attacks without any modification, which static and blacklist-based approaches cannot achieve. Regarding the role of the LLM in instrumentation generation, please see Q2 for details.


**Autoencoder details(Weakness1, Rigor2, Question1)**

Please see Q1.

**Generalization evidence (Weakness2, Rigor3, Question2)**

Please see Q3.

**trigger limitation (Weakness3, Rigor4)**

This is a general limitation of dynamic detection. To maximize coverage, our threat model recommends exercising the model across all expected usage scenarios during testing, including varied input shapes and edge-case inputs, which helps activate conditionally triggered logic. We will add a detailed discussion of this limitation in the revision.

**implementation details (Weakness4)**

Please see Q1/Q2.

**"no prior knowledge" is too strong (Rigor1)**

We will change it to "DDMO does not require predefined attack patterns."

**Unclear detection target (Rigor5)**

Our detection target is whether the model actually executed harmful behavior. The 11 proof-of-concept models in MalHug demonstrate exploitability but do not perform destructive operations, so they are labeled benign under this definition. We will add this clarification in the revision.

**Missing reproducible parameters (Verifiability and Transparency)**

Please see Q1/Q2. All details are added to the public repository.

**Conflicting ablation-result wording (Presentation)**

We will change "detection accuracy" to "precision" to match Table VI.


**LLM configuration (Question3)**

Please see Q2.