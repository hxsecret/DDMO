# DDMO Cross-Layer Reasoning Prompts

## LLM Configuration
- Model: deepseek/deepseek-v4-pro
- Temperature: 0
- Settings: zero-shot (instruction-only, XML output; no few-shot examples)
- System message: "You are a security analyst. Output only the requested XML format exactly."
- Flow: Stage 1 (syscall-only triage) -> Stage 2 (cross-layer reasoning with Python API context)


## Stage 1 Prompt

You are a system-call security analyst. Your task is to perform a first-pass triage on a syscall anomaly to determine if it COULD be part of a known attack pattern. You DO NOT have Python API context yet -- only syscall-level evidence.

### Attack Types to Monitor
1. **File Leak**: Reads a sensitive file and exfiltrates its content.
   - Syscall patterns: openat/open/read on /etc/, /proc/, /home/*/.ssh/, sensitive configuration or credential files; sendmsg/sendto on socket FDs occurring shortly after file reads, indicating data being sent to an external destination.

2. **Malicious Code Write**: Writes malicious payloads into files that may be unintentionally executed.
   - Syscall patterns: openat/open with O_WRONLY/O_RDWR flags targeting files that are likely to be auto-executed, such as .bashrc, .zshrc, .profile, .pythonrc, cron files (/etc/cron.d/, /var/spool/cron/), systemd service files, or __init__.py in importable packages. Pay special attention to any write/pwrite operation into these paths.

3. **IP Exposure**: Leaks the host's network identity or configuration.
   - Syscall patterns: openat/read on /etc/hosts, /etc/resolv.conf, /etc/hostname, /etc/network/interfaces, /proc/net/*; socket/connect/sendmsg/sendto targeting non-localhost addresses. Consider whether the remote IP belongs to a known trusted service or an arbitrary external host.

4. **Remote Shell Access**: Injects SSH keys or backdoor configurations for persistent remote access.
   - Syscall patterns: openat/write on ~/.ssh/authorized_keys, ~/.ssh/config; MatchingFiles on /home/* or /root/* to discover user directories; socket/connect to external hosts combined with file writes to SSH-related paths.

### Analysis Guidelines
- **File writes**: Determine whether the target file is a configuration file that could be auto-executed (shell rc files, cron, systemd) or a credential file (authorized_keys, SSH config). These are high-risk even with small payloads.
- **Network operations**: Check if the destination IP is localhost (127.0.0.1), a private/RFC1918 address (normal), or an arbitrary external host (suspicious). Also consider whether the send payload size and timing suggest data exfiltration vs. legitimate service communication.
- **Library loading** (opening .so, .cache, site-packages, lib-dynload, ld.so, encodings, importlib) is clearly benign model loading behavior and should score 0.

### Task
Analyze below. Determine if this syscall (or sequence) could be part of an attack.

### Output Format (XML)
<SuspiciousScore>
[0-10: 0=clearly benign library loading, 1-3=unlikely, 4-7=needs cross-layer check, 8-10=highly suspicious]
</SuspiciousScore>
<NeedCrossLayer>
[true or false: true if score >= 4]
</NeedCrossLayer>
<Stage1Reasoning>
[Brief reasoning: which attack type this resembles and why, citing specific syscalls, paths, or addresses]
</Stage1Reasoning>

### Inputs
<execution_phase>
{execution_phase}
</execution_phase>
<anomaly_syscall_operation>
{anomaly_syscall_operation}
</anomaly_syscall_operation>

## Stage 2 Prompt

You are a cross-layer security analyst. A syscall anomaly was flagged as suspicious in Stage 1. Now you have BOTH the syscall evidence AND the Python API application log. Perform definitive cross-layer reasoning.

### Attack Types to Monitor
1. **File Leak**: The model reads a sensitive local file and sends its contents to an external destination.
   - API indicators: ReadFile, ImmutableConst on /etc/, /proc/, or credential paths; DebugIdentityV3/V2 with debug_urls pointing to external gRPC addresses; RegisterDataset to non-localhost.
2. **Malicious Code Write**: The model writes payloads into files that are likely to be auto-executed.
   - API indicators: WriteFile, PrintV2, SaveV2, SaveSlices targeting .bashrc, .zshrc, .profile, .pythonrc, cron files, systemd service files, or __init__.py. These files can be unintentionally executed by the shell, cron daemon, or Python import system.
3. **IP Exposure**: The model leaks host network identity or configuration information.
   - API indicators: ReadFile on /etc/hosts, /etc/resolv.conf, /etc/hostname, /proc/net/*; DebugIdentityV3, RegisterDataset, or DataServiceDataset connecting to external addresses.
4. **Remote Shell Access**: The model injects attacker-controlled SSH keys or backdoor configurations.
   - API indicators: WriteFile to authorized_keys or .ssh/config; MatchingFiles on /home/* or /root/* to discover user home directories; combination of file-write APIs with network-send APIs.

### Cross-Layer Analysis Guidelines
- **API selection**: The application log contains ALL API calls from the model execution. You MUST identify which entries are relevant to the current anomaly syscall operation. Focus on APIs whose function name, arguments, or target paths spatially or semantically overlap with the syscall evidence. Ignore unrelated framework operations (Placeholder, MatMul, BiasAdd, etc.) unless they contain suspicious parameters.
- **File sensitivity**: Determine whether the target file path (from either syscall or API args) is a sensitive system file (/etc/passwd, /etc/shadow), a user credential file (.ssh/id_rsa, .aws/credentials), or an auto-executable configuration file (.bashrc, cron). These indicate clear malicious intent.
- **Network destinations**: Filter network targets by whether they are localhost (127.0.0.1, ::1), private/RFC1918 addresses (10.x, 172.16-31.x, 192.168.x -- normal for local services), well-known trusted hosts (pypi.org, conda.io, github.com), or arbitrary external IPs/hostnames (highly suspicious). A gRPC connection to an unknown external host with no corresponding legitimate framework API is a strong malicious indicator.
- **Content assessment**: When the API log shows file contents or network payloads, assess whether they contain benign data (model weights, tensor shapes, configuration flags) or suspicious data (raw file contents, shell commands, SSH public keys, base64-encoded blobs).
- **Context consistency**: Check whether the Python API context can plausibly explain the syscall behavior. For example, an openat on a .so file during Model Loading Phase is fully explained by the framework's dynamic library loading. But an openat on /etc/passwd with a subsequent DebugIdentityV3 to an external URL has no legitimate framework explanation.

### Classification Rules
- **Benign**: The syscall operation is fully explained by the Python API context in the given execution phase. For example, library loading, model weight reading, or localhost service communication are normal.
- **Malicious**: At least ONE of the following is true:
  (a) The syscall operation directly matches an attack pattern AND a corresponding suspicious Python API (ImmutableConst, DebugIdentityV3, WriteFile, RegisterDataset, PrintV2, etc.) is found in the application log with matching malicious parameters (sensitive paths, external URLs).
  (b) The application log contains suspicious APIs with clearly malicious parameters (e.g., ImmutableConst reading /etc/passwd, DebugIdentityV3 sending to an external gRPC address, WriteFile targeting .ssh/authorized_keys) that have NO legitimate framework explanation in the given execution phase.

### Output Format (XML)
<IntentScore>
[0-10: 0=definitely benign, 1-3=unlikely malicious, 4-7=suspicious, 8-10=definitely malicious]
</IntentScore>
<Category>
[Benign or Malicious]
</Category>
<EvidenceChain>
[Step-by-step cross-layer evidence: (1) what the syscall anomaly shows, (2) what the matched API context reveals, (3) whether the two layers are consistent or contradictory, (4) final judgment citing specific file paths, network addresses, API names, and their security implications]
</EvidenceChain>

### Inputs
<execution_phase>
{execution_phase}
</execution_phase>
<anomaly_syscall_operation>
{anomaly_syscall_operation}
</anomaly_syscall_operation>
<application_log>
{application_log}
</application_log>
