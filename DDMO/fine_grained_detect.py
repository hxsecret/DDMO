import re
import json
import os


def parse_anomalies(path):
    with open(path, 'r') as f:
        text = f.read()
    header_match = re.search(r'Direct\s+threshold.*?(\S+).*?Sequence threshold.*?(\S+)', text)
    tau_direct = float(header_match.group(1)) if header_match else 0
    tau_seq = float(header_match.group(2)) if header_match else 0

    direct_block = re.compile(
        r'^\[(\d+)\] ANOMALY \| Type: Direct\s*\n'
        r'    Loss=(\S+) \| Threshold=(\S+)\s*\n'
        r'    PID=(\d+) \| (\S+)\((.*?)\) = (.+)$', re.MULTILINE)
    sequence_block = re.compile(
        r'^\[(\d+)\] ANOMALY \| Type: Sequence \((\d+) steps\)\s*\n'
        r'    Loss=(\S+) \| Threshold=(\S+)\s*\n'
        r'    PID=(\d+) \| (\S+) -> (\S+)\s*\n'
        r'((?:\s{6}Step \d+: .+\n?)+)', re.MULTILINE)

    anomalies = []
    for m in direct_block.finditer(text):
        idx, loss, thresh, pid, syscall, args, ret = m.groups()
        anomalies.append({'id': int(idx), 'type': 'direct', 'pid': int(pid),
                          'loss': float(loss), 'threshold': float(thresh),
                          'syscall': syscall, 'args': args.strip(), 'ret': ret.strip(),
                          'steps': [{'syscall': syscall, 'args': args.strip(), 'ret': ret.strip()}]})
    for m in sequence_block.finditer(text):
        idx, nsteps, loss, thresh, pid, start_sys, end_sys, steps_text = m.groups()
        steps = []
        for sm in re.finditer(r'Step (\d+): (\S+)\((.*?)\) = (.+)$', steps_text, re.MULTILINE):
            steps.append({'syscall': sm.group(2), 'args': sm.group(3).strip(), 'ret': sm.group(4).strip()})
        anomalies.append({'id': int(idx), 'type': 'sequence', 'pid': int(pid),
                          'loss': float(loss), 'threshold': float(thresh),
                          'syscall': f"{start_sys} -> {end_sys}",
                          'args': '', 'ret': '', 'steps': steps, 'nsteps': int(nsteps)})
    anomalies.sort(key=lambda x: x['id'])
    print(f"[*] Parsed {len(anomalies)} anomalies")
    return anomalies, tau_direct, tau_seq


def parse_api_calls(path):
    api_calls = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: api_calls.append(json.loads(line))
            except json.JSONDecodeError: continue
    print(f"[*] Parsed {len(api_calls)} API records")
    return api_calls


def match_api_context(anomaly, api_calls):
    anomaly_paths = set()
    anomaly_keywords = set()
    for s in anomaly['steps']:
        for m in re.finditer(r'"(/[^"]+)"', s['args']):
            path = m.group(1)
            anomaly_paths.add(path)
            for part in path.split('/'):
                if len(part) > 2: anomaly_keywords.add(part.lower())
        for word in re.findall(r'[a-zA-Z_]{4,}', s['args']):
            anomaly_keywords.add(word.lower())

    S_APIS = ('ImmutableConst', 'DebugIdentityV3', 'DebugIdentityV2',
              'ReadFile', 'WriteFile', 'PrintV2', 'RegisterDataset',
              'MatchingFiles', 'FixedLengthRecordDatasetV2', 'CSVDatasetV2')
    matched = []
    for api in api_calls:
        args_str = json.dumps(api.get('args_summary', {}), ensure_ascii=False)
        fn = api.get('func_name', '')
        structural = 0.0
        for p in anomaly_paths:
            if len(p) > 4 and p in args_str: structural += 1.0
            elif len(p) > 4 and os.path.basename(p) in args_str: structural += 0.5
        if structural == 0:
            for kw in anomaly_keywords:
                if len(kw) > 3 and kw in args_str.lower(): structural += 0.02
        if structural >= 0.02:
            is_susp = any(s in fn for s in S_APIS)
            matched.append((api, structural + (0.5 if is_susp else 0.0)))
    matched.sort(key=lambda x: -x[1])
    seen, result = set(), []
    for api, _ in matched[:5]:
        if api['func_name'] not in seen:
            seen.add(api['func_name']); result.append(api)
    return result


def determine_execution_phase(anomaly):
    paths = set()
    for s in anomaly['steps']:
        for m in re.finditer(r'"(/[^"]*)"', s['args']):
            paths.add(m.group(1))
    all_paths = ' '.join(paths)
    if any(k in all_paths for k in ('lib', '.so', 'site-packages', 'lib-dynload',
                                      'ld.so', 'encodings', 'importlib')):
        return "Model Loading Phase"
    return "Model Inference Phase"


def format_syscall(anomaly):
    lines = [f"Operation Type: {anomaly['type']}",
             f"PID: {anomaly['pid']}",
             f"Reconstruction Loss: {anomaly['loss']:.8f} (threshold: {anomaly['threshold']:.8f})",
             f"Number of syscall steps: {len(anomaly['steps'])}", ""]
    for i, s in enumerate(anomaly['steps'], 1):
        lines.append(f"Step {i}: {s['syscall']}({s['args']}) = {s['ret']}")
    return '\n'.join(lines)


def format_full_api_log(api_calls):
    """Format all API entries for the LLM to select from."""
    lines = []
    for i, api in enumerate(api_calls):
        fn = api.get('func_name', 'unknown')
        op = api.get('operation_type', 'general')
        ts = api.get('timestamp', 'N/A')
        args = json.dumps(api.get('args_summary', {}), ensure_ascii=False)
        lines.append(f"[{i+1}] [{op}] {fn} (timestamp: {ts})")
        lines.append(f"    args: {args}")
    return '\n'.join(lines)


def format_app_log(api_context):
    if not api_context: return "(No matching Python API context found)"
    lines = []
    for api in api_context:
        fn = api.get('func_name', 'unknown')
        op = api.get('operation_type', 'general')
        ts = api.get('timestamp', 'N/A')
        args = json.dumps(api.get('args_summary', {}), ensure_ascii=False)
        lines.append(f"- [{op}] {fn} (timestamp: {ts})")
        lines.append(f"  args: {args}")
    return '\n'.join(lines)


STAGE1_PROMPT = """You are a system-call security analyst. Your task is to perform a first-pass triage on a syscall anomaly to determine if it COULD be part of a known attack pattern. You DO NOT have Python API context yet -- only syscall-level evidence.

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
</anomaly_syscall_operation>"""


STAGE2_PROMPT = """You are a cross-layer security analyst. A syscall anomaly was flagged as suspicious in Stage 1. Now you have BOTH the syscall evidence AND the Python API application log. Perform definitive cross-layer reasoning.

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
</application_log>"""


def parse_stage1_response(text):
    ss = re.search(r'<SuspiciousScore>\s*([\d.]+)\s*</SuspiciousScore>', text)
    nc = re.search(r'<NeedCrossLayer>\s*(true|false)\s*</NeedCrossLayer>', text)
    rs = re.search(r'<Stage1Reasoning>\s*(.+?)\s*</Stage1Reasoning>', text, re.DOTALL)
    return {
        'SuspiciousScore': float(ss.group(1)) if ss else 0,
        'NeedCrossLayer': (nc.group(1) == 'true') if nc else False,
        'Stage1Reasoning': rs.group(1).strip() if rs else text.strip()
    }


def parse_stage2_response(text):
    is_ = re.search(r'<IntentScore>\s*([\d.]+)\s*</IntentScore>', text)
    cs = re.search(r'<Category>\s*(\w+)\s*</Category>', text)
    ev = re.search(r'<EvidenceChain>\s*(.+?)\s*</EvidenceChain>', text, re.DOTALL)
    return {
        'IntentScore': float(is_.group(1)) if is_ else -1,
        'Category': cs.group(1) if cs else 'Unknown',
        'EvidenceChain': ev.group(1).strip() if ev else text.strip()
    }


def call_llm(prompt, api_key, base_url, label=""):
    import requests
    print(f"\n{'='*60}\n[LLM {label}] Prompt:\n{prompt}\n{'='*60}")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "deepseek/deepseek-v4-pro", "messages": [
        {"role": "system", "content": "You are a security analyst. Output only the requested XML format exactly."},
        {"role": "user", "content": prompt}
    ], "temperature": 0}

    try:
        resp = requests.post(base_url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            print(f"  [!] HTTP {resp.status_code}: {resp.text[:300]}")
            return None
        data = resp.json()
        if 'error' in data:
            print(f"  [!] API error: {data['error']}")
            return None
        if 'choices' in data and data['choices']:
            return data['choices'][0]['message']['content'].strip()
        print(f"  [!] Unexpected response: {json.dumps(data, ensure_ascii=False)[:500]}")
        return None
    except Exception as e:
        print(f"  [!] LLM call failed: {e}")
        return None


def main():
    anomalies_file = "./result_anomalies.txt"
    api_calls_file = "./test_model/api_calls.log"
    output_file = "./fine_grained_result.txt"
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")

    print("=" * 60)
    print("DDMO Fine-grained Verification (Stage 3)")
    print("=" * 60)

    anomalies, tau_d, tau_s = parse_anomalies(anomalies_file)
    api_calls = parse_api_calls(api_calls_file)

    # Sort by loss descending: most anomalous first
    anomalies.sort(key=lambda a: a['loss'], reverse=True)
    print(f"[*] Processing in descending anomaly score order")

    results = []
    for i, anomaly in enumerate(anomalies):
        phase = determine_execution_phase(anomaly)
        syscall_text = format_syscall(anomaly)

        print(f"\n--- Anomaly {i+1}/{len(anomalies)} (id={anomaly['id']}, loss={anomaly['loss']:.6f}, type={anomaly['type']}) ---")
        s1_prompt = STAGE1_PROMPT.format(execution_phase=phase, anomaly_syscall_operation=syscall_text)
        s1_raw = call_llm(s1_prompt, api_key, base_url, label=f"S1-{anomaly['id']}")
        print(f"[LLM S1-{anomaly['id']} Response]: {s1_raw[:300] if s1_raw else 'None'}")
        s1_result = parse_stage1_response(s1_raw) if s1_raw else {
            'SuspiciousScore': 0, 'NeedCrossLayer': False,
            'Stage1Reasoning': 'LLM call failed'}
        print(f"[LLM S1-{anomaly['id']} Parsed]: Score={s1_result['SuspiciousScore']}, CrossLayer={s1_result['NeedCrossLayer']}")

        need_cross = s1_result.get('NeedCrossLayer', False)

        if need_cross:
            s2_prompt = STAGE2_PROMPT.format(
                execution_phase=phase, anomaly_syscall_operation=syscall_text,
                application_log=format_full_api_log(api_calls))
            s2_raw = call_llm(s2_prompt, api_key, base_url, label=f"S2-{anomaly['id']}")
            print(f"[LLM S2-{anomaly['id']} Response]: {s2_raw[:300] if s2_raw else 'None'}")
            judgment = parse_stage2_response(s2_raw) if s2_raw else {
                'IntentScore': 0, 'Category': 'Benign', 'EvidenceChain': 'LLM Stage 2 failed'}
            print(f"[LLM S2-{anomaly['id']} Parsed]: Score={judgment['IntentScore']}, Cat={judgment['Category']}")
        else:
            judgment = {'IntentScore': s1_result.get('SuspiciousScore', 0),
                        'Category': 'Benign',
                        'EvidenceChain': s1_result.get('Stage1Reasoning', '')}

        results.append({
            'anomaly_id': anomaly['id'], 'type': anomaly['type'],
            'pid': anomaly['pid'], 'syscall': anomaly['syscall'],
            'loss': anomaly['loss'], 'execution_phase': phase,
            'stage1_score': s1_result.get('SuspiciousScore', 0),
            'need_cross': need_cross,
            'stage1_reason': s1_result.get('Stage1Reasoning', ''),
            'IntentScore': judgment.get('IntentScore', -1),
            'Category': judgment.get('Category', 'Unknown'),
            'EvidenceChain': judgment.get('EvidenceChain', ''),
            'anomaly': anomaly
        })

        if (i + 1) % 10 == 0:
            m = sum(1 for r in results if r['Category'] == 'Malicious')
            b = sum(1 for r in results if r['Category'] == 'Benign')
            e = sum(1 for r in results if r['need_cross'])
            print(f"\n  [{i+1}/{len(anomalies)}] S1-escalated: {e}, M={m} B={b}\n")

    with open(output_file, 'w', encoding='utf-8') as out:
        malicious = [r for r in results if r['Category'] == 'Malicious']
        benign = [r for r in results if r['Category'] == 'Benign']
        unknown = [r for r in results if r['Category'] not in ('Malicious', 'Benign')]

        out.write("=" * 80 + "\n")
        out.write("DDMO Fine-grained Verification Results (Two-Stage)\n")
        out.write("=" * 80 + "\n")
        out.write(f"Total anomalies: {len(anomalies)}\n")
        out.write(f"Final: Malicious={len(malicious)}, Benign={len(benign)}, Unknown={len(unknown)}\n")
        out.write(f"{'-'*80}\n\n")

        for r in malicious:
            out.write(f"  [{r['anomaly_id']}] IntentScore={r['IntentScore']} | {r['execution_phase']}\n")
            out.write(f"  {r['syscall']}\n  Evidence: {r['EvidenceChain']}\n")
            for s in r['anomaly']['steps']:
                out.write(f"    -> {s['syscall']}({s['args'][:120]}) = {s['ret']}\n")
            out.write("\n")

        out.write("=" * 80 + "\n")
        out.write(f"Summary | Malicious={len(malicious)} Benign={len(benign)} Unknown={len(unknown)}\n")

    m = len(malicious); b = len(benign)
    print(f"\n[*] Complete: Malicious={m}, Benign={b}")
    print(f"[*] Results: {output_file}")
    return results


if __name__ == "__main__":
    main()
