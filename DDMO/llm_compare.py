import importlib.util
import json
import requests

# Load the DDMO fine-grained detection module (for its prompts + parsers).
spec = importlib.util.spec_from_file_location("fgd", "fine_grained_detect.py")
fgd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fgd)

API_KEY = "sk-vobH4xue91jIgSnOSySmW2dvHbFY62HKxQwcs3v6IZibGdEM"
ENDPOINT = "http://api.pkuoslab.com:3000/v1/chat/completions"
MODELS = ["deepseek-v4-pro", "qwen3.6-plus", "glm-5.2"]

ANOMALIES_FILE = "./result_anomalies.txt"
API_CALLS_FILE = "./test_model/api_calls.log"


def call_llm(model, prompt):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a security analyst. Output only the requested XML format exactly."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    r = requests.post(
        ENDPOINT,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    if r.status_code != 200:
        return None
    data = r.json()
    if "choices" in data and data["choices"]:
        return data["choices"][0]["message"]["content"].strip()
    return None


def run_model(model):
    anomalies, tau_d, tau_s = fgd.parse_anomalies(ANOMALIES_FILE)
    api_calls = fgd.parse_api_calls(API_CALLS_FILE)
    anomalies.sort(key=lambda a: a["loss"], reverse=True)

    results = []
    for i, anomaly in enumerate(anomalies):
        phase = fgd.determine_execution_phase(anomaly)
        syscall_text = fgd.format_syscall(anomaly)

        s1_prompt = fgd.STAGE1_PROMPT.format(
            execution_phase=phase, anomaly_syscall_operation=syscall_text
        )
        s1_raw = call_llm(model, s1_prompt)
        s1 = fgd.parse_stage1_response(s1_raw) if s1_raw else {
            "SuspiciousScore": 0, "NeedCrossLayer": False, "Stage1Reasoning": "LLM call failed"}

        need_cross = s1.get("NeedCrossLayer", False)
        if need_cross:
            s2_prompt = fgd.STAGE2_PROMPT.format(
                execution_phase=phase,
                anomaly_syscall_operation=syscall_text,
                application_log=fgd.format_full_api_log(api_calls),
            )
            s2_raw = call_llm(model, s2_prompt)
            judgment = fgd.parse_stage2_response(s2_raw) if s2_raw else {
                "IntentScore": 0, "Category": "Benign", "EvidenceChain": "LLM Stage 2 failed"}
        else:
            judgment = {
                "IntentScore": s1.get("SuspiciousScore", 0),
                "Category": "Benign",
                "EvidenceChain": s1.get("Stage1Reasoning", ""),
            }

        results.append({
            "anomaly_id": anomaly["id"],
            "type": anomaly["type"],
            "syscall": anomaly["syscall"],
            "loss": round(anomaly["loss"], 6),
            "stage1_score": s1.get("SuspiciousScore", 0),
            "need_cross": need_cross,
            "IntentScore": judgment.get("IntentScore", -1),
            "Category": judgment.get("Category", "Unknown"),
            "EvidenceChain": judgment.get("EvidenceChain", "")[:400],
        })
        print(f"[{model}] {i+1}/{len(anomalies)} id={anomaly['id']} "
              f"S1={s1.get('SuspiciousScore',0)} cross={need_cross} -> {judgment.get('Category')}")

    malicious = [r for r in results if r["Category"] == "Malicious"]
    benign = [r for r in results if r["Category"] == "Benign"]
    unknown = [r for r in results if r["Category"] not in ("Malicious", "Benign")]
    return {
        "model": model,
        "total": len(results),
        "escalated_to_stage2": sum(1 for r in results if r["need_cross"]),
        "malicious": len(malicious),
        "benign": len(benign),
        "unknown": len(unknown),
        "malicious_anomalies": [
            (r["anomaly_id"], r["IntentScore"], r["syscall"]) for r in malicious
        ],
        "results": results,
    }


if __name__ == "__main__":
    all_out = []
    for m in MODELS:
        print(f"\n===== Running model: {m} =====")
        out = run_model(m)
        all_out.append(out)
        print(f"===== [{m}] malicious={out['malicious']} benign={out['benign']} "
              f"unknown={out['unknown']} escalated={out['escalated_to_stage2']}")
        with open(f"./llm_result_{m}.json", "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n\n========== SUMMARY ==========")
    for out in all_out:
        print(f"{out['model']:20s} malicious={out['malicious']:2d} benign={out['benign']:2d} "
              f"unknown={out['unknown']:2d} | malicious_ids={out['malicious_anomalies']}")
    json.dump(all_out, open("./llm_compare_summary.json", "w"), ensure_ascii=False, indent=2)
    print("saved llm_compare_summary.json")