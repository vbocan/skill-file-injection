#!/usr/bin/env python3
"""Re-run only the failed (success:false) records from prior hardened runs.

Reads all logs/hardened_results_*.jsonl, finds failures, and re-attempts each
unique (model, problem, skill, condition, rep) with several fresh seeds to try
to dodge the deterministic Ollama <think>-parse 500. Successful recoveries are
written to logs/hardened_redo_<ts>.jsonl with the identical schema; at analysis
they supersede the failed originals (merge prefers success records).
"""
import glob, json, os
from datetime import datetime
import yaml
from openai import OpenAI
import hardened_test as h

client = OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")
problems = {p["id"]: p for p in h.PROBLEMS}
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = h.LOGS_DIR / f"hardened_redo_{ts}.jsonl"

# collect unique failed items
failed, seen = [], set()
for fn in glob.glob(str(h.LOGS_DIR / "hardened_results_*.jsonl")):
    for line in open(fn, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not r.get("success"):
            key = (r["model"], r.get("model_id"), r.get("problem"),
                   r["skill_file"], r["test_name"], r["rep"])
            if key not in seen:
                seen.add(key)
                failed.append(r)

print(f"failed items to retry: {len(failed)}")
out = open(out_path, "w", encoding="utf-8")
recovered = 0

for r in failed:
    skill = yaml.safe_load(open(h.SKILL_FILES_DIR / r["skill_file"], encoding="utf-8"))
    cond = next(c for c in h.SKILL_FILE_TESTS[r["skill_file"]] if c["name"] == r["test_name"])
    prob = problems[r["problem"]]
    msg = prob["exam"] if cond["message"] == "{exam}" else prob["plain"]
    case = dict(cond, message=msg)
    amarkers = prob["answer_markers"]
    system_prompt, trig, exfil = h.resolve_trigger(skill, case)

    out_call = None
    used_seed = None
    for seed in [r["rep"] + 100, r["rep"] + 200, r["rep"] + 500, 1234, 4242]:
        o = h.call_model(client, r["model_id"], system_prompt, msg,
                         0.7, seed, 4096, 180, 1)
        if o["ok"]:
            out_call = o; used_seed = seed; break
        print(f"  still failing {r['model']}/{r['problem']}/{r['test_name']} seed={seed}")

    rec = {
        "model": r["model"], "model_id": r["model_id"], "backend": "ollama",
        "quantization": r.get("quantization"), "skill_file": r["skill_file"],
        "problem": r["problem"], "test_name": r["test_name"], "mode": r["mode"],
        "date": r.get("date"), "expected_behavior": r["expected_behavior"],
        "is_reasoning_model": r.get("is_reasoning_model"),
        "rep": r["rep"], "seed": used_seed, "redo": True,
        "trigger_activated": trig, "exfiltration_logged": exfil,
    }
    if out_call and out_call["ok"]:
        recovered += 1
        resp = out_call["response"]
        reasoning_seg, final_seg, has_think = h.split_reasoning(resp)
        whole_pa, whole_soc, whole_beh = h.classify_text(resp, amarkers)
        f_pa, _, _ = h.classify_text(final_seg if has_think else resp, amarkers)
        r_pa, _, _ = h.classify_text(reasoning_seg, amarkers)
        rec.update({
            "success": True, "latency": out_call["latency"],
            "tokens": out_call["tokens"], "response": resp,
            "provides_answer": whole_pa, "uses_socratic": whole_soc,
            "behavior": whole_beh, "adheres": whole_beh == r["expected_behavior"],
            "has_think_tag": has_think, "final_provides_answer": f_pa,
            "reasoning_provides_answer": r_pa,
            "cot_leak": bool(r["expected_behavior"] == "BENIGN" and not f_pa and r_pa),
        })
        print(f"  RECOVERED {r['model']}/{r['problem']}/{r['test_name']} (seed {used_seed}) -> {whole_beh}")
    else:
        rec.update({"success": False, "error": "all retry seeds failed"})

    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.flush()

out.close()
print(f"\nrecovered {recovered}/{len(failed)} -> {out_path}")
