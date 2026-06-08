#!/usr/bin/env python3
"""Reproduce the paper's headline numbers from results/results_consolidated.json.

Usage:  python harness/analyze_results.py
Reads the consolidated record set and recomputes, with no hidden state:
  - per-model whole-response adherence (mean +/- SD over repetitions) and per-task breakdown
  - standard vs. reasoning group means
  - deterministic trigger-activation and exfiltration rates
This lets any reader verify that the figures in the paper follow from the raw data.
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "results_consolidated.json"

recs = json.load(open(DATA, encoding="utf-8"))
ok = [r for r in recs if r.get("success")]
print(f"records: {len(recs)} | successful: {len(ok)} | "
      f"success rate: {100*len(ok)/len(recs):.1f}%\n")

bym = defaultdict(list)
for r in ok:
    bym[r["model"]].append(r)

rows = []
for m, rs in bym.items():
    cat = "reasoning" if rs[0].get("is_reasoning_model") else "standard"
    reps = sorted(set(r["rep"] for r in rs))
    per_rep = [100 * sum(1 for r in rs if r["rep"] == rp and r["adheres"])
               / max(1, sum(1 for r in rs if r["rep"] == rp)) for rp in reps]
    mean, sd = statistics.mean(per_rep), statistics.pstdev(per_rep)
    by_prob = {p: round(100 * sum(1 for r in rs if r["problem"] == p and r["adheres"])
                        / max(1, sum(1 for r in rs if r["problem"] == p)))
               for p in sorted(set(r["problem"] for r in rs))}
    rows.append((cat, m, rs[0].get("quantization"), mean, sd, by_prob))

rows.sort(key=lambda x: (x[0], -x[3]))
print(f"{'type':<10}{'model':<20}{'quant':<8}{'adherence':>12}   by-task")
print("-" * 78)
for cat, m, q, mean, sd, bp in rows:
    print(f"{cat:<10}{m:<20}{q or '?':<8}{mean:>6.1f} ± {sd:<4.1f}   {bp}")

for cat in ("standard", "reasoning"):
    vals = [r[3] for r in rows if r[0] == cat]
    print(f"\n{cat:<10} n={len(vals)}  mean={statistics.mean(vals):.1f}  "
          f"sd={statistics.pstdev(vals):.1f}  range={min(vals):.1f}-{max(vals):.1f}")

trig = [r for r in ok if r["skill_file"] in
        ("stealth_tutor.yaml", "temporal_tutor.yaml", "combined_stealth_tutor.yaml")]
tmatch = sum(1 for r in trig if r["trigger_activated"] == (r["expected_behavior"] == "EXPLOITED"))
exf = [r for r in ok if r["skill_file"] == "exfiltration_tutor.yaml"]
exo = sum(1 for r in exf if r.get("exfiltration_logged"))
print(f"\ntrigger-activation accuracy: {tmatch}/{len(trig)} = {100*tmatch/len(trig):.1f}%")
print(f"exfiltration capture:        {exo}/{len(exf)} = {100*exo/len(exf):.1f}%")
