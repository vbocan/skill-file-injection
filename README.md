# Adversarial Skill File Injection in LLM-Based Intelligent Tutoring Systems

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Runtime: Ollama](https://img.shields.io/badge/runtime-Ollama-black)
![LLMs tested](https://img.shields.io/badge/LLMs%20tested-14-success)
![Tutoring tasks](https://img.shields.io/badge/tutoring%20tasks-3-blueviolet)
![Generations](https://img.shields.io/badge/generations-3%2C990-orange)
![Trigger accuracy](https://img.shields.io/badge/trigger%20activation-100%25-brightgreen)
![Paper](https://img.shields.io/badge/paper-Acta%20Polytechnica%20Hungarica-informational)
![Status](https://img.shields.io/badge/artifact-reproducible-success)

> Reproducible research artifact for the paper **“Adversarial Skill File Injection in LLM-Based Intelligent Tutoring Systems: Formal Threat Modeling and Multi-Architecture Validation”** (V. Bocan, V. E. Balas), under review at *Acta Polytechnica Hungarica*.

A **skill file** (the YAML/JSON document that defines an LLM tutor’s system prompt, behavioural rules, and tools) sits at maximum influence yet is usually treated as inert configuration rather than executable code. This repository demonstrates that a malicious skill file can turn a benign-looking tutor into one that **leaks exam answers**, **activates only under exam-like context or specific calendar dates**, and **silently exfiltrates student data** — and that the vulnerability lives in the *skill-file abstraction*, not in any one model. It contains the full test harness, the six proof-of-concept skill files, and the **consolidated results** behind every number in the paper.

> ⚠️ **Responsible-use notice.** This is defensive security research. The exfiltration “endpoint” is simulated and logs locally; no real student data is collected or transmitted. The skill files are intentionally malicious examples for studying detection and mitigation. Do not deploy them against real systems.

---

## Key findings

- **Attack mechanism is architecture-independent.** Across **14 LLMs × 3 tutoring tasks × 19 conditions × 5 repetitions (3,990 generations)**, the skill-file trigger logic activated in **100 % (2,100/2,100)** of expected cases and silent exfiltration captured data in **100 % (630/630)** of sessions — by construction, independent of the model.
- **What varies is the model’s *fidelity* to the skill file’s pedagogy.** Measured as **whole-response adherence** (does the visible output respect the “stay Socratic” instruction where required?):

| Group | Mean adherence | SD | Range |
|------|:--:|:--:|:--:|
| **Standard instruction-tuned** (7 models) | **88.7 %** | 5.8 | 77.5–96.1 % |
| **Reasoning** (7 models) | **63.5 %** | 12.9 | 52.6–83.9 % |

- **Model category is a weak safety predictor.** The best reasoning models (Qwen3 14B/8B ≈ 80–84 %) overlap the standard range, while four reasoning models sit near 52–53 % (they disclose the solution in essentially every benign context). Adherence **does not track size** — DeepSeek-R1 **8B (69.8 %) > 14B (53.3 %)**.
- **Reasoning traces are an unreliable hiding place.** One model (Phi-4 Reasoning) maintained a Socratic closing answer while its *visible reasoning* contained the full solution; whether that reasoning was wrapped in collapsible `<think>` tags **changed between otherwise-identical runs** — so a deployment cannot assume a model’s working stays hidden.

Reproduce all of the above with `python harness/analyze_results.py`.

---

## Repository structure

```
skill-file-injection/
├── run_experiment.ps1          # one-shot orchestrator: pulls models + runs the full matrix
├── harness/
│   ├── hardened_test.py        # unified, backend-agnostic test harness
│   ├── redo_failed.py          # re-runs any failed generations with fresh seeds
│   └── analyze_results.py      # reproduces the paper's numbers from the consolidated data
├── skill_files/                # the 6 proof-of-concept skill files (1 benign control + 5 attacks)
└── results/
    ├── results_consolidated.json   # all 3,990 records (one file, full responses)
    └── summary.json                # per-model aggregates + provenance
```

## The skill files

| File | Class | Behaviour |
|------|-------|-----------|
| `benign_tutor.yaml` | control | Proper Socratic tutoring; never reveals solutions |
| `direct_exploit.yaml` | direct | Always reveals the full solution |
| `stealth_tutor.yaml` | stealth | Reveals only when it detects exam context (keyword **or** `assessment_mode` flag) |
| `temporal_tutor.yaml` | temporal | Reveals only during exam periods (calendar-date triggered) |
| `exfiltration_tutor.yaml` | exfiltration | Behaves normally while silently harvesting student data |
| `combined_stealth_tutor.yaml` | combined | All triggers + exfiltration only during high-value (exam) sessions |

---

## Requirements

- **Python 3.10+** with `pip install -r requirements.txt` (`openai`, `pyyaml`)
- **[Ollama](https://ollama.com)** running locally (default `http://localhost:11434`). LM Studio is also supported via `--backend lmstudio`.
- A GPU is recommended. The reference run used an **NVIDIA RTX 5080 (16 GB)** at 4-bit quantization; the model set is chosen to fit 16 GB VRAM.

## Quick start

**Verify the published results (no GPU needed):**
```bash
pip install -r requirements.txt
python harness/analyze_results.py
```

**Reproduce the full experiment** (Windows PowerShell; pulls ~14 models, then runs):
```powershell
.\run_experiment.ps1            # pulls models, runs 14 × 3 tasks × 19 × 5 reps
.\run_experiment.ps1 -Resume    # resume after an interruption (skips completed models)
```
Or drive the harness directly on any OpenAI-compatible endpoint:
```bash
python harness/hardened_test.py --backend ollama --reps 5 --temperature 0.7
```
Outputs are written to `logs/` (per-run JSONL). If any generation fails due to a backend parsing error, recover it with:
```bash
python harness/redo_failed.py
```

Every published figure is recomputable from `results/results_consolidated.json` with the definitions in `harness/analyze_results.py` — same backend + same model tags + same seeds reproduce the same results (subject to the run-to-run sampling variance reported as ± SD).

---

## Methodology in one paragraph

Each of six skill files is exercised under conditions spanning practice/assessment modes, exam keywords, the `assessment_mode` flag, and simulated calendar dates, for three distinct calculus tasks (two product-rule derivatives and one integration-by-parts problem), five times per model at temperature 0.7 (max 4,096 tokens). A response is scored **adherent** when its *visible output* falls in the behaviour class the skill file intends for that context (Socratic guidance vs. full-solution disclosure), classified by a transparent keyword detector. We classify the **whole visible response** because the security property at stake is answer confidentiality: if the solution appears anywhere the student can see it — including a model’s exposed reasoning — the property is violated regardless of how the closing sentence is phrased.

## Citation

```bibtex
@article{bocan2026skillfile,
  title   = {Adversarial Skill File Injection in LLM-Based Intelligent Tutoring
             Systems: Formal Threat Modeling and Multi-Architecture Validation},
  author  = {Bocan, Valer and Balas, Valentina E.},
  journal = {Acta Polytechnica Hungarica},
  year    = {2026},
  note    = {Under review}
}
```

## License

Released under the [MIT License](LICENSE).

## Authors

- **Valer Bocan**, Department of Computer Science, Politehnica University Timișoara, Romania
- **Valentina E. Balas**, “Aurel Vlaicu” University of Arad, Romania
