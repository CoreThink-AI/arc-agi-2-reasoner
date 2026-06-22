# ARC-AGI 2 Reasoner

A rule-based solution for the ARC-AGI-2 benchmark. This project implements a neurosymbolic approach to solving ARC tasks using pattern recognition and logical reasoning.

## Overview

The ARC-AGI 2 Reasoner is designed to tackle the Abstraction and Reasoning Corpus (ARC) challenge, which tests an AI system's ability to identify patterns and apply abstract reasoning to solve visual tasks. This implementation focuses on:

- Pattern recognition and analysis
- Rule-based reasoning
- Systematic task decomposition
- Visual pattern matching

## Features

- Modular architecture for pattern recognition and solving
- Integration with modern AI models (OpenAI, Anthropic) for enhanced reasoning
- Comprehensive testing framework
- Extensible pattern library
- Visual debugging and analysis tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CoreThink-AI/arc-agi-2-reasoner.git
cd arc-agi-2-reasoner
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

For development, install with additional dependencies:
```bash
pip install -e ".[dev]"
```

4. Create hints and outputs directory for your run
```bash
mkdir outputs
mkdir hints
```

## Environment Setup

1. Create `.env` in the following manner:
```bash
XAI_API_KEY_FLOW_2=<1 xAI key>
XAI_API_KEY=<1 xAI key>
XAI_API_KEYS=<6 xAI keys separated by commas>
OPENAI_API_KEY=<1 OpenAI key>
GROQ_API_KEY=<1 Groq API key>
GROQ_API_KEYS=<3 Groq API keys>
TOGETHER_API_KEY=<1 Together API key>
PYTHONPATH=f"{os.getcwd()}:{os.environ.get('PYTHONPATH', '')}"
```

## Project Structure

```
arc-agi-2-reasoner/
├── arc_agi/
│   ├── __init__.py
│   ├── core.py
│   ├── core_clean.py
│   └── src/
│       ├── solver/           # Core solving logic
│       ├── patterns/         # Pattern recognition modules
│       ├── patterns_intersection/  # Pattern intersection logic
│       ├── objects/          # Object definitions and utilities
│       ├── utils/            # Utility functions
│       ├── low_hanging/      # Low-hanging fruit solutions
│       └── low_hanging_fruits/ # Additional low-hanging fruit modules
├── kiva/                     # KiVA benchmark extension (isolated)
│   ├── __init__.py
│   ├── loader.py             # Dataset loader for KiVA tasks
│   ├── llm_client.py         # LLM client (Gemini / OpenRouter / XAI)
│   ├── pattern_detector.py   # Phase 2: transformation detection
│   ├── solver.py             # Phase 3: 3-stage MCQ solver
│   ├── eval.py               # Phase 4: full evaluation harness
│   └── dataset_setup.py      # Dataset health checker
├── kiva_results/             # Output directory (auto-created on first run)
│   ├── results.csv           # Full per-task results
│   └── hints/                # Cached pattern detection hints (JSON)
├── data/                     # Training and test data (JSON files)
├── testing/
│   ├── unit/                # Unit tests
│   ├── samples/             # Sample test cases
│   ├── e2e.py               # End-to-end testing
│   ├── e2e_api.py           # API end-to-end testing
│   ├── e2e_challenges.py    # Challenge end-to-end testing
│   └── e2e_challenges_parallel.py # Parallel challenge testing
├── e2e_logs/                # End-to-end test logs and visualizations
├── arc_agi.egg-info/        # Package metadata
├── requirements.txt
├── setup.py
├── env_template.txt
├── experiments.ipynb        # Jupyter notebook for experiments
├── debug_test.py            # Debug testing script
├── target.txt               # Target data file
├── results.json             # Results data
├── arc-agi_test_challenges.json # Test challenges data
├── easy.json                # Easy test cases
├── grids_side_by_side.png   # Visualization image
├── LICENSE
├── .gitignore
└── README.md
```

## Usage

1. Generate hints for the solver
```bash
python -u testing/e2e_challenges_parallel.py --batch_size 10 --phase 1 
```

2. Generate solved outputs using the hints
```bash
python -u testing/e2e_challenges_parallel.py --batch_size 5 --phase 2
```

The solved responses are stored in the `outputs/` directory. The test entries are taken from `arc-agi-test_challenges.json`.



---

## KiVA Benchmark

This repository also includes a neurosymbolic evaluation pipeline for the **KiVA (Kid-inspired Visual Analogies)** benchmark. The KiVA extension is completely isolated from the ARC pipeline — importing or running it has no effect on the ARC solver.

KiVA tests 5 visual transformation concepts across 14 subdomains (700 tasks total in KiVA-easy):

| Concept | Parameters | Tasks |
|---------|-----------|-------|
| Colour | Red, Green, Blue | 150 |
| Counting | +1, −1, +2, −2 | 200 |
| Resize | 0.5XY, 2XY, 0.5X, 0.5Y | 200 |
| Reflect | X, Y | 100 |
| 2DRotation | +90, −90, +180 | 150 |

### Step 1 — Clone the KiVA dataset

```bash
git clone https://github.com/ey242/kiva "path/to/kiva"
```

Then set the path in your `.env`:

```
KIVA_DATA_DIR=path/to/kiva
```

### Step 2 — Install KiVA dependencies

```bash
pip install google-genai
```

The KiVA pipeline uses the Google Gemini SDK. All other dependencies are already covered by the base `requirements.txt`.

### Step 3 — Configure your `.env`

Add the following to your existing `.env` file:

```bash
# KiVA dataset path
KIVA_DATA_DIR=path/to/kiva

# Gemini API key (from aistudio.google.com)
GEMINI_API_KEY=AIzaSy...

# Model to use (Gemini 2.5 Flash recommended)
KIVA_VISION_MODEL=gemini-2.5-flash
KIVA_TEXT_MODEL=gemini-2.5-flash

# Optional tuning (defaults shown)
KIVA_DETECTION_REPETITIONS=3     # Vision calls per pattern detection
KIVA_SOLVER_ATTEMPTS=3           # Consensus attempts per MCQ stage
KIVA_DETECTION_CONCURRENCY=3     # Max concurrent LLM calls
```

**Alternative: OpenRouter** (if you have an OpenRouter key instead of a direct Gemini key):

```bash
OPENROUTER_API_KEY=sk-or-...
KIVA_VISION_MODEL=openai/gpt-4o-mini
KIVA_TEXT_MODEL=openai/gpt-4o-mini
```

**Alternative: XAI / Grok**:

```bash
XAI_API_KEY=xai-...
KIVA_VISION_MODEL=grok-2-vision-1212
KIVA_TEXT_MODEL=grok-2-vision-1212
```

### Step 4 — Verify the dataset loads correctly

```bash
python -m kiva.dataset_setup
```

You should see all 14 concept-parameter directories confirmed with PNG counts.

### Step 5 — Smoke test (6 tasks, ~1 minute)

Before running the full benchmark, verify the pipeline is working end-to-end:

```bash
python -m kiva.eval --concepts Colour --trials 2 --attempts 1
```

Expected output: a summary table with non-zero accuracy across Phase2, Cross, Within, and Extrap columns. If all predictions are empty strings, check your API key and model name.

### Step 6 — Full benchmark (700 tasks)

```bash
python -m kiva.eval --attempts 3
```

This runs all 5 concepts, all 50 trials per subdomain, with 3 consensus attempts per solver stage. Estimated runtime: **1–3 hours** depending on API rate limits.

Results are written incrementally to `kiva_results/results.csv`. The run is **crash-safe** — if interrupted, re-run the same command and it resumes from where it stopped.

### Step 7 — View results

The accuracy summary is printed automatically at the end of the run. To reprint from an existing CSV without re-running:

```bash
python -m kiva.eval --summary-only
```

To run a subset of concepts:

```bash
python -m kiva.eval --concepts Colour Counting --attempts 3
```

### KiVA CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--concepts` | all 5 | Space-separated list: `Colour Counting Resize Reflect 2DRotation` |
| `--trials` | 50 | Max trials per concept-parameter subdomain |
| `--attempts` | 3 | Consensus attempts per solver stage |
| `--output-dir` | `kiva_results/` | Directory for CSV and hint cache |
| `--no-resume` | off | Re-run all tasks even if results exist |
| `--kiva-root` | `KIVA_DATA_DIR` | Override KiVA dataset path |
| `--summary-only` | off | Print summary from existing CSV, skip evaluation |

### Output Format

`kiva_results/results.csv` contains one row per task with 19 columns:

| Column | Description |
|--------|-------------|
| `task_id` | e.g. `Colour_Red_0` |
| `concept` / `parameter` / `trial` | Task metadata |
| `hint_concept` / `hint_param` / `hint_confidence` | Phase 2 pattern detection output |
| `hint_concept_correct` / `hint_param_correct` | Phase 2 accuracy flags |
| `cross_predicted` / `cross_gt` / `cross_correct` | Stage 1 MCQ result |
| `within_predicted` / `within_gt` / `within_correct` | Stage 2 MCQ result |
| `extrap_correct` / `extrap_confidence` | Stage 3 MCQ result |
| `all_correct` | True only if all 3 stages correct |
| `elapsed_s` | Wall-clock time for this task |

### Clearing cached results

To delete all cached hints and results and start fresh:

```bash
# Windows (PowerShell)
Remove-Item -Recurse -Force kiva_results

# macOS / Linux
rm -rf kiva_results/
```

---

## License

This project is proprietary software. All rights reserved. See the LICENSE file for details.

The software and its documentation are confidential and protected by copyright laws and international treaties. Unauthorized copying, modification, distribution, or use of this software, via any medium, is strictly prohibited.

For licensing inquiries, please contact the CoreThink AI Team.

## Requirements

- Python >= 3.11
- See [requirements.txt](requirements.txt) for full dependency list

## Acknowledgments
- The ARC-AGI-2 benchmark creators
- The open-source community for various tools and libraries used in this project
