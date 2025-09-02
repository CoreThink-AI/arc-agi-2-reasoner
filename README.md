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
