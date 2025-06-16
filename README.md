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
git clone https://github.com/yourusername/arc-agi-2-reasoner.git
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

## Environment Setup

1. Copy the environment template:
```bash
cp env_template.txt .env
```

2. Edit `.env` and add your API keys for:
- OpenAI
- Anthropic

## Project Structure

```
arc-agi-2-reasoner/
├── arc_agi/
│   ├── solver/     # Core solving logic
│   ├── patterns/   # Pattern recognition modules
│   └── objects/    # Object definitions and utilities
├── data/           # Training and test data
├── testing/        # Test cases and utilities
└── requirements.txt
```

## Usage

[Usage examples will be added as the project develops]

## Development

The project uses several development tools:
- `pytest` for testing
- `black` for code formatting
- `flake8` for linting
- `mypy` for type checking

Run tests:
```bash
pytest
```

Format code:
```bash
black .
```

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
