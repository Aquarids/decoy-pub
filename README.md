## Installation

```bash
conda env create -f environment.yml
conda activate decoy
cp .env.example .env
```

Set the local and cloud model URLs, keys, and model names in `.env`. Both endpoints must support the OpenAI Chat Completions API and function calling.

## Run

```bash
python exp/smoke.py
python exp/smoke.py --query "Your request"
```
