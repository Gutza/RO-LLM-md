Environment Setup for OpenAI Finetuning
=======================================

This project fine-tunes a small OpenAI model to fix Romanian markdown. After researching multiple platforms, `gpt-4.1-nano` emerged as the most cost‑effective choice. We therefore commit to OpenAI’s APIs and tooling while keeping the actual model name configurable, since future migrations might use a different OpenAI model.

## Requirements
* **Python 3.10+**
* **openai** Python package and CLI

Install dependencies with:
```bash
pip install openai
```

Export your API key so both the library and CLI can authenticate:
```bash
export OPENAI_API_KEY="sk-..."
```

Optionally define the base model in an environment variable to keep it flexible:
```bash
export OPENAI_BASE_MODEL="gpt-4.1-nano"
```

## Finetuning Steps
1. **Prepare the dataset** in JSONL format with `messages` objects for each training pair.
2. **Upload the data** using the CLI:
   ```bash
   openai files.create -p fine-tune -f dataset.jsonl
   ```
3. **Create the job**, referencing the uploaded file and the base model:
   ```bash
   openai fine_tuning.jobs.create -t FILE_ID -m "$OPENAI_BASE_MODEL"
   ```
4. **Retrieve the model name** from the job output and use it for inference with the Python library.

Once these steps complete, you can call the fine-tuned model in your Python code:
```python
from openai import OpenAI
client = OpenAI()
completion = client.chat.completions.create(
    model="ft:gpt-4.1-nano...",  # your fine-tuned model
    messages=[...]
)
```
