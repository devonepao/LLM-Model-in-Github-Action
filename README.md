# LLM Model in Github Action

This repository contains a GitHub Action to run the **Google Gemma 3 270M** model for inference directly within GitHub Actions runners.

## Features

- **Model**: Runs Gemma 3 270M (Quantized GGUF).
- **Efficiency**: Uses `llama-cpp-python` for CPU-based inference optimized for GitHub runners.
- **Caching**: Automatically downloads and caches model weights from Hugging Face to speed up subsequent runs.
- **Inputs**: Custom text query and model selection.

## Usage

You can run this action via the **Actions** tab in your repository by selecting the "Run Gemma 3 270M" workflow.

### Workflow Dispatch Inputs

- **Model**: Select `gemma-3-270m` or `gemma-3-270m-instruct`.
- **Query**: Enter your prompt or question.

### Example Workflow

```yaml
name: Run Gemma 3 270M

on:
  workflow_dispatch:
    inputs:
      model:
        default: 'gemma-3-270m'
      query:
        default: 'Hello!'

jobs:
  inference:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ./
        with:
          model: ${{ inputs.model }}
          query: ${{ inputs.query }}
```

## How it works

1.  **Composite Action**: The logic is defined in `action.yml`.
2.  **Dependencies**: Installs `llama-cpp-python` and `huggingface_hub`.
3.  **Caching**: Uses `actions/cache` to store the Hugging Face cache directory (`hf_cache`) located in the workspace.
4.  **Inference**: `run_inference.py` loads the model and generates a response.
