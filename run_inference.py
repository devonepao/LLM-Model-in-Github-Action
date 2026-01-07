import argparse
import os
import sys
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Model Mapping
# Using quantized models for efficiency. 
# bartowski provides reliable GGUF quants. 
# Gemma 3 270M is very small, so Q4_K_M or Q8_0 are good choices. 
# Using Q8_0 for maximum quality since it's still tiny (~300MB).
MODEL_MAPPINGS = {
    "gemma-3-270m": {
        "repo_id": "ggml-org/gemma-3-270m-GGUF",
        "filename": "gemma-3-270m-Q8_0.gguf"
    },
    "gemma-3-270m-instruct": {
        "repo_id": "unsloth/gemma-3-270m-it-GGUF",
        "filename": "gemma-3-270m-it-Q8_0.gguf"
    }
}

def download_model(model_key, hf_token=None):
    """Downloads the model from Hugging Face Hub if not cached."""
    # Prioritize environment variable if passed token is empty
    hf_token = hf_token or os.getenv("HF_TOKEN")
    cache_dir = os.getenv("HF_HOME")
    
    if model_key not in MODEL_MAPPINGS:
        print(f"Error: Unknown model '{model_key}'. Available: {list(MODEL_MAPPINGS.keys())}")
        sys.exit(1)
        
    config = MODEL_MAPPINGS[model_key]
    print(f"Downloading {config['filename']} from {config['repo_id']}...")
    
    try:
        model_path = hf_hub_download(
            repo_id=config['repo_id'],
            filename=config['filename'],
            token=hf_token,
            cache_dir=cache_dir
        )
        print(f"Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Failed to download model: {e}")
        sys.exit(1)

def run_inference(model_path, query):
    """Runs inference using llama-cpp-python."""
    print("Loading model...")
    try:
        # Initialize Llama model
        # n_ctx=2048 is usually sufficient for simple queries, Gemma supports more.
        llm = Llama(
            model_path=model_path,
            n_ctx=4096, 
            n_threads=2, # Conservative for GitHub runners
            verbose=False
        )
        
        print(f"Processing query: {query}")
        
        # Simple completion
        # For instruct models, we should ideally use chat formatting, but a raw prompt works for simple testing.
        # Gemma 3 Instruct uses standard chat templates, but for simplicity we'll just pass the query.
        output = llm(
            query,
            max_tokens=256,
            stop=["<eos>", "<end_of_turn>"],
            echo=False
        )
        
        text = output['choices'][0]['text']
        return text.strip()
        
    except Exception as e:
        print(f"Inference failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run Gemma 3 270M Inference")
    parser.add_argument("--model", required=True, help="Model key (gemma-3-270m or gemma-3-270m-instruct)")
    parser.add_argument("--query", required=True, help="Input query text")
    parser.add_argument("--hf_token", help="Hugging Face Token (optional)")
    
    args = parser.parse_args()
    
    model_path = download_model(args.model, args.hf_token)
    response = run_inference(model_path, args.query)
    
    print("\n--- RESPONSE ---")
    print(response)
    print("----------------")
    
    # Save to file
    with open("response.txt", "w", encoding="utf-8") as f:
        f.write(response)
        
    # Set GitHub Output
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            # Escape newlines for GitHub Actions output
            clean_response = response.replace("\n", "%0A")
            f.write(f"response={clean_response}\n")

if __name__ == "__main__":
    main()
