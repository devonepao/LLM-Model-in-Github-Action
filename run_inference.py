import argparse
import os
import sys
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Lazy import transformers to avoid overhead if not used
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Model Mapping
MODEL_MAPPINGS = {
    "gemma-3-270m": {
        "repo_id": "ggml-org/gemma-3-270m-GGUF",
        "filename": "gemma-3-270m-Q8_0.gguf",
        "backend": "llama-cpp"
    },
    "gemma-3-270m-it": {
        "repo_id": "google/gemma-3-270m-it",
        "backend": "transformers"
    },
    "gemma-3-270m-instruct": {
        "repo_id": "google/gemma-3-270m-it",
        "backend": "transformers"
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
    
    # For transformers, we can rely on from_pretrained to handle download/cache, 
    # but we can pre-fetch if needed. Here we just return the repo_id.
    if config["backend"] == "transformers":
         return config["repo_id"]

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

def run_inference_llama(model_path, query):
    """Runs inference using llama-cpp-python."""
    print("Loading llama-cpp model...")
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=4096, 
            n_threads=2, 
            verbose=False
        )
        
        print(f"Processing query: {query}")
        
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

def run_inference_transformers(model_id, query, hf_token=None):
    """Runs inference using transformers."""
    if not TRANSFORMERS_AVAILABLE:
        print("Error: Transformers library not found. Please install requirements.")
        sys.exit(1)
        
    print(f"Loading transformers model: {model_id}...")
    try:
        hf_token = hf_token or os.getenv("HF_TOKEN")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            token=hf_token,
            torch_dtype=torch.float32,  # explicit float32 for CPU safety
            use_safetensors=True
        )
        
        # Gemma 270M is small, we can run on CPU
        device = "cpu"
        model.to(device)

        inputs = tokenizer(query, return_tensors="pt").to(device)

        # Generation Config: temp 0.7, top_p 0.95, top_k 64, context 32k
        print(f"Processing query with config: temp=0.7, top_p=0.95, top_k=64")
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=64,
        )
        
        # Decode response, skipping the input prompt
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()

    except Exception as e:
        print(f"Transformers inference failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run Gemma 3 270M Inference")
    parser.add_argument("--model", required=True, help="Model key (gemma-3-270m or gemma-3-270m-it)")
    parser.add_argument("--query", required=True, help="Input query text")
    parser.add_argument("--hf_token", help="Hugging Face Token (optional)")
    
    args = parser.parse_args()
    
    if args.model not in MODEL_MAPPINGS:
         print(f"Error: Unknown model '{args.model}'")
         sys.exit(1)

    backend = MODEL_MAPPINGS[args.model].get("backend", "llama-cpp")
    
    model_path_or_id = download_model(args.model, args.hf_token)
    
    if backend == "transformers":
        response = run_inference_transformers(model_path_or_id, args.query, args.hf_token)
    else:
        response = run_inference_llama(model_path_or_id, args.query)
    
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
