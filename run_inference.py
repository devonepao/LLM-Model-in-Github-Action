import argparse
import os
import sys
from typing import Optional, List, Any
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Lazy import transformers to avoid overhead if not used
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Import web navigation and PDF generation modules
try:
    from web_navigator import WebNavigator, parse_navigation_prompt, chunk_text
    from pdf_generator import create_navigation_report
    WEB_NAVIGATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Web navigation not available: {e}")
    WEB_NAVIGATION_AVAILABLE = False

# LangChain imports
try:
    try:
        from langchain_core.language_models import LLM
    except ImportError:
        from langchain.llms.base import LLM
        
    try:
        from langchain.chains.summarize import load_summarize_chain
    except ImportError:
        from langchain.chains import load_summarize_chain
        
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
    from langchain.docstore.document import Document
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LangChain not available: {e}")
    LANGCHAIN_AVAILABLE = False

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
        
        # Force CPU-only execution to match CPU-only PyTorch installation
        # This keeps download size minimal (~200MB vs 2GB+ for CUDA version)
        device = "cpu"
        model.to(device)

        inputs = tokenizer(query, return_tensors="pt").to(device)

        # Generation Config: temp 0.7, top_p 0.95, top_k 64, context 32k
        print(f"Processing query with config: temp=0.7, top_p=0.95, top_k=64")
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1024,
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


# Define GemmaLangChainLLM only when LangChain is available
if LANGCHAIN_AVAILABLE:
    from typing import ClassVar 
    
    class GemmaLangChainLLM(LLM):
        """Custom LangChain LLM wrapper for Gemma model."""
        
        # Model configuration constants
        MAX_INPUT_TOKENS: ClassVar[int] = 6000  # Leave room for output within 32K context
        MAX_OUTPUT_TOKENS: ClassVar[int] = 512
        
        model_id: str
        tokenizer: Any = None
        model: Any = None
        hf_token: Optional[str] = None
        
        def __init__(self, model_id: str, hf_token: Optional[str] = None, max_input_tokens: int = None):
            super().__init__()
            self.model_id = model_id
            self.hf_token = hf_token or os.getenv("HF_TOKEN")
            if max_input_tokens is not None:
                self.MAX_INPUT_TOKENS = max_input_tokens
            self._load_model()
        
        def _load_model(self):
            """Load the model and tokenizer."""
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers library not available")
            
            print(f"Loading model for LangChain: {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.hf_token)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                token=self.hf_token,
                torch_dtype=torch.float32,  # explicit float32 for CPU safety
                use_safetensors=True
            )
            # Force CPU-only execution to match CPU-only PyTorch installation
            self.model.to("cpu")
        
        @property
        def _llm_type(self) -> str:
            return "gemma"
        
        def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
            """Run inference on the prompt."""
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cpu")
            
            # Limit input to avoid context overflow
            if inputs.input_ids.shape[1] > self.MAX_INPUT_TOKENS:
                inputs.input_ids = inputs.input_ids[:, -self.MAX_INPUT_TOKENS:]
                inputs.attention_mask = inputs.attention_mask[:, -self.MAX_INPUT_TOKENS:]
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.MAX_OUTPUT_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=64,
            )
            
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            return response.strip()


def summarize_with_langchain(text: str, model_id: str, hf_token: Optional[str] = None) -> str:
    """Summarize text using LangChain and Gemma model."""
    if not LANGCHAIN_AVAILABLE:
        print("Warning: LangChain not available, using simple truncation")
        return text[:2000] + "..." if len(text) > 2000 else text
    
    try:
        # Create LLM wrapper
        llm = GemmaLangChainLLM(model_id=model_id, hf_token=hf_token)
        
        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Create documents
        texts = text_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts]
        
        print(f"Summarizing {len(docs)} chunks of text...")
        
        # Use map_reduce chain for long documents
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)
        
        return summary
        
    except Exception as e:
        print(f"LangChain summarization failed: {e}")
        # Fallback to simple truncation
        return text[:2000] + "..." if len(text) > 2000 else text


def handle_web_navigation(query: str, model_key: str, hf_token: Optional[str] = None) -> str:
    """Handle web navigation tasks with Playwright and summarization."""
    if not WEB_NAVIGATION_AVAILABLE:
        print("Error: Web navigation dependencies not available")
        return "Error: Web navigation not available. Please install playwright and related dependencies."
    
    # Parse the navigation prompt
    is_nav, task_details = parse_navigation_prompt(query)
    
    if not is_nav or not task_details.get("urls"):
        return "Error: Could not parse navigation task. Please include a URL to navigate to."
    
    print("Starting web navigation task...")
    print(f"Task details: {task_details}")
    
    # Get model config
    if model_key not in MODEL_MAPPINGS:
        model_key = "gemma-3-270m-it"  # Default to instruct model
    
    model_config = MODEL_MAPPINGS[model_key]
    model_id = model_config.get("repo_id") if model_config.get("backend") == "transformers" else None
    
    # Perform web navigation
    with WebNavigator(headless=True) as navigator:
        # Navigate to the URL
        url = task_details["urls"][0]
        navigator.navigate(url)
        
        # Execute actions
        content_to_summarize = ""
        for action in task_details.get("actions", []):
            if action["type"] == "open_first_article":
                navigator.find_and_click_first_article()
                # Get article content
                content_to_summarize = navigator.get_page_text()
            elif action["type"] == "fill_form":
                # This would need form data from the prompt
                pass
        
        # Get all steps and screenshots
        steps = navigator.get_steps()
        screenshots = navigator.get_screenshots()
    
    # Summarize content if needed
    summary = ""
    needs_summary = any(a["type"] == "summarize" for a in task_details.get("actions", []))
    
    if needs_summary and content_to_summarize:
        print("Generating summary...")
        if model_id and LANGCHAIN_AVAILABLE:
            # Use LangChain for better summarization
            summary = summarize_with_langchain(content_to_summarize, model_id, hf_token)
        else:
            # Fallback to simple inference
            chunks = chunk_text(content_to_summarize, max_chars=6000)
            if chunks:
                summary_prompt = f"Summarize the following article:\n\n{chunks[0]}"
                if model_id:
                    summary = run_inference_transformers(model_id, summary_prompt, hf_token)
                else:
                    summary = "Summary generation not available for this model."
    
    # Generate PDF report
    print("Generating PDF report...")
    pdf_filename = "navigation_report.pdf"
    try:
        create_navigation_report(
            steps=steps,
            screenshots=screenshots,
            summary=summary,
            original_prompt=query,
            output_file=pdf_filename
        )
        print(f"PDF report created: {pdf_filename}")
    except Exception as e:
        print(f"Error creating PDF: {e}")
    
    # Create response
    response_parts = []
    response_parts.append("Web Navigation Task Completed")
    response_parts.append(f"\nSteps executed: {len(steps)}")
    response_parts.append(f"Screenshots captured: {len(screenshots)}")
    
    if summary:
        response_parts.append(f"\n\nSummary:\n{summary}")
    
    response_parts.append(f"\n\nDetailed report saved to: {pdf_filename}")
    
    return "\n".join(response_parts)

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
    
    # Check if this is a web navigation task
    is_navigation, _ = parse_navigation_prompt(args.query) if WEB_NAVIGATION_AVAILABLE else (False, None)
    
    if is_navigation:
        # Handle web navigation with Playwright
        response = handle_web_navigation(args.query, args.model, args.hf_token)
    else:
        # Regular inference
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
