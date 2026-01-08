import argparse
import os
import json
import traceback
from typing import Dict, Any
from web_navigator import WebNavigator
try:
    from run_inference import GemmaLangChainLLM, MODEL_MAPPINGS
except ImportError:
    # Fallback for testing/missing dependencies
    class GemmaLangChainLLM:
        def __init__(self, **kwargs): pass
        def _call(self, prompt): return ""
    MODEL_MAPPINGS = {}

def parse_llm_json(response: str) -> Dict[str, Any]:
    """Extract JSON from LLM response."""
    try:
        # Find JSON block
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end != -1:
            json_str = response[start:end]
            return json.loads(json_str)
        return {}
    except Exception:
        return {}

def main():
    parser = argparse.ArgumentParser(description="Web Agent")
    parser.add_argument("--url", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--model", default="gemma-3-270m-it")
    args = parser.parse_args()

    print(f"Starting Web Agent...")
    print(f"URL: {args.url}")
    print(f"Instruction: {args.instruction}")

    # Setup Model
    hf_token = os.getenv("HF_TOKEN")
    model_config = MODEL_MAPPINGS.get(args.model, MODEL_MAPPINGS["gemma-3-270m-it"])
    model_id = model_config.get("repo_id")
    
    try:
        llm = GemmaLangChainLLM(model_id=model_id, hf_token=hf_token)
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Fallback or exit? For now, we need the model.
        return

    with WebNavigator(headless=True) as navigator:
        try:
            # 1. Start at URL
            navigator.navigate(args.url)
            
            # Agent Loop
            max_steps = 10
            for i in range(max_steps):
                print(f"\n--- Step {i+1} ---")
                
                # Get State
                elements_text = navigator.get_interactive_elements()
                current_url = navigator.page.url
                
                # Construct Prompt
                prompt = f"""
You are a web navigation agent. 
Goal: {args.instruction}
Current URL: {current_url}

Interactive Elements:
{elements_text}

Respond in JSON format ONLY:
{{
    "reasoning": "I need to click...",
    "action": "click" | "type" | "finish",
    "element_id": "id from list" (or null if finish),
    "value": "text to type" (optional),
    "summary": "final summary" (only if action is finish)
}}
"""
                print("Thinking...")
                # Reduce prompt size if needed
                response = llm._call(prompt[:5000]) # Direct call to control generation
                print(f"LLM Response: {response}")
                
                decision = parse_llm_json(response)
                
                if not decision:
                    print("Could not parse decision. Retrying or stopping.")
                    break
                    
                action_type = decision.get("action")
                reasoning = decision.get("reasoning")
                print(f"Reasoning: {reasoning}")
                
                if action_type == "finish":
                    print(f"Finished! Summary: {decision.get('summary')}")
                    # Save summary
                    with open("agent_summary.txt", "w") as f:
                        f.write(decision.get('summary', 'No summary provided.'))
                    break
                
                element_id = decision.get("element_id")
                value = decision.get("value")
                
                if element_id:
                    result = navigator.execute_action(element_id, action_type, value)
                    if not result["success"]:
                        print(f"Action failed: {result.get('error')}")
                else:
                    print("No element ID provided for action.")

            # Generate Report
            from pdf_generator import create_navigation_report
            create_navigation_report(
                steps=navigator.get_steps(),
                screenshots=navigator.get_screenshots(),
                summary=open("agent_summary.txt").read() if os.path.exists("agent_summary.txt") else "Agent finished.",
                original_prompt=args.instruction,
                output_file="agent_report.pdf"
            )
            print("Report generated: agent_report.pdf")

        except Exception as e:
            traceback.print_exc()
            print(f"Agent Error: {e}")

if __name__ == "__main__":
    main()
