#!/usr/bin/env python3
"""
Demo script to test web navigation functionality.
This creates a simple test without requiring the full model.
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web_navigator import WebNavigator, parse_navigation_prompt
from pdf_generator import create_navigation_report

def demo_parse_prompt():
    """Demonstrate prompt parsing."""
    print("=" * 70)
    print("DEMO: Prompt Parsing")
    print("=" * 70)
    
    test_prompts = [
        "navigate to https://harshityadav.in open first article and summarize first article",
        "visit https://example.com and fill form",
        "Explain quantum computing",
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        is_nav, details = parse_navigation_prompt(prompt)
        print(f"Is Navigation: {is_nav}")
        if details:
            print(f"Details: {details}")
        print()

def demo_web_navigation():
    """Demonstrate web navigation (limited test)."""
    print("=" * 70)
    print("DEMO: Web Navigation")
    print("=" * 70)
    
    print("\nNavigating to example.com...")
    
    try:
        with WebNavigator(headless=True) as navigator:
            # Navigate to a simple page
            step1 = navigator.navigate("https://example.com")
            print(f"Step 1: {step1}")
            
            # Get page text
            text = navigator.get_page_text()
            print(f"\nPage text (first 200 chars): {text[:200]}...")
            
            # Get steps and screenshots
            steps = navigator.get_steps()
            screenshots = navigator.get_screenshots()
            
            print(f"\nTotal steps recorded: {len(steps)}")
            print(f"Total screenshots captured: {len(screenshots)}")
            
            # Create a demo PDF
            print("\nGenerating demo PDF report...")
            pdf_file = create_navigation_report(
                steps=steps,
                screenshots=screenshots,
                summary="This is a demo navigation to example.com. The page was successfully loaded and a screenshot was captured.",
                original_prompt="navigate to https://example.com",
                output_file="demo_report.pdf"
            )
            print(f"PDF report created: {pdf_file}")
            
    except Exception as e:
        print(f"Error during navigation: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run demo."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "Web Navigation Demo" + " " * 29 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    demo_parse_prompt()
    print()
    demo_web_navigation()
    
    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()
