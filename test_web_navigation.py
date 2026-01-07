#!/usr/bin/env python3
"""
Test script for web navigation functionality.
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web_navigator import parse_navigation_prompt, chunk_text

def test_parse_navigation_prompt():
    """Test parsing navigation prompts."""
    print("Testing parse_navigation_prompt...")
    
    # Test 1: Navigation with article opening
    prompt1 = "navigate to https://harshityadav.in open first article and summarize first article"
    is_nav, details = parse_navigation_prompt(prompt1)
    assert is_nav == True, "Should detect navigation"
    assert len(details["urls"]) == 1, "Should extract one URL"
    assert details["urls"][0] == "https://harshityadav.in", "Should extract correct URL"
    assert any(a["type"] == "open_first_article" for a in details["actions"]), "Should detect open article action"
    assert any(a["type"] == "summarize" for a in details["actions"]), "Should detect summarize action"
    print("✓ Test 1 passed: Navigation with article opening")
    
    # Test 2: Non-navigation prompt
    prompt2 = "What is the capital of France?"
    is_nav, details = parse_navigation_prompt(prompt2)
    assert is_nav == False, "Should not detect navigation"
    print("✓ Test 2 passed: Non-navigation prompt")
    
    # Test 3: Multiple URLs
    prompt3 = "visit https://example.com and then go to https://test.com"
    is_nav, details = parse_navigation_prompt(prompt3)
    assert is_nav == True, "Should detect navigation"
    assert len(details["urls"]) >= 2, "Should extract multiple URLs"
    print("✓ Test 3 passed: Multiple URLs")
    
    print("All parse_navigation_prompt tests passed!\n")

def test_chunk_text():
    """Test text chunking."""
    print("Testing chunk_text...")
    
    # Test 1: Short text
    short_text = "This is a short text."
    chunks = chunk_text(short_text, max_chars=100)
    assert len(chunks) == 1, "Should have one chunk for short text"
    print("✓ Test 1 passed: Short text")
    
    # Test 2: Long text with paragraphs
    long_text = "Paragraph 1.\n\n" * 100
    chunks = chunk_text(long_text, max_chars=500)
    assert len(chunks) > 1, "Should have multiple chunks for long text"
    for chunk in chunks:
        assert len(chunk) <= 600, f"Chunk too long: {len(chunk)} chars"  # Allow some margin
    print("✓ Test 2 passed: Long text with paragraphs")
    
    # Test 3: Very long single paragraph
    very_long = "Word " * 5000
    chunks = chunk_text(very_long, max_chars=1000)
    assert len(chunks) > 1, "Should split very long paragraphs"
    print("✓ Test 3 passed: Very long single paragraph")
    
    print("All chunk_text tests passed!\n")

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import playwright
        print("✓ Playwright imported successfully")
    except ImportError as e:
        print(f"✗ Playwright import failed: {e}")
        return False
    
    try:
        from reportlab.lib.pagesizes import letter
        print("✓ ReportLab imported successfully")
    except ImportError as e:
        print(f"✗ ReportLab import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow imported successfully")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
    
    try:
        import web_navigator
        print("✓ web_navigator module imported successfully")
    except ImportError as e:
        print(f"✗ web_navigator import failed: {e}")
        return False
    
    try:
        import pdf_generator
        print("✓ pdf_generator module imported successfully")
    except ImportError as e:
        print(f"✗ pdf_generator import failed: {e}")
        return False
    
    print("All imports successful!\n")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Web Navigation Tests")
    print("=" * 60 + "\n")
    
    if not test_imports():
        print("\n✗ Import tests failed. Please install dependencies.")
        return 1
    
    try:
        test_parse_navigation_prompt()
        test_chunk_text()
        
        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
