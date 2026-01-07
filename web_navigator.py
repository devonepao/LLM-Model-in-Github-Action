"""
Web navigation module using Playwright for complex web automation tasks.
Supports navigation, form filling, screenshots, and task recording.
"""
import os
import re
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from playwright.sync_api import sync_playwright, Page, Browser
from PIL import Image
from io import BytesIO


class WebNavigator:
    """Handles web navigation and automation using Playwright."""
    
    # Configuration constants
    DEFAULT_TIMEOUT_MS = 30000
    DEFAULT_NETWORK_IDLE_TIMEOUT_MS = 10000
    
    def __init__(self, headless: bool = True, timeout_ms: int = None):
        self.headless = headless
        self.timeout_ms = timeout_ms or self.DEFAULT_TIMEOUT_MS
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.playwright = None
        self.steps: List[Dict] = []
        self.screenshots: List[bytes] = []
        
    def __enter__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.page = self.browser.new_page()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.page:
            self.page.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
    
    def navigate(self, url: str) -> Dict:
        """Navigate to a URL and record the step."""
        print(f"Navigating to: {url}")
        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            self.page.wait_for_load_state("networkidle", timeout=self.DEFAULT_NETWORK_IDLE_TIMEOUT_MS)
            
            step = {
                "action": "navigate",
                "url": url,
                "title": self.page.title(),
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            self.steps.append(step)
            
            # Take screenshot
            screenshot_bytes = self.page.screenshot(full_page=True)
            self.screenshots.append(screenshot_bytes)
            
            return step
        except Exception as e:
            step = {
                "action": "navigate",
                "url": url,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
            self.steps.append(step)
            return step
    
    def get_page_text(self) -> str:
        """Extract visible text from the current page."""
        try:
            # Get text content from body, excluding script and style tags
            text = self.page.evaluate("""() => {
                const body = document.body;
                const walker = document.createTreeWalker(
                    body,
                    NodeFilter.SHOW_TEXT,
                    {
                        acceptNode: function(node) {
                            const parent = node.parentElement;
                            if (parent.tagName === 'SCRIPT' || parent.tagName === 'STYLE') {
                                return NodeFilter.FILTER_REJECT;
                            }
                            if (node.textContent.trim().length === 0) {
                                return NodeFilter.FILTER_REJECT;
                            }
                            return NodeFilter.FILTER_ACCEPT;
                        }
                    }
                );
                
                let text = '';
                let node;
                while (node = walker.nextNode()) {
                    text += node.textContent + ' ';
                }
                return text;
            }""")
            return text.strip()
        except Exception as e:
            print(f"Error extracting page text: {e}")
            return ""
    
    def find_and_click_first_article(self) -> Dict:
        """Find and click the first article link on the page."""
        print("Looking for first article...")
        try:
            # Try different selectors for articles
            selectors = [
                "article a[href]",
                ".article a[href]",
                ".post a[href]",
                "a[href*='article']",
                "a[href*='post']",
                "main a[href]",
                ".content a[href]"
            ]
            
            clicked = False
            for selector in selectors:
                elements = self.page.query_selector_all(selector)
                if elements:
                    # Get the first valid link
                    for elem in elements:
                        href = elem.get_attribute("href")
                        if href and not href.startswith("#") and not href.startswith("javascript:"):
                            print(f"Clicking first article: {href}")
                            elem.click()
                            self.page.wait_for_load_state("domcontentloaded", timeout=10000)
                            clicked = True
                            break
                if clicked:
                    break
            
            if not clicked:
                # Fallback: find any article-like element
                all_links = self.page.query_selector_all("a[href]")
                for link in all_links[:10]:  # Check first 10 links
                    text = link.inner_text().lower()
                    if any(word in text for word in ["read", "article", "post", "blog", "more"]):
                        href = link.get_attribute("href")
                        if href and not href.startswith("#"):
                            print(f"Clicking article link: {href}")
                            link.click()
                            self.page.wait_for_load_state("domcontentloaded", timeout=10000)
                            clicked = True
                            break
            
            step = {
                "action": "click_first_article",
                "url": self.page.url,
                "title": self.page.title(),
                "timestamp": datetime.now().isoformat(),
                "success": clicked
            }
            self.steps.append(step)
            
            # Take screenshot after clicking
            if clicked:
                screenshot_bytes = self.page.screenshot(full_page=True)
                self.screenshots.append(screenshot_bytes)
            
            return step
            
        except Exception as e:
            step = {
                "action": "click_first_article",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
            self.steps.append(step)
            return step
    
    def fill_form(self, form_data: Dict[str, str]) -> Dict:
        """Fill a form with provided data."""
        print(f"Filling form with data: {form_data}")
        try:
            for field_name, value in form_data.items():
                # Try different ways to find the input field
                selectors = [
                    f"input[name='{field_name}']",
                    f"input[id='{field_name}']",
                    f"textarea[name='{field_name}']",
                    f"textarea[id='{field_name}']",
                ]
                
                filled = False
                for selector in selectors:
                    element = self.page.query_selector(selector)
                    if element:
                        element.fill(value)
                        filled = True
                        break
                
                if not filled:
                    print(f"Warning: Could not find field '{field_name}'")
            
            step = {
                "action": "fill_form",
                "form_data": form_data,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            self.steps.append(step)
            
            # Take screenshot after filling
            screenshot_bytes = self.page.screenshot(full_page=True)
            self.screenshots.append(screenshot_bytes)
            
            return step
            
        except Exception as e:
            step = {
                "action": "fill_form",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
            self.steps.append(step)
            return step
    
    def click_element(self, selector: str) -> Dict:
        """Click an element by selector."""
        print(f"Clicking element: {selector}")
        try:
            self.page.click(selector)
            self.page.wait_for_load_state("domcontentloaded", timeout=10000)
            
            step = {
                "action": "click",
                "selector": selector,
                "url": self.page.url,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            self.steps.append(step)
            
            # Take screenshot after clicking
            screenshot_bytes = self.page.screenshot(full_page=True)
            self.screenshots.append(screenshot_bytes)
            
            return step
            
        except Exception as e:
            step = {
                "action": "click",
                "selector": selector,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
            self.steps.append(step)
            return step
    
    def get_steps(self) -> List[Dict]:
        """Return all recorded steps."""
        return self.steps
    
    def get_screenshots(self) -> List[bytes]:
        """Return all captured screenshots."""
        return self.screenshots


def parse_navigation_prompt(prompt: str) -> Tuple[bool, Optional[Dict]]:
    """
    Parse a prompt to determine if it's a navigation task.
    Returns (is_navigation, task_details)
    """
    prompt_lower = prompt.lower()
    
    # Check if it's a navigation prompt
    navigation_keywords = ["navigate", "open", "visit", "go to", "browse"]
    is_navigation = any(keyword in prompt_lower for keyword in navigation_keywords)
    
    if not is_navigation:
        return False, None
    
    # Extract URL
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, prompt)
    
    # Extract actions
    actions = []
    if "first article" in prompt_lower or "open article" in prompt_lower:
        actions.append({"type": "open_first_article"})
    
    if "summarize" in prompt_lower or "summarise" in prompt_lower:
        actions.append({"type": "summarize"})
    
    if "fill form" in prompt_lower:
        actions.append({"type": "fill_form"})
    
    task_details = {
        "urls": urls,
        "actions": actions,
        "original_prompt": prompt
    }
    
    return True, task_details


def chunk_text(text: str, max_chars: int = 8000) -> List[str]:
    """
    Split text into chunks that fit within the context window.
    
    This function implements a hierarchical chunking strategy:
    1. First tries to split by paragraphs (\\n\\n)
    2. If paragraphs are too long, splits by sentences (. )
    3. If sentences are too long, splits by words (spaces)
    
    Args:
        text: The text to chunk
        max_chars: Maximum characters per chunk (default 8000).
                  This is sized to fit within Gemma's 32K token context window,
                  accounting for ~4 chars per token and leaving room for prompts.
    
    Returns:
        List of text chunks, each <= max_chars in length
    
    Edge cases handled:
    - Text shorter than max_chars: returned as-is
    - Single very long paragraph: split recursively by sentences then words
    - Empty text or whitespace: returns list with single empty/whitespace chunk
    
    Note: The relationship between chars and tokens is approximate (~4:1 ratio).
    Actual token count may vary based on the tokenizer.
    """
    # If text is short enough, return as-is
    if len(text) <= max_chars:
        return [text]
    
    # Split by paragraphs first
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If a single paragraph is too long, split it further
        if len(para) > max_chars:
            # If current chunk has content, save it
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split long paragraph by sentences
            sentences = para.split(". ")
            for sentence in sentences:
                # If a single sentence is still too long, split by words
                if len(sentence) > max_chars:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    
                    words = sentence.split()
                    for word in words:
                        if len(current_chunk) + len(word) + 1 <= max_chars:
                            current_chunk += word + " "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = word + " "
                elif len(current_chunk) + len(sentence) + 2 <= max_chars:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "
        elif len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text[:max_chars]]
