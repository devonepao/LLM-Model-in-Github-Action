# Web Navigation Implementation Guide

## Overview

This implementation adds advanced web navigation capabilities to the Gemma 3 270M GitHub Action using Playwright MCP with Chrome, LangChain for AI summarization, and comprehensive PDF reporting.

## Architecture

### Components

1. **web_navigator.py** - Core web automation module
   - Playwright-based browser automation
   - Navigation, clicking, form filling
   - Screenshot capture
   - Step recording
   - Prompt parsing

2. **pdf_generator.py** - Report generation
   - PDF creation with ReportLab
   - Screenshot embedding
   - Formatted text sections
   - Metadata tracking

3. **run_inference.py** - Main orchestrator
   - Integration of web navigation and LLM
   - LangChain wrapper for Gemma
   - Text chunking and summarization
   - Mode detection (navigation vs. inference)

## How It Works

### 1. Prompt Detection

The system automatically detects navigation tasks by looking for keywords:

```python
# Navigation keywords: navigate, visit, go to, browse, open
prompt = "navigate to https://example.com open first article and summarize"
is_navigation, details = parse_navigation_prompt(prompt)
# is_navigation = True
# details = {
#     "urls": ["https://example.com"],
#     "actions": [
#         {"type": "open_first_article"},
#         {"type": "summarize"}
#     ]
# }
```

### 2. Web Navigation

When a navigation task is detected:

```python
with WebNavigator(headless=True, timeout_ms=30000) as navigator:
    # Navigate to URL
    navigator.navigate(url)
    
    # Execute actions
    navigator.find_and_click_first_article()
    
    # Extract content
    content = navigator.get_page_text()
    
    # Get recorded steps and screenshots
    steps = navigator.get_steps()
    screenshots = navigator.get_screenshots()
```

### 3. Content Processing

Large content is automatically chunked to fit within the 32K token context window:

```python
# Split content into manageable chunks
chunks = chunk_text(content, max_chars=8000)

# Chunks are sized considering:
# - ~4 characters per token
# - Need room for prompt and response
# - Total context window of 32K tokens
```

### 4. AI Summarization

Content is summarized using Gemma 3 270M-IT with LangChain:

```python
# Create custom LangChain LLM wrapper
llm = GemmaLangChainLLM(model_id="google/gemma-3-270m-it")

# Split text for processing
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=200
)

# Use map-reduce strategy for long documents
chain = load_summarize_chain(llm, chain_type="map_reduce")
summary = chain.run(docs)
```

### 5. PDF Report Generation

A comprehensive report is created with:
- Title and metadata
- Timestamped steps
- Screenshots at each step
- AI-generated summary

```python
create_navigation_report(
    steps=steps,
    screenshots=screenshots,
    summary=summary,
    original_prompt=prompt,
    output_file="navigation_report.pdf"
)
```

## Configuration

### WebNavigator Options

```python
navigator = WebNavigator(
    headless=True,              # Run browser in headless mode
    timeout_ms=30000           # Page load timeout (default 30s)
)
```

### GemmaLangChainLLM Options

```python
llm = GemmaLangChainLLM(
    model_id="google/gemma-3-270m-it",
    hf_token=hf_token,
    max_input_tokens=6000      # Max tokens for input (leaves room for output)
)
```

### Text Chunking

```python
chunks = chunk_text(
    text,
    max_chars=8000             # Max characters per chunk (~2000 tokens)
)
```

## Supported Actions

### Navigation Actions
- `navigate` - Go to a URL
- `open first article` - Find and click the first article link
- `fill form` - Fill form fields (requires form data)
- `click` - Click specific elements

### Processing Actions
- `summarize` - Generate AI summary of content
- `extract` - Extract page content

## Example Prompts

### Basic Navigation
```
navigate to https://example.com
```

### Article Summarization
```
navigate to https://harshityadav.in open first article and summarize first article
```

### Complex Workflow
```
visit https://mysite.com fill form with data and submit
```

## Output Files

### response.txt
Plain text response with:
- Task completion status
- Number of steps executed
- Number of screenshots captured
- Summary (if requested)
- Path to PDF report

### navigation_report.pdf
Comprehensive PDF including:
- Execution metadata
- All navigation steps with timestamps
- Full-page screenshots at each step
- AI-generated summary
- Professional formatting

## Error Handling

The system handles various error scenarios:

1. **Navigation Failures**: Captured in step metadata
2. **Content Extraction Errors**: Gracefully degraded
3. **PDF Generation Errors**: Logged but doesn't fail the task
4. **Model Loading Errors**: Falls back to simpler methods
5. **Timeout Errors**: Configurable timeouts prevent hanging

## Performance Considerations

### Context Window Management
- Text is automatically chunked to stay within limits
- Each chunk is ~8000 chars (~2000 tokens)
- Leaves room for prompts and responses

### Browser Performance
- Headless mode for efficiency
- Network idle detection for complete page loads
- Configurable timeouts

### Model Efficiency
- CPU-optimized for GitHub runners
- 270M parameter model for fast inference
- Smart caching of model weights

## Testing

Run the test suite:
```bash
python3 test_web_navigation.py
```

Run the demo:
```bash
python3 demo_navigation.py
```

## Limitations

1. **Network Access**: Requires internet connectivity for web navigation
2. **JavaScript Sites**: Some complex JS-heavy sites may not work perfectly
3. **Authentication**: No built-in support for login flows yet
4. **Rate Limiting**: No automatic retry logic for rate-limited sites
5. **Context Size**: Very large articles may be truncated to fit context window

## Future Enhancements

- [ ] Support for authenticated sessions
- [ ] Multi-page navigation workflows
- [ ] Form data extraction from prompts
- [ ] Retry logic with exponential backoff
- [ ] JavaScript interaction support
- [ ] Cookie management
- [ ] Local storage manipulation
- [ ] File download support

## Troubleshooting

### DNS Resolution Errors
If you see `ERR_NAME_NOT_RESOLVED`, check:
- Network connectivity
- DNS configuration
- Firewall rules

### Timeout Errors
Increase timeout values:
```python
navigator = WebNavigator(timeout_ms=60000)  # 60 seconds
```

### Memory Issues
For large pages:
- Reduce screenshot quality
- Limit content extraction
- Increase available memory

### Model Loading Failures
Check:
- HF_TOKEN is set correctly
- Model access permissions
- Available disk space for cache

## Security Considerations

1. **HTML Escaping**: Uses `html.escape()` for safe PDF generation
2. **Input Validation**: URLs and selectors are validated
3. **Sandboxing**: Playwright runs in sandboxed Chrome
4. **No Credentials**: Never stores authentication tokens
5. **Temporary Files**: Screenshots kept in memory

## Support

For issues or questions:
1. Check the test suite: `python3 test_web_navigation.py`
2. Run the demo: `python3 demo_navigation.py`
3. Review GitHub Actions logs for detailed error messages
4. Check PDF report for step-by-step execution details
