# 🤖 Run Gemma 3 (270M) in GitHub Actions

![GitHub Actions](https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

A powerful, efficient GitHub Action to run **Google's Gemma 3 270M** model directly in your CI/CD pipelines. Optimized for standard GitHub-hosted runners using `llama.cpp` and `uv` for lightning-fast inference.

## ✨ New: Web Navigation & Automation

Now supports **advanced web navigation** using Playwright with Chrome! Perform complex web automation tasks like:
- 🌐 Navigate to websites and extract content
- 📄 Open and read articles automatically
- 📝 Fill forms and interact with web pages
- 🤖 Summarize web content using Gemma 3 270M-IT
- 📊 Generate PDF reports with screenshots and summaries
- 🔄 Handle complex multi-step navigation workflows
- 💾 Record and save all steps for reproducibility

Perfect for automating web scraping, content summarization, and testing workflows!

## 🚀 Capabilities

- **Zero-Config Inference**: Runs the ultra-compact Gemma 3 270M model out of the box.
- **Lightning Fast**: Built on `uv` for instant environment setup and `llama-cpp-python` for optimized CPU inference.
- **Smart Caching**: Automatically downloads and caches model weights (~300MB) using GitHub Actions Cache, making subsequent runs instant.
- **Web Automation**: Navigate websites, extract content, and interact with web pages using Playwright.
- **AI-Powered Summarization**: Uses LangChain and Gemma 3 270M-IT to summarize web content intelligently.
- **PDF Reports**: Generate comprehensive reports with screenshots and summaries.
- **Context Window Management**: Automatically handles text chunking to stay within Gemma's 32K token limit.
- **Secure**: Supports gated models via `HF_TOKEN` integration.
- **Cross-Platform**: Tested on both Ubuntu (AMD64) and ARM64 architecture runners.

## 🛠️ Usage

### Quick Start - Basic Inference

Add this step to your workflow:

```yaml
- name: Run Gemma 3 Inference
  uses: harshityadav95/LLM-Model-in-Github-Action@main
  with:
    model: 'gemma-3-270m'
    query: 'Explain quantum computing in one sentence.'
    hf_token: ${{ secrets.HF_TOKEN }} # Required for gated models
```

### Web Navigation & Automation

Navigate websites, extract content, and generate summaries:

```yaml
- name: Web Navigation with AI
  uses: harshityadav95/LLM-Model-in-Github-Action@main
  with:
    model: 'gemma-3-270m-instruct'
    query: 'navigate to https://harshityadav.in open first article and summarize first article'
    hf_token: ${{ secrets.HF_TOKEN }}

- name: Upload Navigation Report
  uses: actions/upload-artifact@v4
  with:
    name: navigation-report
    path: navigation_report.pdf
```

### Full Workflow Example

Create `.github/workflows/ai-inference.yml`:

```yaml
name: AI Model Inference

on:
  workflow_dispatch:
    inputs:
      query:
        description: 'Input text for the model'
        required: true
        default: 'Hello world!'

jobs:
  run-model:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Gemma
        id: gemma
        uses: harshityadav95/LLM-Model-in-Github-Action@main
        with:
          model: 'gemma-3-270m-instruct'
          query: ${{ inputs.query }}
          hf_token: ${{ secrets.HF_TOKEN }}

      - name: View Result
        run: echo "Response: ${{ steps.gemma.outputs.response }}"
```

## 🏗️ Technical Architecture

### Web Navigation Pipeline

The action uses a sophisticated pipeline for web automation:

1. **Prompt Parsing**: Automatically detects navigation tasks and extracts URLs, actions, and intent
2. **Browser Automation**: Uses Playwright with Chromium for reliable, headless web browsing
3. **Content Extraction**: Intelligently extracts visible text from web pages, filtering out scripts/styles
4. **Text Chunking**: Splits content into manageable chunks (<8000 chars) to stay within Gemma's context window
5. **AI Summarization**: 
   - Uses LangChain with map-reduce strategy for long documents
   - Powered by Gemma 3 270M-IT for accurate, context-aware summaries
6. **Screenshot Capture**: Captures full-page screenshots at each step
7. **Step Recording**: Tracks all actions with timestamps and metadata
8. **PDF Generation**: Creates professional reports using ReportLab with screenshots and summaries

### Key Components

- **web_navigator.py**: Playwright-based browser automation with step recording
- **pdf_generator.py**: PDF report generation with images and formatted text
- **run_inference.py**: Main orchestrator integrating LLM, web navigation, and report generation
- **LangChain Integration**: Custom LLM wrapper for Gemma 3 270M-IT with automatic context management

## ⚙️ Configuration

| Input | Description | Required | Default |
|-------|-------------|----------|---------|
| `model` | Model variant to use. Options: `gemma-3-270m`, `gemma-3-270m-instruct`. | ✅ | `gemma-3-270m` |
| `query` | The text prompt to send to the model. For web navigation, use keywords like "navigate", "open", "summarize". | ✅ | - |
| `hf_token` | Hugging Face Access Token for downloading gated models. | ❌ | - |

| Output | Description |
|--------|-------------|
| `response` | The text generated by the model. |

### Web Navigation Prompt Format

The action automatically detects web navigation tasks. Use natural language with these keywords:

- **Navigation**: `navigate`, `visit`, `go to`, `browse`
- **Actions**: `open first article`, `fill form`, `click`
- **Processing**: `summarize`, `extract`, `read`

**Examples:**
- `"navigate to https://example.com open first article and summarize first article"`
- `"visit https://harshityadav.in and extract the main content"`
- `"go to https://mysite.com fill form with data and submit"`

### Output Files

- **response.txt**: Text response from the model
- **navigation_report.pdf**: Comprehensive PDF report with screenshots and summaries (for web navigation tasks only)

## 🔮 Future Scope & Roadmap

We plan to expand this action's capabilities:

- [x] **Web Navigation**: Automated web browsing with Playwright + Chrome
- [x] **AI Summarization**: LangChain integration for intelligent content summarization
- [x] **PDF Reports**: Generate comprehensive reports with screenshots
- [x] **Context Management**: Automatic text chunking for 32K token limit
- [ ] **More Models**: Support for Gemma 1B/2B and other SLMs (TinyLlama, Phi-3).
- [ ] **Custom Models**: Allow users to provide any Hugging Face GGUF repo ID.
- [ ] **GPU Acceleration**: Optimize for self-hosted runners with GPU support.
- [ ] **Chat History**: Support multi-turn conversations for context-aware CI bots.
- [ ] **JSON Output**: structured output mode for programmatic usage in pipelines.
- [ ] **Advanced Form Filling**: Support complex form interactions with dynamic data.
- [ ] **Multi-page Navigation**: Handle complex workflows across multiple pages.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## 👥 Contributors

- **Harshit Yadav** - *Initial Work* - [@harshityadav95](https://github.com/harshityadav95)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Built with ❤️ using [llama.cpp](https://github.com/abetlen/llama-cpp-python), [uv](https://github.com/astral-sh/uv), and Google Gemma.*
