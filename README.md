---
sdk: gradio
sdk_version: 5.36.2
python_version: "3.10"
app_file: app.py
---

# Gen AI Agent Demo

A **ReAct** (Reasoning + Acting) agent with dynamic tool use, powered by multi-provider LLM backends. Try it live: [zaid646/genai-agent-demo](https://huggingface.co/spaces/zaid646/genai-agent-demo)

---

## Overview

The agent follows a Thought → Action → Observation loop, selecting tools autonomously to answer user queries step by step. Each reasoning step is streamed to the UI for full transparency.

**Default provider:** NVIDIA (`minimaxai/minimax-m3`)

### Tools

| Tool | Description | Source |
|---|---|---|
| `calculator` | Evaluates mathematical expressions | Built-in `eval()` with sandboxed globals |
| `get_weather` | Live weather for any city | [Open-Meteo](https://open-meteo.com/) (free, no API key) |
| `web_search` | General web search | [Google Custom Search API](https://programmablesearchengine.google.com/) |

---

## Running Locally

```bash
git clone https://github.com/zaid646/genai-agent-demo.git
cd genai-agent-demo
pip install -r requirements.txt
python app.py
```

### Required Environment Variables

| Variable | Description |
|---|---|
| `CEREBRAS_API_KEY` | Cerebras API key (for Cerebras provider) |
| `NVIDIA_API_KEY` | NVIDIA API key (for NVIDIA provider) |
| `GOOGLE_API_KEY` | Google Custom Search API key |
| `GOOGLE_CX` | Google Programmable Search Engine ID |

---

## License

MIT — see [LICENSE](LICENSE).
