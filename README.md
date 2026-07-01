---
title: Gen AI Agent Demo
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.36.2
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# Gen AI Agent Demo

Multi-provider ReAct agent with tool use. Switch between **Cerebras** and **NVIDIA** backends.

## Features

- ReAct reasoning loop (Thought → Action → Observation)
- Tools: calculator, web search
- Provider toggle between Cerebras (`gpt-oss-120b`) and NVIDIA (`meta/llama-3.1-405b-instruct`)
- Step-by-step reasoning visible to user

## Setting up API Keys

Add these secrets in your Space settings:

| Secret | Value |
|---|---|
| `CEREBRAS_API_KEY` | Your Cerebras API key |
| `NVIDIA_API_KEY` | Your NVIDIA API key |
