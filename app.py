import json
import os
import re
from openai import OpenAI

PROVIDERS = {
    "Cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "models": ["gpt-oss-120b"],
    },
    "NVIDIA": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
        "models": ["minimaxai/minimax-m3"],
    },
}

TOOLS = []

class BaseTool:
    name: str = ""
    description: str = ""
    parameters: dict = {}

    def run(self, **kwargs) -> str:
        raise NotImplementedError


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluate mathematical expressions. Input: expression (string)"
    parameters = {"expression": "mathematical expression to evaluate (e.g., '15 * 7 + 3')"}

    def run(self, expression: str = "") -> str:
        try:
            allowed = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "int": int, "float": float, "str": str, "len": len, "range": range}
            result = eval(expression, {"__builtins__": {}}, allowed)
            return str(result)
        except Exception as e:
            return f"Error evaluating '{expression}': {e}"


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for current information. Input: query (string)"
    parameters = {"query": "search query"}

    def run(self, query: str = "", q: str = "") -> str:
        import httpx
        import re
        search_term = query or q
        if not search_term:
            return "No search query provided."

        # Try Wikipedia API first (fast, free, reliable)
        try:
            resp = httpx.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": search_term,
                    "format": "json",
                    "srlimit": 3,
                },
                headers={"User-Agent": "GenAIAgentDemo/1.0"},
                timeout=10,
            )
            data = resp.json()
            pages = data.get("query", {}).get("search", [])
            if pages:
                results = []
                for p in pages:
                    title = p.get("title", "")
                    snippet = re.sub(r'<[^>]+>', '', p.get("snippet", "")).strip()
                    results.append(f"{title}: {snippet}")
                return "\n\n".join(results)[:2000]
        except Exception:
            pass

        # Fallback to DuckDuckGo HTML search
        try:
            resp = httpx.get(
                "https://html.duckduckgo.com/html/",
                params={"q": search_term},
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
                timeout=20,
            )
            snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', resp.text, re.DOTALL)
            if not snippets:
                snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</', resp.text, re.DOTALL)
            if snippets:
                clean_snippets = []
                for s in snippets[:3]:
                    clean = re.sub(r'<[^>]+>', '', s).strip()
                    for old, new in [('&#x27;', "'"), ('&amp;', '&'), ('&quot;', '"'), ('&lt;', '<'), ('&gt;', '>')]:
                        clean = clean.replace(old, new)
                    clean_snippets.append(clean)
                return "\n\n".join(clean_snippets)[:2000]
            return "No web results found."
        except Exception as e:
            return f"Search error: {e}"


TOOLS = [CalculatorTool(), WebSearchTool()]

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools. Follow the ReAct format.

Respond ONLY with valid JSON (no markdown, no code fences, no extra text).

EXAMPLES:
For calculation: {{"thought": "I need to calculate this", "tool": "calculator", "tool_input": {{"expression": "15 * 7 + 3"}}}}
For web search: {{"thought": "I need to look this up", "tool": "web_search", "tool_input": {{"query": "population of Tokyo"}}}}
When done: {{"thought": "Final answer here", "tool": null, "tool_input": null}}

Available tools:
{}

CRITICAL: Respond with ONLY valid JSON. No markdown. No explanations."""


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        text = match.group()
    return json.loads(text)


def run_agent(message: str, provider_name: str, model: str):
    provider = PROVIDERS.get(provider_name)
    if not provider:
        yield "error", f"Unknown provider: {provider_name}"
        return

    api_key = os.environ.get(provider["api_key_env"], "")
    if not api_key:
        yield "error", f"❌ **{provider_name} API key not set.**\n\nConfigure `{provider['api_key_env']}` as a Space secret."
        return

    client = OpenAI(api_key=api_key, base_url=provider["base_url"])
    tool_descs = "\n".join(
        f"- **{t.name}**: {t.description}  \n  Parameters: {json.dumps(t.parameters)}"
        for t in TOOLS
    )
    tools_dict = {t.name: t for t in TOOLS}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(tool_descs)},
        {"role": "user", "content": message},
    ]

    yield "thinking", f"🤔 Starting agent with **{provider_name}/{model}**..."

    for step in range(10):
        yield "thinking", f"⏳ **Step {step + 1}:** Querying {model}..."

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
            )
        except Exception as e:
            yield "error", f"❌ **API Error:**\n\n```\n{e}\n```"
            return

        content = response.choices[0].message.content
        if not content:
            yield "error", "❌ Empty response from API."
            return

        try:
            action = extract_json(content)
        except (json.JSONDecodeError, KeyError) as e:
            yield "thinking", f"Raw response:\n```\n{content[:800]}\n```"
            yield "error", f"❌ **Parse Error:** Could not understand model response: {e}"
            return

        thought = action.get("thought", "")
        tool_name = action.get("tool")
        tool_input = action.get("tool_input")

        if tool_name is None:
            yield "result", thought
            return

        if tool_name not in tools_dict:
            observation = f"Unknown tool: {tool_name}. Available: {', '.join(tools_dict.keys())}"
        else:
            yield "thinking", f"🔧 **Using:** `{tool_name}` with input `{tool_input}`"
            tool = tools_dict[tool_name]
            try:
                observation = tool.run(**(tool_input or {}))
            except Exception as e:
                observation = f"Tool error: {e}"

        yield "step", (
            f"**Thought:** {thought}\n\n"
            f"**Action:** `{tool_name}({tool_input})`\n\n"
            f"**Observation:** {observation[:600]}"
        )

        messages.append({"role": "assistant", "content": content})
        messages.append({
            "role": "user",
            "content": f"Observation: {observation}\n\nContinue with next step or provide final answer.",
        })

    yield "error", "❌ **Max steps reached** without a final answer."


def format_response(message, history, provider, model):
    steps = []
    final_answer = ""
    error_msg = ""

    for step_type, content in run_agent(message, provider, model):
        if step_type == "thinking":
            steps.append(("💭", content))
        elif step_type == "step":
            steps.append(("🔧", content))
        elif step_type == "result":
            final_answer = content
        elif step_type == "error":
            error_msg = content

    if error_msg:
        return error_msg

    if not final_answer:
        if steps:
            body = "\n\n".join(f"{emo} {text}" for emo, text in steps)
            return f"<details open><summary>🤔 **Agent Steps**</summary>\n\n{body}\n\n</details>\n\n---\n\n⚠️ No final answer generated."
        return "No response generated."

    if steps:
        body = "\n\n".join(f"{emo} {text}" for emo, text in steps)
        return (
            f"<details open><summary>🤔 **Agent Steps ({len(steps)} steps)**</summary>\n\n"
            f"{body}\n\n</details>\n\n---\n\n"
            f"### ✅ Answer\n\n{final_answer}"
        )

    return final_answer


def update_models(provider_name):
    provider = PROVIDERS.get(provider_name)
    if provider:
        models = provider["models"]
        return gr.Dropdown(choices=models, value=models[0])
    return gr.Dropdown(choices=[], value=None)


import gradio as gr

with gr.Blocks(
    title="Gen AI Agent Demo",
    theme=gr.themes.Soft(),
    fill_height=True,
) as demo:
    gr.Markdown(
        "# 🤖 Gen AI Agent Demo\n"
        "### Multi-Provider ReAct Agent powered by **Cerebras** & **NVIDIA**\n\n"
        "The agent uses tools (calculator, web search) to reason step-by-step and answer your questions."
    )

    with gr.Row():
        provider_dd = gr.Dropdown(
            choices=list(PROVIDERS.keys()),
            value=list(PROVIDERS.keys())[0],
            label="AI Provider",
            scale=1,
        )
        model_dd = gr.Dropdown(
            choices=PROVIDERS[list(PROVIDERS.keys())[0]]["models"],
            value=PROVIDERS[list(PROVIDERS.keys())[0]]["models"][0],
            label="Model",
            scale=1,
        )

    provider_dd.change(fn=update_models, inputs=provider_dd, outputs=model_dd)

    gr.ChatInterface(
        fn=format_response,
        additional_inputs=[provider_dd, model_dd],
        title="Chat",
        description="Ask the agent a question. It will use tools to reason and answer.",
    )

if __name__ == "__main__":
    demo.launch()
