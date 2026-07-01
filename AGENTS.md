# genai-agent-demo

## Session Context (Jul 2026)

### Status
- Deployed at: https://zaid646-genai-agent-demo.hf.space
- Default provider: NVIDIA (minimaxai/minimax-m3)
- All 3 tools working: calculator, get_weather, web_search

### Key Decisions
- Switched from Cerebras to NVIDIA minimax-m3 as default model
- Web search uses Google Custom Search API (100 free queries/day)
- Weather data via Open-Meteo (free, no API key)
- Replaced DuckDuckGo HTML scraping (blocked by JS challenges)

### Secrets Required (HF Space)
- `CEREBRAS_API_KEY` — for Cerebras provider
- `NVIDIA_API_KEY` — for NVIDIA provider
- `GOOGLE_API_KEY` — for Google Custom Search
- `GOOGLE_CX` — Google Search Engine ID

### Known Issues
- Web search returns indexed content (not real-time feeds)
- NVIDIA API has rate limits (429 errors under load)
- Google API limited to 100 queries/day on free tier

### Fixed Bugs
1. Stray `}` syntax error after DEFAULT_PROVIDER line
2. Dead DuckDuckGo Instant Answer API → Wikipedia → Google
3. Missing Space secrets (caused silent 401 errors)
