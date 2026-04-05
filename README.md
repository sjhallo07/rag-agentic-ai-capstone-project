# RAG Agentic AI – RSI Agents

A capstone project that combines **Retrieval-Augmented Generation (RAG)** with an **agentic AI** framework to analyse stocks using the **Relative Strength Index (RSI)**.

---

## Overview

The RSI Agent is a [LangChain](https://python.langchain.com/) ReAct agent that:

1. **Retrieves** relevant RSI knowledge from an embedded vector store (RAG).
2. **Fetches** live historical stock prices via [yfinance](https://github.com/ranaroussi/yfinance).
3. **Calculates** the RSI for any traded ticker using a configurable look-back period.
4. **Interprets** the RSI signal in plain English (overbought / oversold / neutral).
5. **Answers** natural-language questions about RSI and specific stocks using an LLM.

---

## Project Structure

```text
.
├── app.py                         # Gradio chatbot that acts as an MCP client
├── hf_integrations_demo.py        # Demo of Gradio + Hugging Face model/Space loading
├── hf_space_tools.py              # Helper to create/update a Hugging Face Space
├── gradio_app.py                  # Gradio UI + MCP server entry-point
├── main.py                        # Entry-point / demo script
├── rsi_mcp_server.py              # STDIO MCP server for local MCP clients
├── requirements.txt               # Python dependencies
├── src/
│   └── rsi_agent/
│       ├── __init__.py            # Package exports
│       ├── agent.py               # RSIAgent (LangChain ReAct + RAG)
│       ├── calculator.py          # RSICalculator (pure Python/pandas)
│       ├── knowledge_base.py      # RSI knowledge documents for RAG
│       ├── service_api.py         # Reusable functions for CLI/Gradio/MCP
│       └── tools.py               # LangChain tools (fetch prices, compute RSI, interpret RSI)
└── tests/
    ├── test_calculator.py         # Unit tests for RSICalculator
    └── test_tools.py              # Unit tests for LangChain tools
```

---

## RSI Calculation

The RSI is computed using the **Wilder Smoothed Moving Average** (exponential moving average with `com = period - 1`) over a default 14-period look-back:

```text
RS  = avg_gain / avg_loss
RSI = 100 − (100 / (1 + RS))
```

| RSI Range | Signal      | Meaning                                   |
|-----------|-------------|-------------------------------------------|
| ≥ 70      | Overbought  | Potential pullback / reversal             |
| 30 – 70   | Neutral     | No extreme momentum                       |
| ≤ 30      | Oversold    | Potential rebound / reversal              |

---

## Setup

> Recommended Python version: **3.10–3.13**. The current LangChain stack used by the LLM-backed RSI agent is not yet compatible with Python 3.14 in this project.

```bash
# 1. Clone the repository
git clone https://github.com/sjhallo07/rag-agentic-ai-capstone-project.git
cd rag-agentic-ai-capstone-project

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

The requirements file includes `gradio[mcp]`, so the Gradio app can be launched as an MCP server without any extra install step.

If you want to use the Gradio chatbot as an MCP client, the same requirements file also installs `anthropic` and `mcp`.

---

## Usage

### Demo (no API key required)

```bash
python main.py
```

Runs the RSI calculator on synthetic price data and prints the interpretations.

### Full agent (requires OpenAI API key)

```bash
export OPENAI_API_KEY="sk-..."
python main.py
```

The agent will fetch live data, calculate RSI, retrieve knowledge-base context, and answer questions using an LLM.

### Gradio UI + MCP server

```bash
python gradio_app.py
```

This launches both:

- a local Gradio web app for RSI tools and agent queries
- an MCP server endpoint at `http://127.0.0.1:7860/gradio_api/mcp/`

If `OPENAI_API_KEY` is configured, the **Ask RSI Agent** tool will use the LangChain RAG agent. If not, the non-LLM tools still work:

- fetch recent stock prices
- compute RSI for a ticker
- interpret a raw RSI value
- search the built-in RSI knowledge base

Example MCP client config:

```json
{
    "mcpServers": {
        "rsi-gradio": {
            "url": "http://127.0.0.1:7860/gradio_api/mcp/"
        }
    }
}
```

Useful MCP URLs once the app is running:

- schema: `http://127.0.0.1:7860/gradio_api/mcp/schema`
- endpoint: `http://127.0.0.1:7860/gradio_api/mcp/`

### STDIO MCP server for local MCP clients

```bash
python rsi_mcp_server.py
```

This starts a **stdio MCP server** that exposes these tools:

- `fetch_prices`
- `compute_rsi`
- `interpret_rsi`
- `search_knowledge`
- `ask_agent`

This mode is useful for MCP clients that launch local scripts directly.

### Gradio chatbot as an MCP client

```bash
python app.py
```

This launches a Gradio chat UI that acts as an **MCP client**. By default it connects to `rsi_mcp_server.py`, but you can point it to any compatible local stdio MCP server script.

To use the chatbot client, add the following variable to your local environment or `.env` file:

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Suggested flow:

1. Run `python app.py`
2. Open the Gradio URL in your browser
3. Leave the default server path as `rsi_mcp_server.py`
4. Click **Connect**
5. Ask questions like:
    - `What is the current RSI for AAPL?`
    - `Search the knowledge base for RSI divergence.`
    - `Fetch recent prices for TSLA and explain the momentum.`

### Hugging Face integrations demo

```bash
python hf_integrations_demo.py
```

This launches a small Gradio app showing two practical Hugging Face integrations:

- loading a Hub model through `gr.load(..., src="models")`
- remixing existing Hugging Face Spaces through `gr.load(..., src="spaces")`

If a remote Space uses a newer config shape than the locally installed Gradio version supports, the demo falls back to showing a direct Hugging Face link instead of crashing.

It uses these public examples:

- `Helsinki-NLP/opus-mt-en-es`
- `gradio/en2es`
- `abidlabs/en2fr`

### Publish this app to a Hugging Face Space

```bash
python hf_space_tools.py your-space-name
```

This helper will:

- create or reuse a Gradio Space under your Hugging Face account
- upload `gradio_app.py` as the Space `app.py`
- upload `requirements.txt`
- upload the `src/` package needed by the app

Before running it, set a Hugging Face write token in your environment or `.env` file:

```bash
HF_TOKEN=your_hugging_face_token_here
```

### Programmatic usage

```python
import os
from langchain_openai import ChatOpenAI
from src.rsi_agent import RSIAgent, RSICalculator

# --- RSI calculator (no LLM needed) ---
calc = RSICalculator(period=14)
prices = [100, 102, 101, 105, 110, 108, 107, 112, 115, 113,
          116, 120, 118, 117, 122, 125, 123, 128, 130, 127]
rsi = calc.compute(prices)
print(RSICalculator.interpret(float(rsi.iloc[-1])))

# --- Full RSI agent ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = RSIAgent(llm=llm, use_rag=True, verbose=True)
answer = agent.run("What is the RSI for TSLA and what does it mean?")
print(answer)
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Dependencies

| Package              | Purpose                              |
|----------------------|--------------------------------------|
| `langchain`          | Agent framework                      |
| `langchain-openai`   | OpenAI LLM integration               |
| `langchain-community`| FAISS vector store integration       |
| `gradio[mcp]`        | Web UI + MCP server integration      |
| `anthropic`          | Claude-powered MCP chat client       |
| `mcp`                | Python MCP client/server SDK         |
| `huggingface_hub`    | Hugging Face Hub and Spaces uploads  |
| `faiss-cpu`          | In-memory vector search (RAG)        |
| `openai`             | OpenAI API client                    |
| `yfinance`           | Historical stock price data          |
| `pandas`             | Data manipulation                    |
| `numpy`              | Numerical computation                |
| `python-dotenv`      | Environment variable management      |
| `pydantic`           | Data validation for tool schemas     |

---

## License

MIT
