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

```
.
├── main.py                        # Entry-point / demo script
├── requirements.txt               # Python dependencies
├── src/
│   └── rsi_agent/
│       ├── __init__.py            # Package exports
│       ├── agent.py               # RSIAgent (LangChain ReAct + RAG)
│       ├── calculator.py          # RSICalculator (pure Python/pandas)
│       ├── knowledge_base.py      # RSI knowledge documents for RAG
│       └── tools.py               # LangChain tools (fetch prices, compute RSI, interpret RSI)
└── tests/
    ├── test_calculator.py         # Unit tests for RSICalculator
    └── test_tools.py              # Unit tests for LangChain tools
```

---

## RSI Calculation

The RSI is computed using the **Wilder Smoothed Moving Average** (exponential moving average with `com = period - 1`) over a default 14-period look-back:

```
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
