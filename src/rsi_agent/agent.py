"""RSI Agent – a LangChain ReAct agent with RAG-augmented RSI knowledge."""

from __future__ import annotations

import os
from typing import Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever

from .knowledge_base import RSI_KNOWLEDGE_DOCS
from .tools import get_rsi_tools


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PREFIX = """You are an expert financial analyst specializing in technical analysis,
specifically the Relative Strength Index (RSI). You help users understand RSI signals for
stocks and other traded assets.

You have access to a knowledge base with detailed RSI concepts (retrieved below) and a set
of tools to fetch live stock data and calculate RSI in real time.

RSI Knowledge (retrieved context):
{rag_context}

Use the tools available to you to answer the user's question accurately. Think step-by-step,
call tools when you need real data, and always cite RSI levels in your final answer.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


# ---------------------------------------------------------------------------
# Helper: build RAG retriever
# ---------------------------------------------------------------------------


def _build_retriever(
    llm: BaseLanguageModel,
) -> VectorStoreRetriever:
    """Build an in-memory FAISS vector store from the RSI knowledge base."""
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(RSI_KNOWLEDGE_DOCS, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


# ---------------------------------------------------------------------------
# RSIAgent
# ---------------------------------------------------------------------------


class RSIAgent:
    """LangChain ReAct agent for RSI analysis, augmented with a RAG knowledge base.

    Args:
        llm: A LangChain-compatible LLM (e.g. ChatOpenAI).
        use_rag: Whether to retrieve context from the RSI knowledge base
                 before each query. Defaults to True.
        verbose: Whether to stream agent reasoning steps to stdout.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        use_rag: bool = True,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.use_rag = use_rag
        self.verbose = verbose
        self._retriever: Optional[VectorStoreRetriever] = None
        self._executor: Optional[AgentExecutor] = None
        self._tools = get_rsi_tools()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_retriever(self) -> VectorStoreRetriever:
        if self._retriever is None:
            self._retriever = _build_retriever(self.llm)
        return self._retriever

    def _retrieve_context(self, query: str) -> str:
        """Retrieve relevant RSI knowledge snippets for the given query."""
        docs = self._get_retriever().invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)

    def _build_executor(self, rag_context: str) -> AgentExecutor:
        prompt = PromptTemplate.from_template(_SYSTEM_PREFIX)
        prompt = prompt.partial(rag_context=rag_context)
        agent = create_react_agent(self.llm, self._tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self._tools,
            verbose=self.verbose,
            handle_parsing_errors=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str) -> str:
        """Run the RSI agent on the given query and return the response.

        Args:
            query: A natural language question about RSI or a specific stock.

        Returns:
            The agent's final answer as a string.
        """
        rag_context = (
            self._retrieve_context(query)
            if self.use_rag
            else "No context retrieved (RAG disabled)."
        )
        executor = self._build_executor(rag_context)
        result = executor.invoke({"input": query})
        return result.get("output", str(result))
