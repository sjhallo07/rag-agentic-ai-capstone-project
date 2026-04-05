"""Gradio chatbot acting as an MCP client for local STDIO MCP servers."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import gradio as gr
from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

load_dotenv()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


class MCPClientWrapper:
    """Manage local or remote MCP connections and relay chat messages through Anthropic."""

    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self.exit_stack: AsyncExitStack | None = None
        self.remote_url: str | None = None
        self.tools: list[dict[str, Any]] = []
        self.anthropic: Anthropic | None = None

    def connect(self, server_path: str) -> str:
        """Connect to either a local stdio server script or a remote MCP URL."""
        return loop.run_until_complete(self._connect(server_path))

    @staticmethod
    def _is_remote_target(target: str) -> bool:
        parsed = urlparse(target.strip())
        return parsed.scheme in {"http", "https"}

    async def _connect(self, server_path: str) -> str:
        if self.exit_stack:
            await self.exit_stack.aclose()
            self.session = None
            self.exit_stack = None

        self.remote_url = None
        self.tools = []

        target = server_path.strip()
        if not target:
            return "Please provide either a local MCP server script path or a remote MCP URL."

        if self._is_remote_target(target):
            return await self._connect_remote(target)

        resolved = Path(target).expanduser()
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        if not resolved.exists():
            return f"Server script not found: {resolved}"

        self.exit_stack = AsyncExitStack()

        command = sys.executable if resolved.suffix.lower() == ".py" else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[str(resolved)],
            env={
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
            },
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await self.session.initialize()

        return await self._load_tools_and_format_status(
            f"Connected to local MCP server at {resolved}"
        )

    async def _connect_remote(self, url: str) -> str:
        self.remote_url = url
        self.tools = await self._list_remote_tools(url)
        tool_names = ", ".join(tool["name"] for tool in self.tools) or "(no tools found)"
        return f"Connected to remote MCP server at {url}. Available tools: {tool_names}"

    async def _load_tools_and_format_status(self, prefix: str) -> str:
        response = await self.session.list_tools()
        self.tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]

        tool_names = ", ".join(tool["name"] for tool in self.tools) or "(no tools found)"
        return f"{prefix}. Available tools: {tool_names}"

    async def _list_remote_tools(self, url: str) -> list[dict[str, Any]]:
        async with streamablehttp_client(url) as transport:
            read_stream, write_stream, _ = transport
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                response = await session.list_tools()
                return [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                    for tool in response.tools
                ]

    async def _call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        if self.remote_url:
            async with streamablehttp_client(self.remote_url) as transport:
                read_stream, write_stream, _ = transport
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    return await session.call_tool(tool_name, tool_args)

        return await self.session.call_tool(tool_name, tool_args)

    def process_message(self, message: str, history: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
        """Send a user message through Anthropic and the connected MCP server."""
        if not message.strip():
            return history, ""
        if not self.session and not self.remote_url:
            return history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Please connect to an MCP server first."},
            ], ""

        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key or api_key == "your_anthropic_api_key_here":
            return history + [
                {"role": "user", "content": message},
                {
                    "role": "assistant",
                    "content": (
                        "ANTHROPIC_API_KEY is not configured. Add it to your environment or `.env` file "
                        "to enable the MCP chatbot client."
                    ),
                },
            ], ""

        if self.anthropic is None:
            self.anthropic = Anthropic(api_key=api_key)

        new_messages = loop.run_until_complete(self._process_query(message, history))
        return history + [{"role": "user", "content": message}] + new_messages, ""

    async def _process_query(self, message: str, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        claude_messages = self._to_claude_messages(history)
        claude_messages.append({"role": "user", "content": message})

        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1200,
            messages=claude_messages,
            tools=self.tools,
        )

        result_messages: list[dict[str, Any]] = []

        for content in response.content:
            if content.type == "text" and content.text.strip():
                result_messages.append({"role": "assistant", "content": content.text})
                continue

            if content.type != "tool_use":
                continue

            tool_name = content.name
            tool_args = content.input
            result_messages.append(
                {
                    "role": "assistant",
                    "content": f"Using tool `{tool_name}` with arguments:\n```json\n{json.dumps(tool_args, indent=2)}\n```",
                }
            )

            result = await self._call_tool(tool_name, tool_args)
            result_text = self._normalize_tool_result(result.content)
            result_messages.extend(self._format_tool_result(tool_name, result_text))

            claude_messages.append({"role": "assistant", "content": f"Calling tool {tool_name}."})
            claude_messages.append({"role": "user", "content": f"Tool result for {tool_name}: {result_text}"})

            follow_up = self.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=claude_messages,
            )
            assistant_reply = self._extract_text_response(follow_up)
            if assistant_reply:
                result_messages.append({"role": "assistant", "content": assistant_reply})

        if not result_messages:
            result_messages.append(
                {
                    "role": "assistant",
                    "content": "No response was produced. Try a clearer request or reconnect to the MCP server.",
                }
            )
        return result_messages

    @staticmethod
    def _to_claude_messages(history: list[dict[str, Any]]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            if role in {"user", "assistant", "system"} and isinstance(content, str):
                messages.append({"role": role, "content": content})
        return messages

    @staticmethod
    def _normalize_tool_result(result_content: Any) -> str:
        if isinstance(result_content, list):
            normalized: list[str] = []
            for item in result_content:
                text_value = getattr(item, "text", None)
                if text_value is not None:
                    normalized.append(str(text_value))
                else:
                    normalized.append(str(item))
            return "\n".join(normalized)
        return str(result_content)

    @staticmethod
    def _extract_text_response(response: Any) -> str:
        parts: list[str] = []
        for item in getattr(response, "content", []):
            if getattr(item, "type", None) == "text" and getattr(item, "text", "").strip():
                parts.append(item.text)
        return "\n\n".join(parts)

    @staticmethod
    def _format_tool_result(tool_name: str, result_text: str) -> list[dict[str, str]]:
        try:
            parsed = json.loads(result_text)
        except Exception:
            parsed = None

        if isinstance(parsed, dict) and parsed.get("type") == "image" and parsed.get("url"):
            alt = parsed.get("message", f"Output from {tool_name}")
            return [
                {"role": "assistant", "content": f"Tool `{tool_name}` returned an image:\n\n![{alt}]({parsed['url']})"}
            ]

        pretty = result_text
        if isinstance(parsed, (dict, list)):
            pretty = json.dumps(parsed, indent=2)

        return [
            {
                "role": "assistant",
                "content": f"Tool `{tool_name}` result:\n```\n{pretty}\n```",
            }
        ]


client = MCPClientWrapper()


def build_interface() -> gr.Blocks:
    """Build the Gradio chatbot UI for the MCP client."""
    with gr.Blocks(title="RSI MCP Chat Client") as demo:
        gr.Markdown("# RSI MCP Chat Client")
        gr.Markdown(
            "Connect to either a local stdio MCP server such as `rsi_mcp_server.py` or a remote MCP URL such as `https://gradio-docs-mcp.hf.space/gradio_api/mcp/`."
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                server_path = gr.Textbox(
                    label="Server Script Path or MCP URL",
                    placeholder="Enter a local script path or remote MCP URL",
                    value="rsi_mcp_server.py",
                )
            with gr.Column(scale=1):
                connect_btn = gr.Button("Connect")

        status = gr.Textbox(label="Connection Status", interactive=False)

        chatbot = gr.Chatbot(
            value=[],
            height=520,
        )

        with gr.Row(equal_height=True):
            msg = gr.Textbox(
                label="Your Question",
                placeholder="Ask about RSI, prices, or anything your MCP tools support",
                scale=4,
            )
            clear_btn = gr.Button("Clear Chat", scale=1)

        connect_btn.click(client.connect, inputs=server_path, outputs=status)
        msg.submit(client.process_message, inputs=[msg, chatbot], outputs=[chatbot, msg])
        clear_btn.click(lambda: [], outputs=chatbot)

    return demo


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "Warning: ANTHROPIC_API_KEY not found in environment. Add it to `.env` or your shell before chatting."
        )

    demo = build_interface()
    demo.launch(debug=True)
