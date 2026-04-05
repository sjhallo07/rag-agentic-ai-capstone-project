"""Gradio chatbot acting as an MCP client for local STDIO MCP servers."""

from __future__ import annotations

import asyncio
import html
import json
import os
import sys
import textwrap
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

try:
    from ibm_watsonx_ai import APIClient, Credentials

    IBM_WATSONX_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on optional package install
    APIClient = None  # type: ignore[assignment]
    Credentials = None  # type: ignore[assignment]
    IBM_WATSONX_IMPORT_ERROR = exc

load_dotenv()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

MODE_CONVERSATIONAL = "Conversational"
MODE_TOOL_ASSISTED = "Tool-assisted"
IBM_WATSONX_DEFAULT_URL = "https://us-south.ml.cloud.ibm.com"
IBM_WATSONX_PLACEHOLDERS = {
    "",
    "your_ibm_watsonx_token_here",
    "your_ibm_watsonx_api_key_here",
}
WATSON_ASSISTANT_PLACEHOLDERS = {
    "",
    "your_watson_assistant_integration_id_here",
    "your_watson_assistant_service_instance_id_here",
}


def _get_watson_assistant_embed() -> tuple[str, str]:
    """Return the Watson Assistant embed script and a human-readable status."""
    integration_id = os.getenv("WATSON_ASSISTANT_INTEGRATION_ID", "").strip()
    region = os.getenv("WATSON_ASSISTANT_REGION", "us-south").strip() or "us-south"
    service_instance_id = os.getenv("WATSON_ASSISTANT_SERVICE_INSTANCE_ID", "").strip()
    client_version = os.getenv("WATSON_ASSISTANT_CLIENT_VERSION", "latest").strip() or "latest"

    if (
        integration_id in WATSON_ASSISTANT_PLACEHOLDERS
        or service_instance_id in WATSON_ASSISTANT_PLACEHOLDERS
    ):
        return "", "Watson Assistant widget not configured yet. Add the integration and service instance IDs to `.env`."

    script = textwrap.dedent(
        f"""
        <script>
          window.watsonAssistantChatOptions = {{
            integrationID: {json.dumps(integration_id)},
            region: {json.dumps(region)},
            serviceInstanceID: {json.dumps(service_instance_id)},
            clientVersion: {json.dumps(client_version)},
            onLoad: async (instance) => {{ await instance.render(); }}
          }};
          setTimeout(function() {{
            const t = document.createElement('script');
            t.src = "https://web-chat.global.assistant.watson.appdomain.cloud/versions/" +
              (window.watsonAssistantChatOptions.clientVersion || 'latest') +
              "/WatsonAssistantChatEntry.js";
            document.head.appendChild(t);
          }});
        </script>
        """
    ).strip()
    return script, f"Watson Assistant widget configured for region `{region}`."


def _get_watson_assistant_preview() -> tuple[str, str, str]:
    """Return the configured Watson Assistant preview URL, status, and iframe HTML."""
    preview_url = os.getenv("WATSON_ASSISTANT_PREVIEW_URL", "").strip()
    if not preview_url:
        return "", "Watson Assistant preview URL not configured.", ""

    safe_url = html.escape(preview_url, quote=True)
    iframe_html = textwrap.dedent(
        f"""
        <iframe
          src="{safe_url}"
          title="Watson Assistant Preview"
          style="width: 100%; height: 640px; border: 1px solid #d0d7de; border-radius: 12px; background: white;"
          loading="lazy"
          referrerpolicy="no-referrer"
        ></iframe>
        """
    ).strip()
    return preview_url, "Watson Assistant preview URL configured.", iframe_html


class MCPClientWrapper:
    """Manage local or remote MCP connections and relay chat messages through Anthropic."""

    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self.exit_stack: AsyncExitStack | None = None
        self.remote_url: str | None = None
        self.tools: list[dict[str, Any]] = []
        self.anthropic: Anthropic | None = None
        self.ibm_watsonx: Any | None = None
        self.ibm_watsonx_status = self._initialize_ibm_watsonx_client()

    def connect(self, server_path: str) -> str:
        """Connect to either a local stdio server script or a remote MCP URL."""
        return loop.run_until_complete(self._connect(server_path))

    def _initialize_ibm_watsonx_client(self) -> str:
        """Initialize an optional IBM watsonx.ai client from environment variables."""
        if IBM_WATSONX_IMPORT_ERROR is not None or APIClient is None or Credentials is None:
            return "IBM watsonx.ai SDK not installed. Add `ibm-watsonx-ai` to use this connection."

        url = os.getenv("IBM_WATSONX_URL", IBM_WATSONX_DEFAULT_URL).strip() or IBM_WATSONX_DEFAULT_URL
        token = os.getenv("IBM_WATSONX_TOKEN", "").strip()
        api_key = os.getenv("IBM_WATSONX_API_KEY", "").strip()

        if token in IBM_WATSONX_PLACEHOLDERS:
            token = ""
        if api_key in IBM_WATSONX_PLACEHOLDERS:
            api_key = ""

        if not token and not api_key:
            return "IBM watsonx.ai not configured yet. Set IBM_WATSONX_TOKEN or IBM_WATSONX_API_KEY in `.env`."

        try:
            if token:
                credentials = Credentials(url=url, token=token)
            else:
                credentials = Credentials(url=url, api_key=api_key)

            self.ibm_watsonx = APIClient(credentials)
            auth_mode = "token" if token else "api_key"
            return f"IBM watsonx.ai ready ({auth_mode}) at {url}"
        except Exception as exc:  # pragma: no cover - depends on external credentials
            self.ibm_watsonx = None
            return f"IBM watsonx.ai configuration error: {exc}"

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

    @staticmethod
    def _build_system_prompt(response_mode: str) -> str:
        if response_mode == MODE_CONVERSATIONAL:
            return (
                "You are a warm, natural, human-sounding assistant. "
                "Be conversational, clear, and helpful. "
                "Use tools when they add value, but do not overemphasize the tool usage. "
                "Avoid robotic phrasing, avoid unnecessary jargon, and do not dump raw technical data unless the user asks for it. "
                "When you have tool results, summarize them naturally and directly."
            )

        return (
            "You are a precise technical assistant using MCP tools. "
            "Be structured, explicit, and transparent about the information you gather. "
            "When tools are used, present the relevant intermediate details clearly."
        )

    def process_message(
        self,
        message: str,
        history: list[dict[str, Any]],
        response_mode: str,
    ) -> tuple[list[dict[str, Any]], str]:
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

        new_messages = loop.run_until_complete(
            self._process_query(message, history, response_mode)
        )
        return history + [{"role": "user", "content": message}] + new_messages, ""

    async def _process_query(
        self,
        message: str,
        history: list[dict[str, Any]],
        response_mode: str,
    ) -> list[dict[str, Any]]:
        claude_messages = self._to_claude_messages(history)
        claude_messages.append({"role": "user", "content": message})
        system_prompt = self._build_system_prompt(response_mode)

        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1200,
            messages=claude_messages,
            tools=self.tools,
            system=system_prompt,
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
            if response_mode == MODE_CONVERSATIONAL:
                result_messages.append(
                    {
                        "role": "assistant",
                        "content": "Let me check that for you.",
                    }
                )
            else:
                result_messages.append(
                    {
                        "role": "assistant",
                        "content": f"Using tool `{tool_name}` with arguments:\n```json\n{json.dumps(tool_args, indent=2)}\n```",
                    }
                )

            result = await self._call_tool(tool_name, tool_args)
            result_text = self._normalize_tool_result(result.content)
            if response_mode != MODE_CONVERSATIONAL:
                result_messages.extend(self._format_tool_result(tool_name, result_text))

            claude_messages.append({"role": "assistant", "content": f"Calling tool {tool_name}."})
            claude_messages.append({"role": "user", "content": f"Tool result for {tool_name}: {result_text}"})

            follow_up = self.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=claude_messages,
                system=system_prompt,
            )
            assistant_reply = self._extract_text_response(follow_up)
            if assistant_reply:
                result_messages.append({"role": "assistant", "content": assistant_reply})
            elif response_mode == MODE_CONVERSATIONAL:
                result_messages.extend(self._format_tool_result(tool_name, result_text))

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
WATSON_ASSISTANT_HEAD, WATSON_ASSISTANT_STATUS = _get_watson_assistant_embed()
WATSON_ASSISTANT_PREVIEW_URL, WATSON_ASSISTANT_PREVIEW_STATUS, WATSON_ASSISTANT_PREVIEW_IFRAME = (
    _get_watson_assistant_preview()
)


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
        ibm_status = gr.Textbox(
            label="IBM watsonx.ai Status",
            value=client.ibm_watsonx_status,
            interactive=False,
        )
        watson_assistant_status = gr.Textbox(
            label="Watson Assistant Widget Status",
            value=WATSON_ASSISTANT_STATUS,
            interactive=False,
        )
        watson_assistant_preview_status = gr.Textbox(
            label="Watson Assistant Preview Status",
            value=WATSON_ASSISTANT_PREVIEW_STATUS,
            interactive=False,
        )

        if WATSON_ASSISTANT_PREVIEW_URL:
            gr.HTML(
                f'<p>Open the official Watson Assistant preview here: <a href="{html.escape(WATSON_ASSISTANT_PREVIEW_URL, quote=True)}" target="_blank" rel="noopener noreferrer">Launch Watson Assistant preview</a></p>'
            )
            with gr.Accordion("Watson Assistant Preview", open=False):
                gr.HTML(WATSON_ASSISTANT_PREVIEW_IFRAME)

        response_mode = gr.Radio(
            choices=[MODE_CONVERSATIONAL, MODE_TOOL_ASSISTED],
            value=MODE_CONVERSATIONAL,
            label="Response Style",
            info="Conversational sounds more human and hides most tool chatter. Tool-assisted is more explicit and technical.",
        )

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
        msg.submit(
            client.process_message,
            inputs=[msg, chatbot, response_mode],
            outputs=[chatbot, msg],
        )
        clear_btn.click(lambda: [], outputs=chatbot)

    return demo


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "Warning: ANTHROPIC_API_KEY not found in environment. Add it to `.env` or your shell before chatting."
        )

    demo = build_interface()
    demo.launch(debug=True, head=WATSON_ASSISTANT_HEAD)
