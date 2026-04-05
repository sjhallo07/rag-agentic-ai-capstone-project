# Gradio Docs MCP Server

This workspace includes a ready-to-copy MCP configuration in `mcp.json` for the official Gradio Docs MCP server.

## Included entries

- `gradio-docs` — for MCP clients that support streamable HTTP directly.
- `gradio-docs-claude` — for stdio-only clients such as Claude Desktop, using `mcp-remote` over `npx`.

## Official server URL

`https://gradio-docs-mcp.hf.space/gradio_api/mcp/`

## Tools exposed by the official server

- `docs_mcp_load_gradio_docs`
- `docs_mcp_search_gradio_docs`

## Client notes

### Cursor / Windsurf / Cline

Use the `gradio-docs` entry or copy just this snippet:

```json
{
  "mcpServers": {
    "gradio-docs": {
      "url": "https://gradio-docs-mcp.hf.space/gradio_api/mcp/"
    }
  }
}
```

### Claude Desktop

Use the `gradio-docs-claude` entry or copy just this snippet:

```json
{
  "mcpServers": {
    "gradio-docs": {
      "command": "npx",
      "args": [
        "-y",
        "mcp-remote@latest",
        "https://gradio-docs-mcp.hf.space/gradio_api/mcp/"
      ]
    }
  }
}
```

## Local environment check

Node.js and `npx` are installed on this machine, and direct remote MCP connections work through the Python MCP SDK.

However, an `npx`-based probe using the MCP Inspector currently fails because the local npm installation is missing a dependency (`lru-cache`).

Practical impact:

- `gradio-docs` (direct remote MCP URL) is working.
- `gradio-docs-claude` may require repairing the local Node/npm installation before it works reliably through `mcp-remote`.
