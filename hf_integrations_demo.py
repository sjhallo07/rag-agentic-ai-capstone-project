"""Examples of using Gradio with Hugging Face models and Spaces."""

from __future__ import annotations

import gradio as gr


MODEL_ID = "Helsinki-NLP/opus-mt-en-es"
SPACE_EN2ES = "gradio/en2es"
SPACE_EN2FR = "abidlabs/en2fr"


def _space_url(space_id: str) -> str:
    return f"https://huggingface.co/spaces/{space_id}"


def _render_space(space_id: str) -> None:
    """Render a Space via `gr.load()` when possible, else show a safe fallback."""
    try:
        gr.load(space_id, src="spaces")
    except Exception as exc:
        gr.Warning(
            f"Could not load Space `{space_id}` directly in this local Gradio version. Showing a link instead."
        )
        gr.Markdown(
            f"Direct loading for `{space_id}` failed in this environment: `{exc}`\n\n"
            f"Open it on Hugging Face instead: {_space_url(space_id)}"
        )



def build_demo() -> gr.Blocks:
    """Build a small showcase of Hugging Face integrations in Gradio."""
    with gr.Blocks(title="Gradio + Hugging Face Integrations") as demo:
        gr.Markdown("# Gradio + Hugging Face Integrations")
        gr.Markdown(
            "This demo shows two lightweight Hugging Face integrations: loading a model from the Hub and remixing existing Spaces."
        )

        with gr.Tab("Model Hub via gr.load"):
            gr.Markdown(
                f"Loading model `{MODEL_ID}` using `gr.load(..., src=\"models\")`."
            )
            gr.load(MODEL_ID, src="models")

        with gr.Tab("Spaces Remix"):
            with gr.Tabs():
                with gr.Tab("Translate to Spanish"):
                    gr.Markdown(
                        f"Loading Space `{SPACE_EN2ES}` using `gr.load(..., src=\"spaces\")`."
                    )
                    _render_space(SPACE_EN2ES)
                with gr.Tab("Translate to French"):
                    gr.Markdown(
                        f"Loading Space `{SPACE_EN2FR}` using `gr.load(..., src=\"spaces\")`."
                    )
                    _render_space(SPACE_EN2FR)

    return demo


if __name__ == "__main__":
    build_demo().launch()
