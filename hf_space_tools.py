"""Helpers for publishing the Gradio app to a Hugging Face Space."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import create_repo, upload_file, upload_folder, whoami

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent



def _get_hf_token() -> str | None:
    """Return a configured Hugging Face token or ``None`` if unavailable."""
    token = os.getenv("HF_TOKEN", "").strip()
    if not token or token == "your_hugging_face_token_here":
        return None
    return token



def _normalize_repo_id(space_name: str, token: str) -> str:
    """Normalize a target Space name to a full repo id.

    Args:
        space_name: Either ``space-name`` or ``owner/space-name``.
        token: Hugging Face token.

    Returns:
        A repo id suitable for `huggingface_hub`, such as ``owner/space-name``.
    """
    cleaned = space_name.strip()
    if "/" in cleaned:
        return cleaned
    username = whoami(token=token)["name"]
    return f"{username}/{cleaned}"



def create_or_update_space(space_name: str) -> str:
    """Create or update a Gradio Space for this project.

    Args:
        space_name: Target Space name or full repo id.

    Returns:
        A summary message with the resulting Space URL.
    """
    token = _get_hf_token()
    if not token:
        return (
            "HF_TOKEN is not configured. Add a Hugging Face write token to your environment or `.env` file "
            "before creating a Space."
        )

    repo_id = _normalize_repo_id(space_name, token)
    create_repo(
        repo_id=repo_id,
        token=token,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )

    upload_file(
        path_or_fileobj=str(PROJECT_ROOT / "gradio_app.py"),
        path_in_repo="app.py",
        repo_id=repo_id,
        repo_type="space",
        token=token,
    )
    upload_file(
        path_or_fileobj=str(PROJECT_ROOT / "requirements.txt"),
        path_in_repo="requirements.txt",
        repo_id=repo_id,
        repo_type="space",
        token=token,
    )
    upload_folder(
        folder_path=str(PROJECT_ROOT / "src"),
        path_in_repo="src",
        repo_id=repo_id,
        repo_type="space",
        token=token,
        ignore_patterns=["**/__pycache__", "**/*.pyc"],
    )

    readme_path = PROJECT_ROOT / "README.md"
    if readme_path.exists():
        upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="space",
            token=token,
        )

    return f"Space updated successfully: https://huggingface.co/spaces/{repo_id}"



def main() -> None:
    """CLI entrypoint for creating or updating a Hugging Face Space."""
    parser = argparse.ArgumentParser(description="Create or update a Hugging Face Gradio Space.")
    parser.add_argument(
        "space_name",
        help="Target Space name or full repo id (for example `my-space` or `username/my-space`).",
    )
    args = parser.parse_args()
    print(create_or_update_space(args.space_name))


if __name__ == "__main__":
    main()
