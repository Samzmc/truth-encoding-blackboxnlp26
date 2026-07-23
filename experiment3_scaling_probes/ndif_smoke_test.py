"""Minimal remote Llama-3.1-8B test using NNsight and NDIF.

Before running:
  1. Put NDIF_API_KEY and HF_TOKEN in the adjacent .env file.
  2. Ensure the Hugging Face account behind HF_TOKEN has Meta Llama 3.1 access.
  3. Install the repository requirements.

Run from this directory:
  python ndif_smoke_test.py

This script sends one short prompt to NDIF. It does not download or run
Llama weights on the local computer.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_local_env(env_path: Path) -> None:
    """Load simple KEY=VALUE entries without adding a python-dotenv dependency."""
    if not env_path.exists():
        raise FileNotFoundError(f"Missing credentials file: {env_path}")

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env line: {raw_line!r}")

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def require_secret(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value or value.startswith("PASTE_YOUR_"):
        raise RuntimeError(
            f"Set {name} in .env before running this script. "
            "Do not paste the key into Python code or commit it to Git."
        )
    return value


def main() -> None:
    load_local_env(Path(__file__).with_name(".env"))

    ndif_api_key = require_secret("NDIF_API_KEY")
    require_secret("HF_TOKEN")
    from nnsight import CONFIG, LanguageModel

    # The key is read from the local environment, never written into source code.
    CONFIG.set_default_api_key(ndif_api_key)

    print("Creating the lightweight local model definition...")
    # This exact ID is currently listed by ndif_status() as a running model.
    model = LanguageModel("meta-llama/Llama-3.1-8B")

    prompt = "The Eiffel Tower is located in"
    print("Submitting one remote NDIF trace for Llama-3.1-8B...")
    with model.trace(prompt, remote=True):
        # Save the final layer's small sequence tensor. Indexing is performed
        # locally below because this NDIF deployment does not whitelist the
        # internal module NNsight uses to serialize remote tensor slicing.
        hidden_sequence = model.model.layers[-1].output[0].save()

    hidden = hidden_sequence[-1, :]

    print("Success: NDIF returned one hidden representation.")
    print(f"Returned shape: {tuple(hidden.shape)}")
    print(f"Returned dtype: {hidden.dtype}")


if __name__ == "__main__":
    main()
