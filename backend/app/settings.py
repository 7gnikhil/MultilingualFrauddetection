"""Runtime settings, read from environment variables.

Nothing secret is hard-coded. Every value can be overridden with an
environment variable (see .env.example). Settings are re-read whenever
``get_settings()`` is called with ``fresh=True`` so tests (and serverless
cold starts that rotate env vars) always see current values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass
class Settings:
    # Storage
    data_dir: Path
    feedback_storage: str  # "auto" | "local" | "github"
    github_token: str | None
    feedback_repo: str | None  # "owner/name"
    feedback_branch: str

    # Admin / security
    admin_api_key: str | None

    # Model manifest
    model_manifest_path: Path
    model_manifest_url: str | None
    model_version: str
    model_url: str
    model_config_url: str
    model_sha256: str
    model_config_sha256: str

    # Limits
    max_text_chars: int
    max_batch_rows: int
    max_body_bytes: int
    rate_limit_per_minute: int

    @property
    def github_configured(self) -> bool:
        return bool(self.github_token and self.feedback_repo)

    @property
    def use_github_store(self) -> bool:
        if self.feedback_storage == "github":
            return self.github_configured
        if self.feedback_storage == "local":
            return False
        return self.github_configured  # "auto"


def get_settings(fresh: bool = False) -> Settings:
    if os.environ.get("DATA_DIR"):
        data_dir = Path(os.environ["DATA_DIR"])
    elif os.environ.get("VERCEL"):
        # Serverless filesystems are read-only outside /tmp; the local
        # store would 500 on every write if it used the bundled path.
        # (Durable production storage is the GitHub inbox store.)
        data_dir = Path("/tmp/fraud-detector-data")
    else:
        data_dir = BACKEND_DIR / "data"
    manifest_default = BACKEND_DIR / "model_manifest.json"
    return Settings(
        data_dir=data_dir,
        feedback_storage=os.environ.get("FEEDBACK_STORAGE", "auto").lower(),
        github_token=os.environ.get("GITHUB_FEEDBACK_TOKEN") or None,
        feedback_repo=os.environ.get("FEEDBACK_REPO") or None,
        feedback_branch=os.environ.get("FEEDBACK_BRANCH", "feedback-inbox"),
        admin_api_key=os.environ.get("ADMIN_API_KEY") or None,
        model_manifest_path=Path(
            os.environ.get("MODEL_MANIFEST_PATH", str(manifest_default))
        ),
        model_manifest_url=os.environ.get("MODEL_MANIFEST_URL") or None,
        model_version=os.environ.get("MODEL_VERSION", "0.0.0-dev"),
        model_url=os.environ.get("MODEL_URL", ""),
        model_config_url=os.environ.get("MODEL_CONFIG_URL", ""),
        model_sha256=os.environ.get("MODEL_SHA256", ""),
        model_config_sha256=os.environ.get("MODEL_CONFIG_SHA256", ""),
        max_text_chars=_int_env("MAX_TEXT_CHARS", 2000),
        max_batch_rows=_int_env("MAX_BATCH_ROWS", 500),
        max_body_bytes=_int_env("MAX_BODY_BYTES", 1_048_576),
        rate_limit_per_minute=_int_env("RATE_LIMIT_PER_MINUTE", 120),
    )
