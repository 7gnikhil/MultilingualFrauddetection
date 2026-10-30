"""Vercel serverless entry point.

Vercel's Python runtime serves the ASGI ``app`` exported from
``api/index.py``. All logic lives in ``app/main.py`` so the same app
runs unchanged under a local uvicorn server.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app  # noqa: E402,F401
