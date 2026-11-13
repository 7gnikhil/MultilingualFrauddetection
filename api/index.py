"""Vercel serverless entry point (repo-root api/ directory).

Vercel only recognises Serverless Functions inside the top-level
``api`` directory of the project root, so the entry point lives here
and delegates to the FastAPI app in ``backend/``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.main import app  # noqa: E402,F401
