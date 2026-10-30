"""Feedback storage.

Two stores share one interface:

* ``LocalJSONLStore`` — appends events to ``feedback.jsonl`` in a local
  data directory. Right for development and for a single-container host.
  Note: on Vercel serverless the filesystem is ephemeral, so this store
  is a dev/fallback mode there, not durable storage.

* ``GitHubInboxStore`` — commits one JSON file per event into a PRIVATE
  GitHub repository (default branch ``feedback-inbox``) through the
  GitHub Contents API. A GitHub Actions retraining job can then read the
  inbox branch, preprocess the events, and open a model-update PR. The
  token is read from the environment and never logged or returned.
"""

from __future__ import annotations

import base64
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

GITHUB_API = "https://api.github.com"


class LocalJSONLStore:
    def __init__(self, data_dir: Path, filename: str = "feedback.jsonl") -> None:
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / filename
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        self._load_seen()

    def _load_seen(self) -> None:
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._seen.add(json.loads(line)["event_id"])
                    except (ValueError, KeyError):
                        continue

    def add(self, record: dict[str, Any]) -> bool:
        """Store a record. Returns False if the event_id is a duplicate."""
        with self._lock:
            event_id = record["event_id"]
            if event_id in self._seen:
                return False
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._seen.add(event_id)
            return True

    def list_events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except ValueError:
                        continue
        return events


class GitHubInboxStore:
    """Commits feedback events to a private GitHub repo via Contents API."""

    def __init__(
        self,
        token: str,
        repo: str,
        branch: str = "feedback-inbox",
        client: httpx.Client | None = None,
    ) -> None:
        self._token = token
        self.repo = repo
        self.branch = branch
        self._client = client or httpx.Client(timeout=20.0)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    @staticmethod
    def _path_for(record: dict[str, Any]) -> str:
        day = str(record.get("client_created_at", ""))[:10] or datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d")
        return f"feedback/inbox/{day}/{record['event_id']}.json"

    def _get_file(self, path: str) -> httpx.Response:
        return self._client.get(
            f"{GITHUB_API}/repos/{self.repo}/contents/{path}",
            params={"ref": self.branch},
            headers=self._headers(),
        )

    def add(self, record: dict[str, Any]) -> bool:
        """Commit one event file. Returns False if it already exists."""
        path = self._path_for(record)
        existing = self._get_file(path)
        if existing.status_code == 200:
            return False
        payload = {
            "message": f"feedback: {record['event_id']}",
            "content": base64.b64encode(
                json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8")
            ).decode("ascii"),
            "branch": self.branch,
        }
        resp = self._client.put(
            f"{GITHUB_API}/repos/{self.repo}/contents/{path}",
            json=payload,
            headers=self._headers(),
        )
        if resp.status_code == 422:  # created concurrently -> duplicate
            return False
        resp.raise_for_status()
        return True

    def list_events(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Read back events via the git tree (used by the CSV export)."""
        tree = self._client.get(
            f"{GITHUB_API}/repos/{self.repo}/git/trees/{self.branch}",
            params={"recursive": "1"},
            headers=self._headers(),
        )
        tree.raise_for_status()
        events: list[dict[str, Any]] = []
        for item in tree.json().get("tree", []):
            path = item.get("path", "")
            if not (path.startswith("feedback/inbox/") and path.endswith(".json")):
                continue
            resp = self._get_file(path)
            if resp.status_code != 200:
                continue
            raw = base64.b64decode(resp.json()["content"])
            events.append(json.loads(raw.decode("utf-8")))
            if len(events) >= limit:
                break
        return events
