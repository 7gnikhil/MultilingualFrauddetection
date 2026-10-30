"""Fraud Detector v4 — feedback & model-manifest API.

Endpoints
---------
GET  /health                     liveness probe
GET  /v1/model/latest            manifest of the model the extension should run
POST /v1/feedback                one consented, anonymized feedback event
POST /v1/feedback/batch          many events: JSON rows, a JSON list, or a CSV file upload
GET  /v1/feedback/export.csv     stored events as CSV (ML-ready), admin-key protected

Design notes
------------
* Feedback is explicit opt-in only: ``consent`` must be true, and text
  must already be anonymized. A server-side PII backstop rejects events
  that still look like they carry raw phone/OTP/email/UPI/URL values.
* Nothing here retrains or deploys a model. Events land in an inbox
  (local JSONL in dev, or a private GitHub branch in production); the
  GitHub Actions pipeline preprocesses them and only promotes a
  validated challenger model.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import ValidationError

from .models import BatchResult, FeedbackAccepted, FeedbackEvent
from .settings import Settings, get_settings
from .storage import GitHubInboxStore, LocalJSONLStore

EXPORT_COLUMNS = [
    "event_id",
    "text_anonymized",
    "language",
    "predicted_label",
    "predicted_probability",
    "corrected_label",
    "reward",
    "model_version",
    "extension_version",
    "client_created_at",
    "received_at",
    "consent",
]


class RateLimiter:
    """Simple in-memory sliding-window limiter (per client IP)."""

    def __init__(self, limit_per_minute: int) -> None:
        self.limit = max(limit_per_minute, 1)
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] < now - 60.0:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True


def load_manifest(settings: Settings) -> dict[str, Any]:
    """Manifest priority: MODEL_MANIFEST_URL -> local JSON file -> env/defaults."""
    if settings.model_manifest_url:
        try:
            resp = httpx.get(settings.model_manifest_url, timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and "version" in data:
                return data
        except Exception:
            pass  # fall through to the local/default manifest
    if settings.model_manifest_path.exists():
        try:
            data = json.loads(settings.model_manifest_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except ValueError:
            pass
    return {
        "version": settings.model_version,
        "model_url": settings.model_url,
        "model_sha256": settings.model_sha256,
        "config_url": settings.model_config_url,
        "config_sha256": settings.model_config_sha256,
    }


def build_store(settings: Settings) -> LocalJSONLStore | GitHubInboxStore:
    if settings.use_github_store:
        assert settings.github_token and settings.feedback_repo
        return GitHubInboxStore(
            token=settings.github_token,
            repo=settings.feedback_repo,
            branch=settings.feedback_branch,
        )
    return LocalJSONLStore(settings.data_dir)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings(fresh=True)
    store = build_store(settings)
    limiter = RateLimiter(settings.rate_limit_per_minute)

    app = FastAPI(title="Fraud Detector Feedback API", version="1.0.0")
    app.state.settings = settings
    app.state.store = store

    @app.middleware("http")
    async def limits_middleware(request: Request, call_next):
        if request.method == "POST":
            length = request.headers.get("content-length")
            if length is not None:
                try:
                    if int(length) > settings.max_body_bytes:
                        return JSONResponse(
                            status_code=413,
                            content={"detail": "request body too large"},
                        )
                except ValueError:
                    pass
            if request.url.path.startswith("/v1/feedback"):
                client = request.client.host if request.client else "unknown"
                if not limiter.allow(client):
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "rate limit exceeded; slow down"},
                    )
        return await call_next(request)

    # -- basic routes ----------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "storage": "github" if settings.use_github_store else "local",
            "time": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/v1/model/latest")
    def model_latest() -> dict[str, Any]:
        return load_manifest(settings)

    # -- feedback --------------------------------------------------------

    def persist(event: FeedbackEvent) -> bool:
        record = event.to_record(datetime.now(timezone.utc))
        try:
            return store.add(record)
        except httpx.HTTPError as exc:  # GitHub inbox unreachable
            raise HTTPException(
                status_code=502, detail="feedback storage unavailable"
            ) from exc

    @app.post("/v1/feedback", response_model=FeedbackAccepted, status_code=202)
    def submit_feedback(event: FeedbackEvent) -> FeedbackAccepted:
        if len(event.text_anonymized) > settings.max_text_chars:
            raise HTTPException(
                status_code=422,
                detail=f"text_anonymized exceeds {settings.max_text_chars} characters",
            )
        added = persist(event)
        return FeedbackAccepted(
            status="accepted" if added else "duplicate", event_id=event.event_id
        )

    @app.post("/v1/feedback/batch", response_model=BatchResult)
    async def submit_feedback_batch(request: Request) -> BatchResult:
        content_type = request.headers.get("content-type", "")
        raw_rows: list[Any] = []

        if "multipart/form-data" in content_type:
            form = await request.form()
            upload = form.get("file")
            if upload is None or not hasattr(upload, "read"):
                raise HTTPException(
                    status_code=422, detail="multipart upload must include a 'file' field"
                )
            raw_bytes = await upload.read()
            raw_rows = _rows_from_csv(raw_bytes)
        elif "text/csv" in content_type:
            raw_rows = _rows_from_csv(await request.body())
        else:
            try:
                payload = await request.json()
            except ValueError:
                raise HTTPException(status_code=422, detail="body must be JSON or CSV")
            if isinstance(payload, list):
                raw_rows = payload
            elif isinstance(payload, dict) and isinstance(
                payload.get("rows", payload.get("events")), list
            ):
                raw_rows = payload.get("rows", payload.get("events"))
            else:
                raise HTTPException(
                    status_code=422,
                    detail="expected a JSON list, or an object with a 'rows' list",
                )

        if len(raw_rows) > settings.max_batch_rows:
            raise HTTPException(
                status_code=413,
                detail=f"batch exceeds {settings.max_batch_rows} rows",
            )

        accepted = duplicates = 0
        rejected: list[dict[str, Any]] = []
        for idx, row in enumerate(raw_rows):
            try:
                if isinstance(row, dict):
                    row = _normalize_batch_row(row)
                event = FeedbackEvent(**row) if isinstance(row, dict) else None
                if event is None:
                    raise ValueError("row is not an object")
                if len(event.text_anonymized) > settings.max_text_chars:
                    raise ValueError(
                        f"text_anonymized exceeds {settings.max_text_chars} characters"
                    )
            except (ValidationError, ValueError) as exc:
                rejected.append({"row": idx, "error": _short_error(exc)})
                continue
            if persist(event):
                accepted += 1
            else:
                duplicates += 1
        return BatchResult(accepted=accepted, duplicates=duplicates, rejected=rejected)

    # -- export ----------------------------------------------------------

    @app.get("/v1/feedback/export.csv")
    def export_csv(
        x_admin_api_key: str | None = Header(default=None),
    ) -> PlainTextResponse:
        if settings.admin_api_key:
            if not x_admin_api_key or not secrets.compare_digest(
                x_admin_api_key, settings.admin_api_key
            ):
                raise HTTPException(status_code=401, detail="invalid or missing admin API key")
        try:
            events = store.list_events()
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502, detail="feedback storage unavailable"
            ) from exc
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for event in events:
            writer.writerow(event)
        return PlainTextResponse(
            buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=feedback_export.csv"},
        )

    return app


def _normalize_batch_row(row: dict[str, Any]) -> dict[str, Any]:
    """Make extension CSV exports importable through the batch API.

    The popup's corrections CSV (columns: timestamp, language,
    corrected_label, model_probability_before, probability_after,
    message/message_anonymized) lacks the bookkeeping fields the
    single-event endpoint requires. Fill them deterministically:
    a stable event_id hashed from the row itself (so re-uploading the
    same CSV dedupes), the predicted label as the opposite of the
    correction (a correction means the model said the other thing),
    reward -1, and csv-import provenance markers. Uploading a CSV is
    itself the user's explicit act, so consent defaults to true here;
    the single-event endpoint stays strict.
    """
    row = dict(row)
    if not row.get("predicted_probability") and row.get("model_probability_before"):
        row["predicted_probability"] = row["model_probability_before"]
    if not row.get("predicted_label") and row.get("corrected_label"):
        corrected = str(row["corrected_label"]).strip().lower()
        row["predicted_label"] = (
            "safe" if corrected in {"yes", "1", "fraud", "scam", "true"} else "fraud"
        )
    if not row.get("event_id"):
        basis = "|".join(
            str(row.get(k, ""))
            for k in ("timestamp", "language", "corrected_label", "message", "message_anonymized", "text_anonymized")
        )
        row["event_id"] = "csv-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]
    row.setdefault("reward", -1)
    row.setdefault("model_version", "csv-import")
    row.setdefault("extension_version", "csv-import")
    row.setdefault("consent", True)
    if not row.get("predicted_probability"):
        row["predicted_probability"] = 0.5
    return row


def _rows_from_csv(raw: bytes) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=422, detail="CSV file must be UTF-8 encoded")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise HTTPException(status_code=422, detail="CSV file has no header row")
    return [dict(row) for row in reader]


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        messages = [str(e.get("msg", "invalid")) for e in exc.errors()]
        return "; ".join(messages)[:300]
    return str(exc)[:300]


app = create_app()
