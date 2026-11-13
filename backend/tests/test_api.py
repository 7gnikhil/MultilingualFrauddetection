"""API tests: health, manifest, feedback validation, privacy, export."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings, get_settings


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    base = get_settings(fresh=True)
    base.data_dir = tmp_path / "data"
    base.feedback_storage = "local"
    base.github_token = None
    base.feedback_repo = None
    base.admin_api_key = None
    base.model_manifest_url = None
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "4.0.0-test",
                "model_url": "https://example.invalid/model.onnx",
                "model_sha256": "a" * 64,
                "config_url": "https://example.invalid/config.json",
                "config_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    base.model_manifest_path = manifest
    return base


@pytest.fixture()
def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))


def valid_event(**overrides) -> dict:
    event = {
        "event_id": "evt-2026-0001",
        "text_anonymized": (
            "Your account will be blocked today. Share the [OTP] and pay "
            "via [UPI] to keep it active."
        ),
        "language": "en",
        "predicted_label": "safe",
        "predicted_probability": 0.12,
        "corrected_label": "fraud",
        "reward": -1.0,
        "model_version": "4.0.0-test",
        "extension_version": "2.0.0",
        "client_created_at": "2026-10-30T10:00:00+05:30",
        "consent": True,
    }
    event.update(overrides)
    return event


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["storage"] == "local"


def test_model_manifest(client: TestClient) -> None:
    resp = client.get("/v1/model/latest")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == "4.0.0-test"
    assert body["model_url"].endswith("model.onnx")
    assert len(body["model_sha256"]) == 64
    assert body["config_url"].endswith("config.json")


def test_valid_feedback_accepted(client: TestClient, settings: Settings) -> None:
    resp = client.post("/v1/feedback", json=valid_event())
    assert resp.status_code == 202
    assert resp.json() == {"status": "accepted", "event_id": "evt-2026-0001"}
    stored = (settings.data_dir / "feedback.jsonl").read_text(encoding="utf-8")
    record = json.loads(stored.strip())
    assert record["corrected_label"] == "fraud"
    assert record["language"] == "en"


def test_duplicate_event_id_is_idempotent(client: TestClient) -> None:
    assert client.post("/v1/feedback", json=valid_event()).status_code == 202
    resp = client.post("/v1/feedback", json=valid_event())
    assert resp.status_code == 202
    assert resp.json()["status"] == "duplicate"


def test_language_and_label_aliases_normalized(client: TestClient) -> None:
    resp = client.post(
        "/v1/feedback",
        json=valid_event(
            event_id="evt-2026-0002",
            language="Telugu",
            predicted_label="No",
            corrected_label="Yes",
        ),
    )
    assert resp.status_code == 202


def test_feedback_without_consent_rejected(client: TestClient) -> None:
    resp = client.post("/v1/feedback", json=valid_event(consent=False))
    assert resp.status_code == 422


def test_invalid_probability_rejected(client: TestClient) -> None:
    resp = client.post(
        "/v1/feedback", json=valid_event(predicted_probability=1.5)
    )
    assert resp.status_code == 422


def test_invalid_language_rejected(client: TestClient) -> None:
    resp = client.post("/v1/feedback", json=valid_event(language="fr"))
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "raw_text",
    [
        "Call me on 9876543210 to claim your prize",  # raw phone
        "Your OTP is 482913, do not share it",  # raw OTP code
        "Send money to winner@okaxis now",  # raw UPI id
        "Pay at https://evil.example.com/claim today",  # raw URL
        "Write to helpdesk@examplemail.com for refund",  # raw email
    ],
)
def test_raw_pii_rejected(client: TestClient, raw_text: str) -> None:
    resp = client.post(
        "/v1/feedback",
        json=valid_event(event_id="evt-pii-00001", text_anonymized=raw_text),
    )
    assert resp.status_code == 422


def test_batch_json_rows(client: TestClient) -> None:
    rows = [
        valid_event(event_id="evt-batch-001"),
        valid_event(event_id="evt-batch-002", language="hindi"),
        valid_event(event_id="evt-batch-003", consent=False),
        valid_event(event_id="evt-batch-001"),  # duplicate of first row
    ]
    resp = client.post("/v1/feedback/batch", json={"rows": rows})
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] == 2
    assert body["duplicates"] == 1
    assert len(body["rejected"]) == 1
    assert body["rejected"][0]["row"] == 2


def test_batch_csv_upload(client: TestClient) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(valid_event().keys()))
    writer.writeheader()
    writer.writerow(valid_event(event_id="evt-csv-00001"))
    resp = client.post(
        "/v1/feedback/batch",
        files={"file": ("feedback.csv", buffer.getvalue(), "text/csv")},
    )
    assert resp.status_code == 200
    assert resp.json()["accepted"] == 1


def test_export_csv_open_in_dev_mode(client: TestClient) -> None:
    client.post("/v1/feedback", json=valid_event())
    resp = client.get("/v1/feedback/export.csv")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]
    rows = list(csv.DictReader(io.StringIO(resp.text)))
    assert len(rows) == 1
    assert rows[0]["event_id"] == "evt-2026-0001"
    assert rows[0]["corrected_label"] == "fraud"
    assert "label" not in rows[0] or rows[0].get("corrected_label")


def test_export_csv_requires_admin_key_when_configured(
    settings: Settings,
) -> None:
    settings.admin_api_key = "test-admin-key"
    client = TestClient(create_app(settings))
    client.post("/v1/feedback", json=valid_event())
    assert client.get("/v1/feedback/export.csv").status_code == 401
    wrong = client.get(
        "/v1/feedback/export.csv", headers={"X-Admin-API-Key": "nope"}
    )
    assert wrong.status_code == 401
    ok = client.get(
        "/v1/feedback/export.csv", headers={"X-Admin-API-Key": "test-admin-key"}
    )
    assert ok.status_code == 200


def test_oversized_text_rejected(client: TestClient, settings: Settings) -> None:
    settings.max_text_chars = 20
    resp = client.post("/v1/feedback", json=valid_event())
    assert resp.status_code == 422


def test_extension_legacy_payload_accepted(client: TestClient) -> None:
    """The first v4 extension build sent anonymized_text + timestamp."""
    event = valid_event(event_id="evt-legacy-0001")
    event["anonymized_text"] = event.pop("text_anonymized")
    event["timestamp"] = event.pop("client_created_at")
    resp = client.post("/v1/feedback", json=event)
    assert resp.status_code == 202
    assert resp.json()["status"] == "accepted"


def test_popup_csv_batch_import_and_dedupe(client: TestClient) -> None:
    csv_body = (
        "timestamp,language,corrected_label,model_probability_before,probability_after,message_anonymized\r\n"
        "2026-10-30T05:00:00.000Z,English,Yes,0.1200,0.9500,"
        "\"Your account will be blocked today. Share the [OTP] and pay via [UPI].\"\r\n"
    )
    resp = client.post(
        "/v1/feedback/batch", content=csv_body, headers={"Content-Type": "text/csv"}
    )
    assert resp.status_code == 200
    assert resp.json()["accepted"] == 1
    again = client.post(
        "/v1/feedback/batch", content=csv_body, headers={"Content-Type": "text/csv"}
    )
    assert again.json()["duplicates"] == 1


def test_v3_csv_batch_import(client: TestClient) -> None:
    csv_body = (
        "timestamp,language,corrected_label,model_probability_before,probability_after,message\r\n"
        "2026-10-30T05:05:00.000Z,Telugu,No,0.9900,0.1000,"
        "\"Debited Rs [OTP] from your account. Do not share your OTP. Bank never asks for OTP.\"\r\n"
    )
    resp = client.post(
        "/v1/feedback/batch", content=csv_body, headers={"Content-Type": "text/csv"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["accepted"] == 1, body
