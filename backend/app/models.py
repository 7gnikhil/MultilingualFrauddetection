"""Request/response schemas and validation for feedback events."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .privacy import find_raw_pii

LANGUAGE_ALIASES = {
    "en": "en",
    "english": "en",
    "hi": "hi",
    "hindi": "hi",
    "hinglish": "hi",
    "te": "te",
    "telugu": "te",
}

LABEL_ALIASES = {
    "fraud": "fraud",
    "yes": "fraud",
    "1": "fraud",
    "scam": "fraud",
    "safe": "safe",
    "ham": "safe",
    "no": "safe",
    "0": "safe",
    "legit": "safe",
}


def normalize_language(value: Any) -> str:
    key = str(value).strip().lower()
    if key not in LANGUAGE_ALIASES:
        raise ValueError(
            f"unsupported language {value!r}; expected one of en/hi/te "
            "(or english/hindi/telugu)"
        )
    return LANGUAGE_ALIASES[key]


def normalize_label(value: Any) -> str:
    if isinstance(value, bool):
        return "fraud" if value else "safe"
    if isinstance(value, (int, float)):
        if value == 1:
            return "fraud"
        if value == 0:
            return "safe"
        raise ValueError(f"unsupported label {value!r}")
    key = str(value).strip().lower()
    if key not in LABEL_ALIASES:
        raise ValueError(
            f"unsupported label {value!r}; expected fraud/safe (or yes/no, 1/0)"
        )
    return LABEL_ALIASES[key]


def parse_consent(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


class FeedbackEvent(BaseModel):
    """One explicit, user-consented correction from the extension."""

    event_id: str = Field(min_length=8, max_length=64)
    text_anonymized: str = Field(min_length=1, max_length=10000)
    language: str
    predicted_label: str
    predicted_probability: float = Field(ge=0.0, le=1.0)
    corrected_label: str
    # Human-feedback reward: +1 confirms the model, -1 is a correction
    # signal against it; kept in [-1, 1] for later RL-style weighting.
    reward: float = Field(ge=-1.0, le=1.0)
    model_version: str = Field(min_length=1, max_length=64)
    extension_version: str = Field(min_length=1, max_length=64)
    client_created_at: datetime
    consent: bool

    @field_validator("event_id")
    @classmethod
    def _event_id_shape(cls, v: str) -> str:
        v = v.strip()
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        if not v or any(ch not in allowed for ch in v):
            raise ValueError("event_id may only contain letters, digits, '-' and '_'")
        return v

    @field_validator("language", mode="before")
    @classmethod
    def _language(cls, v: Any) -> str:
        return normalize_language(v)

    @field_validator("predicted_label", "corrected_label", mode="before")
    @classmethod
    def _labels(cls, v: Any) -> str:
        return normalize_label(v)

    @field_validator("consent", mode="before")
    @classmethod
    def _consent_parse(cls, v: Any) -> bool:
        return parse_consent(v)

    @model_validator(mode="after")
    def _consent_and_privacy(self) -> "FeedbackEvent":
        if self.consent is not True:
            raise ValueError(
                "consent=true is required: feedback is collected only with "
                "explicit user opt-in"
            )
        text = self.text_anonymized.strip()
        if not text:
            raise ValueError("text_anonymized must not be blank")
        hit = find_raw_pii(text)
        if hit is not None:
            raise ValueError(
                f"text_anonymized appears to contain a raw {hit}; "
                "anonymize personal data client-side before sending"
            )
        return self

    def to_record(self, received_at: datetime) -> dict[str, Any]:
        record = self.model_dump(mode="json")
        record["received_at"] = received_at.isoformat()
        return record


class FeedbackAccepted(BaseModel):
    status: str  # "accepted" | "duplicate"
    event_id: str


class BatchResult(BaseModel):
    accepted: int
    duplicates: int
    rejected: list[dict[str, Any]]
