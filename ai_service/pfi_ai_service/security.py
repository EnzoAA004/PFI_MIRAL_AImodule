from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import HTTPException


SENSITIVE_KEYS = {
    "path",
    "sourcePath",
    "inputPath",
    "input_path",
    "imagePath",
    "maskPath",
    "confidencePath",
    "overlayPath",
    "maskPreviewPath",
    "overlay_path",
    "outputFiles",
    "outputPath",
    "checkpointPath",
    "modelPath",
    "token",
    "accessToken",
    "refreshToken",
    "authorization",
    "password",
    "secret",
}

SAFE_PATH_KEYS = {"path", "method"}
PATH_LIKE = re.compile(
    r"([A-Za-z]:[\\/]|/tmp\b|/var\b|/app\b|/content\b|/models\b|/outputs\b|\\\\|localhost|host\.docker\.internal|cloudflare)",
    re.IGNORECASE,
)
TOKEN_LIKE = re.compile(r"(bearer\s+[A-Za-z0-9._~+/=-]+|eyJ[A-Za-z0-9._~+/=-]+)", re.IGNORECASE)
SAFE_CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def require_public_input_reference(case_id: str, plane: str, input_id: str | None, input_path: str | None) -> None:
    """Public HTTP inference uses registered IDs; only a fixed, non-filesystem demo alias remains."""
    normalized_id = input_id.strip() if isinstance(input_id, str) else ""
    normalized_path = input_path.strip() if isinstance(input_path, str) else ""
    if normalized_id:
        if normalized_path:
            raise HTTPException(status_code=400, detail="Enviar solamente inputId, no inputPath.")
        return
    safe_case = isinstance(case_id, str) and SAFE_CASE_ID.fullmatch(case_id.strip()) is not None
    aliases = {f"demo/{case_id}"}
    if plane in {"sagittal", "axial"}:
        aliases.add(f"demo/{case_id}/{plane}")
    if safe_case and normalized_path in aliases:
        return
    raise HTTPException(
        status_code=400,
        detail="inputId es obligatorio; las rutas del filesystem solo se admiten dentro del flujo interno.",
    )


def sanitize_public_payload(value: Any, *, keep_error_path: bool = False) -> Any:
    if isinstance(value, Path):
        return None
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            if _drop_key(text_key, keep_error_path=keep_error_path):
                continue
            clean = sanitize_public_payload(item, keep_error_path=keep_error_path)
            if clean is not None:
                sanitized[text_key] = clean
        return sanitized
    if isinstance(value, list):
        return [
            clean
            for item in value
            if (clean := sanitize_public_payload(item, keep_error_path=keep_error_path)) is not None
        ]
    if isinstance(value, tuple):
        return [
            clean
            for item in value
            if (clean := sanitize_public_payload(item, keep_error_path=keep_error_path)) is not None
        ]
    if isinstance(value, str):
        return sanitize_public_text(value)
    return value


def sanitize_public_text(value: str) -> str:
    cleaned = TOKEN_LIKE.sub("[redacted-token]", value)
    if PATH_LIKE.search(cleaned):
        return "[redacted-internal-detail]"
    return cleaned


def safe_exception_type(exc: Exception) -> str:
    return type(exc).__name__


def _drop_key(key: str, *, keep_error_path: bool) -> bool:
    if keep_error_path and key in SAFE_PATH_KEYS:
        return False
    normalized = key.lower()
    return key in SENSITIVE_KEYS or normalized.endswith("token") or "secret" in normalized or "password" in normalized
