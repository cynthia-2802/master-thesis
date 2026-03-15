"""Shared helpers."""

import re


def sanitize_feature_name(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_")
