# llm/prompts/utils.py

import json
import re
from typing import Any


def format_value(value: Any) -> str:
    """
    Convert model output values into a readable prompt string.
    """
    if value in (None, "", [], {}):
        return "없음"

    if isinstance(value, str):
        return value

    return json.dumps(value, ensure_ascii=False)


def sort_image_roles(roles: list[str]) -> list[str]:
    """
    Sort image roles in a stable and human-readable order.

    Examples:
    - seg3d_1, seg3d_2, seg3d_10
    - bbox_overlay
    """
    def sort_key(role: str):
        match = re.fullmatch(r"(.+?)_(\d+)", role)
        if match:
            return match.group(1), int(match.group(2))
        return role, -1

    return sorted(roles, key=sort_key)


def to_pretty_json(data: Any) -> str:
    """
    Convert Python objects into pretty JSON for prompt context.
    """
    return json.dumps(data, ensure_ascii=False, indent=2)