from __future__ import annotations
import json
from typing import Union

RemarkType = Union[dict, str]

def build_message(
    code: int,
    event: str = "",
    detail: str = "",
    remark: RemarkType = "",
) -> str:
    payload: dict = {
        "Code":   int(code),
        "Event":  event,
        "Detail": detail,
        "Remark": remark,
    }
    return json.dumps(payload)

def parse_message(raw: str) -> dict:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid BCI message JSON: {exc}") from exc

    if "Code" not in msg:
        raise ValueError("BCI message is missing required 'Code' field.")

    return msg
