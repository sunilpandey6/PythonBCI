from __future__ import annotations

import json
from enum import IntEnum
from typing import Union

class BCICode(IntEnum):
    # 1xx - SSVEP flicker detection
    FLICKER_DETECTED     = 100
    FLICKER_NOT_DETECTED = 101

    # 2xx - Training completion
    ACTIVE_OBJ1_TRAIN_COMPLETE  = 201
    ACTIVE_OBJ2_TRAIN_COMPLETE  = 202
    IMAGERY_OBJ1_TRAIN_COMPLETE = 203
    IMAGERY_OBJ2_TRAIN_COMPLETE = 204

    # 3xx - Online prediction
    ACTIVE_OBJ1_PREDICT  = 300
    ACTIVE_OBJ2_PREDICT  = 301
    IMAGERY_OBJ1_PREDICT = 302
    IMAGERY_OBJ2_PREDICT = 303


RemarkType = Union[dict, str]


def build_message(
    code: BCICode,
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
