"""
protocol.py
===========
Defines the canonical output protocol for the BCI backend.

Every outgoing LSL sample sent to Unity MUST be serialised using
``build_message()`` so that all consumers share a single, version-controlled
schema.

Field ownership
---------------
  Code   – always assigned by the Python backend (a BCICode member).
  Event  – the raw Unity event/marker string forwarded verbatim from the
           Unity LSL event log (e.g. ``"Flicker_Start"``, ``"Train_OBJ1"``).
  Detail – the human-readable description string from the Unity log entry
           (e.g. the ActionType or ObjectName field logged by ExperimentLogger).
  Remark – a dict (or plain string) produced by the Python backend containing
           all signal-processing findings, e.g. confidence scores, detected
           frequency, sub-method scores, epoch statistics.

Message schema
--------------
{
    "Code":   int,         # One of BCICode.*  (set by Python)
    "Event":  str,         # Unity log event string  (forwarded from Unity)
    "Detail": str,         # Unity log detail string  (forwarded from Unity)
    "Remark": dict | str   # BCI findings / diagnostics  (set by Python)
}
"""

from __future__ import annotations

import json
from enum import IntEnum
from typing import Union


# ---------------------------------------------------------------------------
# Output protocol codes
# ---------------------------------------------------------------------------

class BCICode(IntEnum):
    """Canonical event codes sent from the Python BCI backend to Unity.

    Range 1xx – SSVEP flicker detection results.
    Range 2xx – Training-phase completion signals.
    Range 3xx – Online prediction results.
    """

    # -- Flicker detection (SSVEP) ------------------------------------------
    FLICKER_DETECTED     = 100
    FLICKER_NOT_DETECTED = 101

    # -- Active Training completion -----------------------------------------
    ACTIVE_OBJ1_TRAIN_COMPLETE  = 201
    ACTIVE_OBJ2_TRAIN_COMPLETE  = 202

    # -- Imagery Training completion ----------------------------------------
    IMAGERY_OBJ1_TRAIN_COMPLETE  = 203
    IMAGERY_OBJ2_TRAIN_COMPLETE  = 204

    # -- Active Online prediction -------------------------------------------
    ACTIVE_OBJ1_PREDICT  = 300
    ACTIVE_OBJ2_PREDICT  = 301

    # -- Imagery Online prediction ------------------------------------------
    IMAGERY_OBJ1_PREDICT  = 302
    IMAGERY_OBJ2_PREDICT  = 303


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

# Type alias for the Remark field
RemarkType = Union[dict, str]


def build_message(
    code: BCICode,
    event: str = "",
    detail: str = "",
    remark: RemarkType = "",
) -> str:
    """Serialise a BCI event into a JSON string ready for LSL transmission.

    **Field ownership contract**

    ``event`` and ``detail`` MUST originate from the Unity LSL event log and
    be forwarded verbatim.  They describe *what Unity did*, not what Python
    computed.  All signal-processing findings (scores, frequencies, etc.) MUST
    be placed in ``remark``.

    Parameters
    ----------
    code:
        A ``BCICode`` member assigned by the Python backend to classify the
        result (e.g. ``BCICode.FLICKER_DETECTED``).
    event:
        The raw Unity event/marker string forwarded from the Unity LSL event
        log (e.g. ``"Flicker_Start"``, ``"Train_OBJ1"``, ``"Dwell_Complete"``).
        Defaults to ``""`` when no Unity log entry is associated with this
        packet (e.g. purely autonomous backend notifications).
    detail:
        The human-readable detail string from the Unity log entry (e.g. the
        ``ActionType`` or ``ObjectName`` field logged by ``ExperimentLogger``).
        Defaults to ``""`` when no Unity log entry is associated.
    remark:
        A dict (preferred) or plain string produced by the Python backend
        containing all signal-processing findings, e.g.::

            {
                "Detected_Frequency": 12.0,
                "Confidence_Score":   0.91,
                "FFT_Score":          0.74,
                "PSD_Score":          0.68,
                "CCA_Score":          0.91,
                "SSVEP_Present":      True,
            }

    Returns
    -------
    str
        A JSON-encoded string conforming to the message schema::

            {
                "Code":   int,         # BCICode  (Python)
                "Event":  str,         # Unity log event string
                "Detail": str,         # Unity log detail string
                "Remark": dict | str   # BCI findings  (Python)
            }

    Examples
    --------
    >>> msg = build_message(
    ...     code=BCICode.FLICKER_DETECTED,
    ...     event="Flicker_Start",
    ...     detail="Door_A",
    ...     remark={"Detected_Frequency": 12.0, "Confidence_Score": 0.91},
    ... )
    >>> import json; print(json.loads(msg)["Event"])
    Flicker_Start
    """
    payload: dict = {
        "Code":   int(code),
        "Event":  event,
        "Detail": detail,
        "Remark": remark,
    }
    return json.dumps(payload)


def parse_message(raw: str) -> dict:
    """Deserialise a JSON string produced by :func:`build_message`.

    Parameters
    ----------
    raw:
        A JSON-encoded string as returned by ``build_message()``.

    Returns
    -------
    dict
        The decoded message dictionary.

    Raises
    ------
    ValueError
        If ``raw`` is not valid JSON or does not contain a ``"Code"`` key.
    """
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid BCI message JSON: {exc}") from exc

    if "Code" not in msg:
        raise ValueError("BCI message is missing required 'Code' field.")

    return msg
