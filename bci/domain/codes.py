from __future__ import annotations
from enum import IntEnum

class BCICode(IntEnum):
    # 1xx - SSVEP flicker detection
    FLICKER_DETECTED     = 100
    FLICKER_NOT_DETECTED = 101

    # 2xx - Training completion
    ACTIVE_OBJ1_TRAIN_COMPLETE  = 201
    ACTIVE_OBJ2_TRAIN_COMPLETE  = 202

    # 3xx - Online prediction
    ACTIVE_OBJ1_PREDICT  = 300
    ACTIVE_OBJ2_PREDICT  = 301
