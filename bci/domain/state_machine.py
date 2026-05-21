from __future__ import annotations
from typing import Optional

class BCIState:
    IDLE = "IDLE"
    SSVEP_TEST = "SSVEP_TEST"
    TRAIN_ACTIVE_OBJ1 = "TRAIN_ACTIVE_OBJ1"
    TRAIN_ACTIVE_OBJ2 = "TRAIN_ACTIVE_OBJ2"
    TRAIN_IMAGERY_OBJ1 = "TRAIN_IMAGERY_OBJ1"
    TRAIN_IMAGERY_OBJ2 = "TRAIN_IMAGERY_OBJ2"
    PREDICT_ACTIVE = "PREDICT_ACTIVE"
    PREDICT_IMAGERY = "PREDICT_IMAGERY"
    PREDICT_MIXED = "PREDICT_MIXED"

TRANSITIONS = {
    "Flicker_Start": BCIState.SSVEP_TEST,
    "Flicker_End": BCIState.IDLE,
    "TAD1S": BCIState.TRAIN_ACTIVE_OBJ1,
    "TAD1E": BCIState.IDLE,
    "Training_Active_Door1_Flicker_Start": BCIState.TRAIN_ACTIVE_OBJ1,
    "Training_Active_Door1_Flicker_End": BCIState.IDLE,
    "TID1S": BCIState.TRAIN_IMAGERY_OBJ1,
    "TID1E": BCIState.IDLE,
    "TAD2S": BCIState.TRAIN_ACTIVE_OBJ2,
    "TAD2E": BCIState.IDLE,
    "TID2S": BCIState.TRAIN_IMAGERY_OBJ2,
    "TID2E": BCIState.IDLE,
    "Predict_Start_Active": BCIState.PREDICT_ACTIVE,
    "Predict_Start_Imagery": BCIState.PREDICT_IMAGERY,
    "Start_predict": BCIState.PREDICT_MIXED,
    "Predict_End": BCIState.IDLE,
}

class BCIStateMachine:
    def __init__(self, initial_state: str = BCIState.IDLE) -> None:
        self._state = initial_state

    @property
    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        self._state = state

    def get_next_state(self, action: str) -> Optional[str]:
        return TRANSITIONS.get(action)
