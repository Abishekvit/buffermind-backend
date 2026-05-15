# File: models.py

from pydantic import BaseModel
from typing import List, Optional

class PredictionRequest(BaseModel):
    walking: bool
    weak_signal: bool
    repeated_playback: bool
    gps_moving: bool
    playback_duration_minutes: float

class PredictionResponse(BaseModel):
    disconnect_probability: float
    confidence: float
    should_buffer: bool
    buffer_minutes: int
    reason: str