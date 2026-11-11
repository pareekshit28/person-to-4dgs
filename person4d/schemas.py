from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class JobOptions(BaseModel):
    preset: Literal["preview", "balanced", "quality"] = "quality"
    views: Literal[12, 24, 48] | None = None
    start_time: float = Field(default=0, ge=0, le=86400)
    target_fps: int = Field(default=24, ge=12, le=30)
    steps: int | None = Field(default=None, ge=1000, le=100000)
    seed: int = Field(default=42, ge=0, le=2147483647)

    @field_validator("start_time")
    @classmethod
    def finite_start(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Start time must be finite")
        return value


class Login(BaseModel):
    token: str = Field(max_length=512)
