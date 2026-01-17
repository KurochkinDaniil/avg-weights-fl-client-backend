"""Pydantic models for API requests/responses."""
from pydantic import BaseModel, Field
from typing import List


class CoordinatePoint(BaseModel):
    """Single coordinate point in swipe gesture."""
    x: float = Field(..., ge=0.0, le=1080.0, description="Pixel x coordinate (0..1080)")
    y: float = Field(..., ge=0.0, le=631.0, description="Pixel y coordinate (0..631)")
    t: float = Field(..., ge=0.0, description="Timestamp or relative time")


class SwipeRequest(BaseModel):
    """Request model for swipe gesture submission."""
    gesture_id: str = Field(..., description="Unique gesture identifier (UUID)")
    coords: List[CoordinatePoint] = Field(..., min_length=1, description="Swipe trajectory points")
    word: str = Field(default="", description="Target word (label), empty for prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "gesture_id": "550e8400-e29b-41d4-a716-446655440000",
                "coords": [
                    {"x": 342.36, "y": 263.07, "t": 0.0},
                    {"x": 342.36, "y": 268.18, "t": 0.067},
                    {"x": 347.90, "y": 296.62, "t": 0.085}
                ],
                "word": "привет"
            }
        }


class SwipeResponse(BaseModel):
    """Response model for swipe gesture submission."""
    status: str = Field(..., description="Status of the operation")
    gesture_id: str = Field(..., description="Gesture ID that was saved")
    message: str = Field(..., description="Human-readable message")
