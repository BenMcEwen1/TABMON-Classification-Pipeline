from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class DeviceSchema(BaseModel):
    device_id: str
    country: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    model_name: Optional[str] = None
    model_checkpoint: Optional[str] = None
    date_updated: Optional[datetime] = None
    date_deployed: Optional[datetime] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "device_id": "1717d9df",
                    "country": "Norway",
                    "lat": 58.754313,
                    "lng": 5.532336,
                    "model_name": "birdnet",
                    "model_checkpoint": "BirdNET_GLOBAL_6K_V2.4_Model_FP32",
                    "date_updated": "2025-01-30T08:37:42.234000",
                    "date_deployed": "2024-05-01T17:50:45.234000"
                }
            ]
        }
    }

class AudioSchema(BaseModel):
    filename: str
    sample_rate: Optional[int] = None
    date_recorded: Optional[datetime] = None
    device_id: Optional[int] = None

class SegmentSchema(BaseModel):
    start_time: float
    filename: Optional[str] = None
    duration: Optional[float] = None
    uncertainty: Optional[float] = None
    energy: Optional[float] = None
    date_processed: Optional[datetime] = None
    audio_id: int
    embedding_id: int

class PredictionSchema(BaseModel):
    predicted_species: str
    confidence: float
    segment_id: int

class SegmentWithPredictions(SegmentSchema):
    predictions: list[PredictionSchema]