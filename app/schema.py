from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class Filter(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    country: Optional[str] = None
    device_id: Optional[str] = None
    predicted_species: Optional[str] = None
    confidence: Optional[float] = None
    uncertainty: Optional[float] = None
    energy: Optional[float] = None
    annotated: Optional[bool] = None
    query_limit: Optional[int] = 100

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
    energy: Optional[dict] = None
    date_processed: Optional[datetime] = None

    label: Optional[str | None] = None
    notes: Optional[str] = None
    
    audio_id: int
    embedding_id: int

class PredictionSchema(BaseModel):
    predicted_species: str
    confidence: float
    segment_id: int

class RetrievalSchema(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    country: Optional[str] = Field(None, example=None) 
    device_id: Optional[str] = None
    confidence: Optional[float] = None
    predicted_species: Optional[list[str]] = None
    uncertainty: Optional[float] = None
    indice: Optional[str] = None
    energy: Optional[float] = None
    annotated: Optional[bool] = None
    embeddings: Optional[bool] = None
    query_limit: Optional[int] = 100

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "start_date": None,
                    "end_date": None,
                    "country": None,
                    "device_id": None,
                    "confidence": None,
                    "predicted_species": None,
                    "uncertainty": None,
                    "indice": None,
                    "energy": None,
                    "annotated": None,
                    "embeddings": False,
                    "query_limit": 100
                }
            ]
        }
    }

class SegmentWithPredictions(SegmentSchema):
    predictions: list[PredictionSchema]


class PipelineSchema(BaseModel):
    slist: str = 'pipeline/inputs/list_sp_ml.csv'
    flist: Optional[str] = None
    i: str = 'audio/brambling'

    device_id: str
    country: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    model_name: str = 'birdnet'
    model_checkpoint: Optional[str] = None
    date_updated: Optional[datetime] = None
    date_deployed: Optional[datetime] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "slist": 'pipeline/inputs/list_sp_ml.csv',
                    "flist": None,
                    "i": 'audio/brambling',
                    "device_id": "123456",
                    "country": None,
                    "lat": None,
                    "lng": None,
                    "model_name":'birdnet',
                    "model_checkpoint": None,
                    "date_updated": None,
                    "date_deployed": None
                }
            ]
        }
    }