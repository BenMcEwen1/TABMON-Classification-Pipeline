from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy import and_
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from dataclasses import asdict
import torch
import faiss

# Absolute imports
from app.database import SessionLocal, Device, Audio, Segment, Predictions
from app.schema import DeviceSchema, AudioSchema, SegmentSchema, PredictionSchema, RetrievalSchema, PipelineSchema
from app.services import add_embedding, apply_filters, apply_filters_body, segmentsWithPredictions, flatten, normalise
from pipeline.analyze import run
from pipeline.util import load_species_list

import pandas as pd
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import time
import tempfile
from pathlib import Path
import zipfile
import os


app = FastAPI()

# Define allowed origins (for local development)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,  # Allow cookies to be sent across origins
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers (e.g., Content-Type, Authorization)
)

# Dependency: Get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
async def status():
    return {"status", "success"}

@app.get("/species/")
async def species_list(db:Session=Depends(get_db)):
    prediction = db.query(Predictions).all()
    species_predicted = [p.predicted_species for p in prediction]
    species = load_species_list("./pipeline/inputs/BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.csv")
    all_species = []
    for name in species:
        scientific_name = name.split(',')[0]
        common_name = name.split(',')[1]
        predicted = name.split(',')[0] in species_predicted
        all_species.append({"common_name": common_name, "scientific_name": scientific_name, "predicted": predicted})
    return all_species


# @app.get("/devices", response_model=list[DeviceSchema])
# async def devices(db:Session=Depends(get_db)):
#     return db.query(Device).all()

# @app.get("/audio", response_model=list[AudioSchema])
# async def audio(db:Session=Depends(get_db)):
#     return db.query(Audio).all()

# @app.get("/segments", response_model=list[SegmentSchema])
# async def segments(db:Session=Depends(get_db)):
#     return db.query(Segment).all()

# @app.get("/predictions", response_model=list[PredictionSchema])
# async def predictions(db:Session=Depends(get_db)):
#     return db.query(Predictions).all()

@app.post("/analyse")
async def analyse(parameters:PipelineSchema, db:Session=Depends(get_db)):
    predictions = run(parameters, db)
    # status = normalise(predictions, db)
    return status

@app.get("/export/")
def export(start_date: datetime | None = None, 
           end_date: datetime | None = None, 
           country: str | None = None, 
           device_id: int | None = None,
           confidence: float | None = None,
           predicted_species: str | None = None,
           uncertainty: float | None = None,
           energy: float | None = None,
           annotated: bool | None = None,
           embeddings: bool = False,
           limit: int | None = 100,
           db:Session=Depends(get_db)):
    
    EMBEDDING_DIR = "./audio/embeddings/"
    SEGMENT_DIR = "./audio/segments/"

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    filters = locals()
    results, segments = apply_filters(filters)

    if embeddings:
        filenames = [os.path.join(EMBEDDING_DIR, f'{segment.filename[:-6]}.pt') for segment in segments]
        prefix = "embeddings"
    else:
        filenames = [os.path.join(SEGMENT_DIR, segment.filename) for segment in segments]
        prefix = "audio"

    csv_filename = flatten(results, SEGMENT_DIR, timestamp)
    filenames.append(csv_filename)

    # Create a temporary file for the zip archive
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
        zip_path = tmp_zip.name # Get the temporary file path

    # Create the zip file
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in filenames:
            file_path = Path(file)
            if file_path.exists():
                zipf.write(file_path, arcname=file_path.name)  # Add file with its filename
    if len(filenames) > 1:
        return FileResponse(zip_path, filename=f"{prefix}_{timestamp}.zip", media_type="application/zip")
    else:
        return HTTPException(status_code=204, detail="No files available")

@app.patch("/add_filenames/", response_model=list[SegmentSchema])
def add_filename(db:Session=Depends(get_db)):
    segments = db.query(Segment).all()
    for segment in segments:
        audio = db.query(Audio).filter(Audio.id == segment.audio_id).first()
        index = int(segment.start_time/3)
        filename = audio.filename
        segment.filename = os.path.splitext(filename)[0] + f"_{index}.wav"
        flag_modified(segment, "filename")
    db.commit()
    return segments

@app.patch("/device/", response_model=DeviceSchema)
def add_device_name(device_id:int, device_info:DeviceSchema, db:Session=Depends(get_db)):
    device = db.query(Device).filter(Device.id == device_id).first()
    device.device_id = device_info.device_id
    device.lat = device_info.lat
    device.lng = device_info.lng
    db.commit()
    return device

@app.get("/uncertain/")
def uncertainty(num_queries:int = 100, db:Session=Depends(get_db)):
    segments =  db.query(Segment).order_by(Segment.uncertainty.desc()).limit(num_queries).all()
    return segmentsWithPredictions(segments, db)

@app.get("/energy/")
def energy(num_queries:int=100, db:Session=Depends(get_db)):
    segments = db.query(Segment).order_by(Segment.energy.desc()).limit(num_queries).all()
    return segmentsWithPredictions(segments, db)

@app.get("/confidence/")
def confidence(num_queries:int=100, species: str | None = None, db:Session=Depends(get_db)):
    predictions = db.query(Predictions)
    
    if species:
        predictions = predictions.filter(Predictions.predicted_species == species)

    predictions = predictions.order_by(Predictions.confidence.desc()).all()
    segment_ids = [prediction.segment_id for prediction in predictions]
    segment_ids = list(dict.fromkeys(segment_ids)) # deduplicate

    segments = [db.query(Segment).filter(Segment.id == segment_id).limit(num_queries).first()  for segment_id in segment_ids]
    return segmentsWithPredictions(segments, db)

# ### Devices ###
# @app.post("/devices/", response_model=DeviceSchema)
# def create_device(device: DeviceSchema, db: Session = Depends(get_db)):
#     new_device = Device(**device.dict())
#     db.add(new_device) 
#     db.commit()
#     db.refresh(new_device)
#     return new_device

# @app.get("/devices/", response_model=list[DeviceSchema])
# def read_devices(db: Session = Depends(get_db)):
#     return db.query(Device).all()

# @app.get("/devices/{device_id}", response_model=DeviceSchema)
# def read_device(device_id: int, db: Session = Depends(get_db)):
#     device = db.query(Device).filter(Device.id == device_id).first()
#     if not device:
#         raise HTTPException(status_code=404, detail="Device not found")
#     return device

# @app.put("/devices/{device_id}", response_model=DeviceSchema)
# def update_device(device_id: int, device: DeviceSchema, db: Session = Depends(get_db)):
#     db_device = db.query(Device).filter(Device.id == device_id).first()
#     if not db_device:
#         raise HTTPException(status_code=404, detail="Device not found")
#     for key, value in device.dict().items():
#         setattr(db_device, key, value)
#     db.commit()
#     db.refresh(db_device)
#     return db_device

@app.delete("/devices/")
def delete_device(db: Session = Depends(get_db)):
    db_device = db.query(Device).all()
    if not db_device:
        raise HTTPException(status_code=404, detail="Device not found")
    for device in db_device:
        db.delete(device)
        db.commit()
    return {"detail": "Device deleted successfully"}

### Audio ###
@app.post("/audio/", response_model=AudioSchema)
def create_audio(audio: AudioSchema, db: Session = Depends(get_db)):
    new_audio = Audio(**audio.dict())
    db.add(new_audio)
    db.commit()
    db.refresh(new_audio)
    return new_audio

# @app.get("/audio/", response_model=list[AudioSchema])
# def read_audio(db: Session = Depends(get_db)):
#     return db.query(Audio).all()

# @app.get("/audio/{audio_id}", response_model=AudioSchema)
# def read_audio(audio_id: int, db: Session = Depends(get_db)):
#     audio = db.query(Audio).filter(Audio.id == audio_id).first()
#     if not audio:
#         raise HTTPException(status_code=404, detail="Audio not found")
#     return audio

# @app.put("/audio/{audio_id}", response_model=AudioSchema)
# def update_audio(audio_id: int, audio: AudioSchema, db: Session = Depends(get_db)):
#     db_audio = db.query(Audio).filter(Audio.id == audio_id).first()
#     if not db_audio:
#         raise HTTPException(status_code=404, detail="Audio not found")
#     for key, value in audio.dict().items():
#         setattr(db_audio, key, value)
#     db.commit()
#     db.refresh(db_audio)
#     return db_audio

# @app.delete("/audio/{audio_id}")
# def delete_audio(audio_id: int, db: Session = Depends(get_db)):
#     db_audio = db.query(Audio).filter(Audio.id == audio_id).first()
#     if not db_audio:
#         raise HTTPException(status_code=404, detail="Audio not found")
#     db.delete(db_audio)
#     db.commit()
#     return {"detail": "Audio deleted successfully"}

### Segments ###
@app.post("/segments/", response_model=SegmentSchema)
def create_segment(segment: SegmentSchema, embedding: list[float], db: Session = Depends(get_db)):
    embedding = torch.tensor([embedding])
    index, id = add_embedding(embedding)
    faiss.write_index(index, "./embeddings.bin")

    segment.embedding_id = id
    new_segment = Segment(**segment.dict()) # dict data
    db.add(new_segment)
    db.commit()
    db.refresh(new_segment)
    return new_segment

@app.get("/segments/", response_model=list[SegmentSchema])
def read_segment(db: Session = Depends(get_db)):
    return db.query(Segment).limit(100).all()

@app.post("/label/{id}/", response_model=SegmentSchema)
def add_label(id:int, label:dict, db:Session=Depends(get_db)):
    print(label)
    segment = db.query(Segment).filter(Segment.id == id).first()
    segment.label = label['label']
    flag_modified(segment, "label")
    db.commit()
    return segment

# @app.get("/segments/{segment_id}", response_model=SegmentSchema)
# def read_segment(segment_id: int, db:Session=Depends(get_db)):
#     segment = db.query(Segment).filter(Segment.id == segment_id).first()
#     return segment

@app.post("/retrieve/")
def retrieve(filters: RetrievalSchema, db:Session=Depends(get_db)):
    segments,_ = apply_filters_body(filters, db)
    return segments

@app.get("/segments/audio/{filename}", response_class=FileResponse)
def get_segments(filename:str):
    path = f"./audio/segments/{filename}"
    filename = f"{filename}"
    return FileResponse(path=path, filename=filename, media_type="audio/mpeg")


    # return {"status": "complete"}

# @app.put("/segments/{segment_id}", response_model=SegmentSchema)
# def update_segment(segment_id:int, segment:SegmentSchema, db:Session=Depends(get_db)):
#     db_segment = db.query(Segment).filter(Segment.id == segment_id).first()
#     if not db_segment:
#         raise HTTPException(status_code=404, detail="Segment not found")
#     for key, value in segment.dict().items():
#         setattr(db_segment, key, value)
#     db.commit()
#     db.refresh(db_segment)
#     return db_segment

# @app.delete("/segments/{segment_id}")
# def delete_segment(segment_id: int, db: Session = Depends(get_db)):
#     db_segment = db.query(Segment).filter(Segment.id == segment_id).first()
#     if not db_segment:
#         raise HTTPException(status_code=404, detail="Segment not found")
#     db.delete(db_segment)
#     db.commit()
#     return {"detail": "Segment deleted successfully"}

### Predictions ###
# @app.post("/predictions/", response_model=PredictionSchema)
# def create_prediction(prediction: PredictionSchema, db: Session = Depends(get_db)):
#     new_prediction = Predictions(**prediction.dict())
#     db.add(new_prediction)
#     db.commit()
#     db.refresh(new_prediction)
#     return new_prediction

@app.get("/predictions/", response_model=list[PredictionSchema])
def read_prediction(db:Session=Depends(get_db)):
    return db.query(Predictions).all()

@app.get("/predictions/{segment_id}", response_model=list[PredictionSchema])
def read_prediction(segment_id: int, db: Session = Depends(get_db)):
    prediction = db.query(Predictions).filter(Predictions.segment_id == segment_id).all()
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction

# @app.put("/predictions/{prediction_id}", response_model=PredictionSchema)
# def update_prediction(prediction_id: int, prediction: PredictionSchema, db: Session = Depends(get_db)):
#     db_prediction = db.query(Predictions).filter(Predictions.id == prediction_id).first()
#     if not db_prediction:
#         raise HTTPException(status_code=404, detail="Prediction not found")
#     for key, value in prediction.dict().items():
#         setattr(db_prediction, key, value)
#     db.commit()
#     db.refresh(db_prediction)
#     return db_prediction

# @app.delete("/predictions/{prediction_id}")
# def delete_prediction(prediction_id: int, db: Session = Depends(get_db)):
#     db_prediction = db.query(Predictions).filter(Predictions.id == prediction_id).first()
#     if not db_prediction:
#         raise HTTPException(status_code=404, detail="Prediction not found")
#     db.delete(db_prediction)
#     db.commit()
#     return {"detail": "Prediction deleted successfully"}