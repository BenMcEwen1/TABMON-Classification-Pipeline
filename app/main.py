from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.database import ParquetDatabase
from app.schema import RetrievalSchema, PipelineSchema, Filter
from app.file_utils import find_audio_file, create_zip_archive, find_embedding_file, select_samples_from_recordings
from pipeline.analyze import run
from pipeline.util import load_species_list

from datetime import datetime
import time
import os
from typing import Optional
import json
import pandas as pd
import csv


app = FastAPI(title="TABMON API", description="Bird sound classification API")

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_instance = ParquetDatabase()

def get_db():
    return db_instance

# --- Status endpoints ---

@app.get("/", tags=["Status"])
async def status(db: ParquetDatabase = Depends(get_db)):
    return {
        "status": "success",
        "database": db.get_status(),
        "timestamp": datetime.now().isoformat()
    }

# --- Species endpoints ---

@app.get("/species/", tags=["Species"])
async def species_list(db: ParquetDatabase = Depends(get_db)):
    try:
        predictions_df = db.get_predictions()
        species_predicted = predictions_df["predicted_species"].unique().tolist() if not predictions_df.empty else []
    except Exception as e:
        print(f"Error querying predictions: {e}")
        species_predicted = []
    
    species = load_species_list("./pipeline/inputs/BirdNET_GLOBAL_6K_V2.4_Labels_en_uk.csv")
    all_species = []
    for name in species:
        scientific_name = name.split(',')[0]
        common_name = name.split(',')[1]
        predicted = scientific_name in species_predicted
        all_species.append({
            "common_name": common_name, 
            "scientific_name": scientific_name, 
            "predicted": predicted
        })
    return all_species

# --- Query endpoints ---

@app.post("/retrieve/", tags=["Query"])
async def retrieve(filters: RetrievalSchema, db: ParquetDatabase = Depends(get_db)):
    """Retrieve segments based on filters."""
    result = db.get_segments_with_predictions(filters)

    if "energy" in result.columns:
        result = result.drop(columns=["energy"])

    return json.loads(result.to_json(orient='records'))

@app.get("/export/", tags=["Query"])
def export(
    start_date: Optional[datetime] = None, 
    end_date: Optional[datetime] = None, 
    country: Optional[str] = None, 
    device_id: Optional[str] = None,
    confidence: Optional[float] = None,
    predicted_species: Optional[str] = None,
    uncertainty: Optional[float] = None,
    energy: Optional[float] = None,
    annotated: Optional[bool] = None,
    embeddings: bool = False,
    query_limit: Optional[int] = 100,
    stratified: Optional[bool] = False,
    db: ParquetDatabase = Depends(get_db)
):
    """Export segments and predictions as CSV and audio files."""

    filters = Filter(
        start_date=start_date,
        end_date=end_date,
        country=country,
        device_id=device_id,
        confidence=confidence,
        predicted_species=predicted_species,
        uncertainty=uncertainty,
        energy=energy,
        annotated=annotated,
        stratified=stratified,
        query_limit=query_limit
    )

    print("[Step 1] Retrieving results...", end="")
    start = time.time()
    results_df = db.get_segments_with_predictions(filters)
    print(f" Complete [{(time.time() - start):.2f} s]")

    if results_df.empty:
        return HTTPException(status_code=204, detail="No files available")

    filenames = [f"{data['filename'][:-4]}_{data['device_id']}_{data['start_time']//3}.mp3" for index, data in results_df.iterrows()]

    annotator = pd.DataFrame({"Country": results_df["country"],
                              "Device_id": results_df["device_id"], 
                              "Filename": results_df["filename"], 
                              "Start_time": results_df["start_time"],                                               
                              "Sample_filename": filenames,
                              "Predictions": results_df["predicted_species_list"], 
                              "Confidence": results_df["confidence_list"],
                              })
    
    import numpy as np
    def flatten_array(x):
        if isinstance(x, (list, np.ndarray)):
            return ", ".join(x)  # comma-separated string
        return str(x)

    annotator["Predictions"] = annotator["Predictions"].apply(flatten_array)

    EMBEDDING_DIR = "./pipeline/outputs/embeddings/"
    SEGMENT_DIR = "./pipeline/outputs/segments/"
    EXPORT_DIR = "./pipeline/outputs/exports/"
    DATASET_PATH =  "/DYNI/tabmon/tabmon_data"
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    os.makedirs(EXPORT_DIR, exist_ok=True)

    # EXPORT AS A CSV
    csv_file = f"export_{timestamp}.csv"
    annotator.to_csv(os.path.join(EXPORT_DIR, csv_file), index=False)
    prefix = "audio"
    
    PADDING = 3 # seconds (before and after samples)
    zip_path = select_samples_from_recordings(filters, csv_file, PADDING, EXPORT_DIR, DATASET_PATH)
    
    if zip_path:
        return FileResponse(zip_path, filename=f"{prefix}_{timestamp}.zip", media_type="application/zip")
    else:
        return HTTPException(status_code=204, detail="No files available")

# --- Analysis endpoint ---

@app.post("/analyse", tags=["Analysis"])
async def analyse(parameters: PipelineSchema, db: ParquetDatabase = Depends(get_db)):
    predictions = run(parameters, db)
    return predictions

# --- Database access endpoints ---
@app.get("/countries", tags=["Database"])
async def get_countries(db: ParquetDatabase = Depends(get_db)):
    devices_df = db.get_countries()
    return devices_df.to_dict(orient='records')

@app.get("/devices", tags=["Database"])
async def get_devices(db: ParquetDatabase = Depends(get_db)):
    devices_df = db.get_devices()
    return devices_df.to_dict(orient='records')

@app.get("/segments/", tags=["Database"])
def get_segments(db: ParquetDatabase = Depends(get_db)):
    segments_df = db.get_segments()
    return segments_df.to_dict(orient='records')

@app.get("/predictions/", tags=["Database"])
def get_predictions(db: ParquetDatabase = Depends(get_db)):
    predictions_df = db.get_predictions()
    return predictions_df.to_dict(orient='records')

@app.get("/predictions/{segment_id}", tags=["Database"])
def get_segment_predictions(segment_id: int, db: ParquetDatabase = Depends(get_db)):
    predictions_df = db.get_predictions(segment_id=segment_id)
    if predictions_df.empty:
        raise HTTPException(status_code=404, detail="Predictions not found")
    return predictions_df.to_dict(orient='records')

# --- File access endpoints ---

@app.get("/segments/audio/{filename}", response_class=FileResponse, tags=["Files"])
def get_audio_segment(filename: str):
    path = f"./pipeline/outputs/segments/{filename}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path=path, filename=filename, media_type="audio/mpeg")

@app.get("/audio_file/{filename}", response_class=FileResponse, tags=["Files"])
def get_audio_file(filename: str):
    path = f"./audio/{filename}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path=path, filename=filename, media_type="audio/mpeg")