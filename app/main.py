from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.database import ParquetDatabase
from app.schema import RetrievalSchema, PipelineSchema, Filter
from app.file_utils import find_audio_file, create_zip_archive, find_embedding_file
from pipeline.analyze import run
from pipeline.util import load_species_list

from datetime import datetime
import time
import os
from typing import Optional

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
def retrieve(filters: RetrievalSchema, db: ParquetDatabase = Depends(get_db)):
    """Retrieve segments based on filters."""
    result = db.get_segments_with_predictions(filters)
    return result.to_dict(orient='records')

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
        query_limit=query_limit
    )
    
    results_df = db.get_segments_with_predictions(filters)

    if "energy" in results_df.columns:
        results_df = results_df.drop(columns=["energy"])
    print(results_df['device_id'])
    
    if results_df.empty:
        return HTTPException(status_code=204, detail="No files available")
    
    EMBEDDING_DIR = "./pipeline/outputs/embeddings/"
    SEGMENT_DIR = "./pipeline/outputs/segments/"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    files_to_export = []
    
    if embeddings:
        for _, row in results_df.iterrows():
            embedding_file = find_embedding_file(row.to_dict(), EMBEDDING_DIR)
            if embedding_file:
                files_to_export.append(embedding_file)
        prefix = "embeddings"
    else:
        for _, row in results_df.iterrows():
            audio_file = find_audio_file(row.to_dict(), SEGMENT_DIR)
            if audio_file:
                files_to_export.append(audio_file)
        prefix = "audio"
    
    # EXPORT AS A CSV
    csv_path = f"{SEGMENT_DIR}/export_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    files_to_export.append(csv_path)
    
    zip_path = create_zip_archive(files_to_export, prefix)
    
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
    path = f"./audio/segments/{filename}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path=path, filename=filename, media_type="audio/mpeg")

@app.get("/audio_file/{filename}", response_class=FileResponse, tags=["Files"])
def get_audio_file(filename: str):
    path = f"./audio/{filename}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path=path, filename=filename, media_type="audio/mpeg")