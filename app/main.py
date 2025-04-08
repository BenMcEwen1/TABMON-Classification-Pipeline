from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from dataclasses import asdict
import torch
import faiss
import pandas as pd
import numpy as np

# Updated import for ParquetDatabase
from app.database import initialize_database
from app.schema import DeviceSchema, AudioSchema, SegmentSchema, PredictionSchema, RetrievalSchema, PipelineSchema
from pipeline.analyze import run
from pipeline.util import load_species_list

from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import time
import tempfile
from pathlib import Path
import zipfile
import os
from types import SimpleNamespace


app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the database
db_instance = initialize_database()

# Dependency: Get database instance
def get_db():
    return db_instance

@app.get("/")
async def status():
    return {"status": "success"}

@app.get("/species/")
async def species_list(db=Depends(get_db)):
    try:
        # Query predictions using DuckDB
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
        all_species.append({"common_name": common_name, "scientific_name": scientific_name, "predicted": predicted})
    return all_species

@app.post("/analyse")
async def analyse(parameters: PipelineSchema, db=Depends(get_db)):
    predictions = run(parameters, db)
    return predictions

# Function to apply filters for DuckDB
def apply_filters_duckdb(filters, db):
    try:
        query_parts = ["SELECT s.*, a.filename as audio_filename, a.date_recorded, d.device_id, d.country"]
        query_parts.append("FROM segments s")
        query_parts.append("JOIN audio a ON s.audio_id = a.id")
        query_parts.append("JOIN devices d ON a.device_id = d.device_id")
        
        where_conditions = []

        if hasattr(filters, 'predicted_species') and filters.predicted_species:
            # Fix: Add f prefix to make this an actual f-string
            pred_join = f"""
            JOIN predictions p ON s.id = p.segment_id 
            WHERE p.predicted_species = '{filters.predicted_species}'
            """
            if where_conditions:
                pred_join = pred_join.replace('WHERE', 'AND')
            query_parts.append(pred_join)
        
        if hasattr(filters, 'start_date') and filters.start_date:
            where_conditions.append(f"a.date_recorded >= '{filters.start_date}'")
            
        if hasattr(filters, 'end_date') and filters.end_date:
            where_conditions.append(f"a.date_recorded <= '{filters.end_date}'")
            
        if hasattr(filters, 'country') and filters.country:
            where_conditions.append(f"d.country = '{filters.country}'")
            
        if hasattr(filters, 'device_id') and filters.device_id:
            where_conditions.append(f"d.device_id = '{filters.device_id}'")
            
        if hasattr(filters, 'uncertainty') and filters.uncertainty:
            where_conditions.append(f"s.uncertainty >= {filters.uncertainty}")
            
        if hasattr(filters, 'energy') and filters.energy:
            where_conditions.append(f"s.energy >= {filters.energy}")
            
        if hasattr(filters, 'annotated') and filters.annotated is not None:
            where_conditions.append(f"s.label IS NOT NULL" if filters.annotated else "s.label IS NULL")

        if where_conditions:
            query_parts.append("WHERE " + " AND ".join(where_conditions))
            
        if hasattr(filters, 'predicted_species') and filters.predicted_species:
            # This requires a join with predictions
            pred_join = """
            JOIN predictions p ON s.id = p.segment_id 
            WHERE p.predicted_species = '{filters.predicted_species}'
            """
            if where_conditions:
                pred_join = pred_join.replace('WHERE', 'AND')
            query_parts.append(pred_join)
            
            if hasattr(filters, 'confidence') and filters.confidence:
                query_parts.append(f"AND p.confidence >= {filters.confidence}")
                
        limit_clause = ""
        if hasattr(filters, 'query_limit') and filters.query_limit:
            limit_clause = f" LIMIT {filters.query_limit}"
        
        query = " ".join(query_parts) + limit_clause
        segments_df = db.execute_query(query)
        
        # Get predictions for these segments
        if not segments_df.empty:
            segment_ids = segments_df['id'].tolist()
            pred_list_str = ",".join(map(str, segment_ids))
            predictions_df = db.execute_query(f"SELECT * FROM predictions WHERE segment_id IN ({pred_list_str})")
        else:
            predictions_df = pd.DataFrame()
            
        return segments_df, predictions_df
    except Exception as e:
        print(f"Error applying filters: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Helper function similar to segmentsWithPredictions
def segments_with_predictions_duckdb(segments_df, db):
    result = []
    for _, segment in segments_df.iterrows():
        segment_dict = segment.to_dict()
        
        # Get predictions for this segment
        predictions_df = db.get_predictions(segment_id=segment_dict['id'])
        segment_dict['predictions'] = predictions_df.to_dict(orient='records')
        
        result.append(segment_dict)
    return result

# Helper function to flatten dataframes to CSV
def flatten_duckdb(segments_df, predictions_df, segment_dir, timestamp):
    # Create a flat dataframe with all info
    flat_rows = []
    
    for _, segment in segments_df.iterrows():
        segment_dict = segment.to_dict()
        segment_predictions = predictions_df[predictions_df['segment_id'] == segment_dict['id']]
        
        for _, prediction in segment_predictions.iterrows():
            pred_dict = prediction.to_dict()
            flat_row = {**segment_dict, **pred_dict}
            flat_rows.append(flat_row)
    
    if flat_rows:
        flat_df = pd.DataFrame(flat_rows)
        csv_path = f"{segment_dir}/export_{timestamp}.csv"
        flat_df.to_csv(csv_path, index=False)
        return csv_path
    return None

@app.post("/retrieve/")
def retrieve(filters: RetrievalSchema, db=Depends(get_db)):
    start = time.time()
    
    # Debug information
    print(f"Parquet directory: {db.parquet_dir}")
    print(f"Existing files: {os.listdir(db.parquet_dir)}")
    
    segments_df, _ = apply_filters_duckdb(filters, db)
    end = time.time()
    print(f"Time taken to retrieve: {end - start} seconds")
    print(f"Segments found: {len(segments_df)}")
    return segments_df.to_dict(orient='records')

@app.get("/count")
def count(db=Depends(get_db)):
    audio_df = db.get_audio_files()
    return len(audio_df)

@app.get("/export/", tags=["Sampling"])
def export(start_date: datetime | None = None, 
           end_date: datetime | None = None, 
           country: str | None = None, 
           device_id: str | None = None,
           confidence: float | None = None,
           predicted_species: str | None = None,
           uncertainty: float | None = None,
           indice: str | None = None,
           energy: float | None = None,
           annotated: bool | None = None,
           embeddings: bool = False,
           query_limit: int | None = 100,
           db=Depends(get_db)):
    
    EMBEDDING_DIR = "./audio/embeddings/"
    SEGMENT_DIR = "./audio/segments/"

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    filters = SimpleNamespace(**locals())
    segments_df, predictions_df = apply_filters_duckdb(filters, db)
    
    if segments_df.empty:
        return HTTPException(status_code=204, detail="No files available")

    if embeddings:
        filenames = [os.path.join(EMBEDDING_DIR, f'{row["filename"][:-4].lower()}.pt') 
                     for _, row in segments_df.iterrows()]
        prefix = "embeddings"
    else:
        # Fix: Don't just use filename directly - check if it exists
        file_list = []
        
        # Get all files in segment directory first (for case-insensitive lookup)
        segment_files_lower = {}
        if os.path.exists(SEGMENT_DIR):
            for filename in os.listdir(SEGMENT_DIR):
                segment_files_lower[filename.lower()] = os.path.join(SEGMENT_DIR, filename)
        
        for _, row in segments_df.iterrows():
            # 1. Try exact filename first
            filepath = os.path.join(SEGMENT_DIR, row["filename"])
            if os.path.exists(filepath):
                file_list.append(filepath)
                continue
                
            # 2. Try lowercase version
            lower_filename = row["filename"].lower()
            if lower_filename in segment_files_lower:
                file_list.append(segment_files_lower[lower_filename])
                continue
                
            # 3. Try the expected transformed filename format
            base_filename = os.path.splitext(row["audio_filename"])[0]
            segment_start = int(row["start_time"])
            index = int(segment_start / 3)
            device_id = row["device_id"]
            
            # Try the transformed filename with device_id and index
            transformed_filename = f"{base_filename}_{device_id}_{index}.wav"
            
            # Try direct match of transformed filename
            filepath = os.path.join(SEGMENT_DIR, transformed_filename)
            if os.path.exists(filepath):
                file_list.append(filepath)
                continue
                
            # Try case-insensitive match of transformed filename
            lower_transformed = transformed_filename.lower()
            if lower_transformed in segment_files_lower:
                file_list.append(segment_files_lower[lower_transformed])
                continue
                
            print(f"Warning: Could not find audio file for segment: {row['filename']} or {transformed_filename}")
            print(f"Filename in DB: {row['filename']}")
            print(f"Transformed filename: {transformed_filename}")
            print(f"Looking for file in: {SEGMENT_DIR}")

        filenames = file_list
        prefix = "audio"

    csv_filename = flatten_duckdb(segments_df, predictions_df, SEGMENT_DIR, timestamp)
    if csv_filename:
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

@app.get("/uncertain/", tags=["Sampling"])
def uncertainty(num_queries: int = 100, db=Depends(get_db)):
    query = f"SELECT * FROM segments ORDER BY uncertainty DESC LIMIT {num_queries}"
    segments_df = db.execute_query(query)
    return segments_with_predictions_duckdb(segments_df, db)

@app.get("/energy/", tags=["Sampling"])
def energy(num_queries: int = 100, db=Depends(get_db)):
    query = f"SELECT * FROM segments ORDER BY energy DESC LIMIT {num_queries}"
    segments_df = db.execute_query(query)
    return segments_with_predictions_duckdb(segments_df, db)

@app.get("/confidence/", tags=["Sampling"])
def confidence(num_queries: int = 100, species: str | None = None, db=Depends(get_db)):
    if species:
        query = f"""
        SELECT p.*, s.* FROM predictions p
        JOIN segments s ON p.segment_id = s.id
        WHERE p.predicted_species = '{species}'
        ORDER BY p.confidence DESC
        LIMIT {num_queries}
        """
    else:
        query = f"""
        SELECT p.*, s.* FROM predictions p
        JOIN segments s ON p.segment_id = s.id
        ORDER BY p.confidence DESC
        LIMIT {num_queries}
        """
        
    results_df = db.execute_query(query)
    
    # Extract unique segments
    segment_ids = results_df['id'].unique()
    segments_df = db.execute_query(f"SELECT * FROM segments WHERE id IN ({','.join(map(str, segment_ids))})")
    
    return segments_with_predictions_duckdb(segments_df, db)

@app.get("/devices", tags=["Devices"])
async def devices(db=Depends(get_db)):
    devices_df = db.get_devices()
    return devices_df.to_dict(orient='records')

@app.get("/devices/{device_id}", tags=["Devices"])
def read_device(device_id: int, db=Depends(get_db)):
    devices_df = db.get_devices(filters={"id": device_id})
    if devices_df.empty:
        raise HTTPException(status_code=404, detail="Device not found")
    return devices_df.iloc[0].to_dict()

@app.patch("/device/", tags=["Devices"])
def add_device_name(device_id: int, device_info: DeviceSchema, db=Depends(get_db)):
    # This requires writing to parquet files
    # For DuckDB to make this work, we'd need to implement writing functionality
    # in ParquetDatabase class
    
    # Example of how it could work (assuming update_device is implemented)
    device_dict = device_info.dict()
    updated_device = db.update_device(device_id, device_dict)
    if not updated_device:
        raise HTTPException(status_code=404, detail="Device not found")
    return updated_device

@app.delete("/devices/{device_id}", tags=["Devices"])
def delete_device(device_id: str, db=Depends(get_db)):
    # Similar to update, deletion requires writing functionality
    result = db.delete_device(device_id)
    if not result:
        raise HTTPException(status_code=404, detail="Device not found")
    return {"detail": "Device deleted successfully"}

@app.get("/segments/", tags=["Segments"])
def read_segment(db=Depends(get_db)):
    segments_df = db.get_segments()
    return segments_df.to_dict(orient='records')

@app.post("/segments/", tags=["Segments"])
def create_segment(segment: SegmentSchema, embedding: list[float], db=Depends(get_db)):
    # For creating new records, we'd need to implement append functionality
    embedding_tensor = torch.tensor([embedding])
    
    # This would require implementing add_embedding and create_segment methods
    segment_dict = segment.dict()
    new_segment = db.create_segment(segment_dict, embedding_tensor)
    
    return new_segment

@app.post("/label/{id}/", tags=["Segments"])
def add_label(id: int, label: dict, db=Depends(get_db)):
    # Update the label field
    result = db.update_segment(id, {"label": label['label']})
    if not result:
        raise HTTPException(status_code=404, detail="Segment not found")
    return result

@app.get("/segments/audio/{filename}", response_class=FileResponse, tags=["Segments"])
def get_audio_segment(filename: str):
    path = f"./audio/segments/{filename}"
    return FileResponse(path=path, filename=filename, media_type="audio/mpeg")

@app.get("/audio_file/{filename}", response_class=FileResponse, tags=["Segments"])
def get_audio_file(filename: str):
    path = f"./audio/{filename}"
    return FileResponse(path=path, filename=filename, media_type="audio/mpeg")

@app.get("/predictions/", tags=["Predictions"])
def read_prediction(db=Depends(get_db)):
    predictions_df = db.get_predictions()
    return predictions_df.to_dict(orient='records')

@app.get("/predictions/{segment_id}", tags=["Predictions"])
def read_prediction(segment_id: int, db=Depends(get_db)):
    predictions_df = db.get_predictions(segment_id=segment_id)
    if predictions_df.empty:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return predictions_df.to_dict(orient='records')

@app.put("/predictions/{prediction_id}", tags=["Predictions"])
def update_prediction(prediction_id: int, prediction: PredictionSchema, db=Depends(get_db)):
    prediction_dict = prediction.dict()
    result = db.update_prediction(prediction_id, prediction_dict)
    if not result:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return result

@app.delete("/predictions/{prediction_id}", tags=["Predictions"])
def delete_prediction(prediction_id: int, db=Depends(get_db)):
    result = db.delete_prediction(prediction_id)
    if not result:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {"detail": "Prediction deleted successfully"}

@app.put("/dates", tags=["Patches"])
def update_dates(db=Depends(get_db)):
    # This would require implementing an update_dates method
    # Here's how it might work conceptually
    audio_df = db.get_audio_files()
    
    # Process rows with missing date_recorded
    updated_rows = []
    for idx, row in audio_df.iterrows():
        if pd.isna(row['date_recorded']):
            filename = row['filename']
            try:
                date_recorded = datetime.strptime(filename.split(".")[0], "%Y-%m-%dT%H_%M_%S")
                audio_df.at[idx, 'date_recorded'] = date_recorded
                updated_rows.append(row.to_dict())
            except:
                print(f"Failed to parse date for {filename}")
    
    # Update the parquet files with the new data
    if updated_rows:
        # This would require implementing an update_audio_files method
        db.update_audio_files(audio_df)
    
    return {"updated": len(updated_rows)}

@app.get("/audio", tags=["Audio"])
def read_audio(db=Depends(get_db)):
    audio_df = db.get_audio_files()
    return audio_df.to_dict(orient='records')