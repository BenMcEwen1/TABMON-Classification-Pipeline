import os
import pandas as pd
from datetime import datetime
import time
from app.database import Device, Audio, Segment, Prediction

def normalise(predictions_df, db):
    """
    Normalize the database by splitting it into smaller linked tables.
    """
    start = time.time()
    
    # Fetch existing data
    try:
        devices_df = db.get_devices()
        audio_df = db.get_audio_files()
        segments_df = db.get_segments()
        predictions_df_existing = db.get_predictions()
    except Exception as e:
        print(f"Error fetching existing data: {e}")
        # Initialize empty DataFrames if tables don't exist yet
        devices_df = pd.DataFrame(columns=['id', 'device_id', 'lat', 'lng', 'model_name', 'model_checkpoint', 'date_updated'])
        audio_df = pd.DataFrame(columns=['id', 'filename', 'device_id', 'date_recorded'])
        segments_df = pd.DataFrame(columns=['id', 'start_time', 'filename', 'duration', 'uncertainty', 
                                           'energy', 'date_processed', 'label', 'notes', 'audio_id', 'embedding_id'])
        predictions_df_existing = pd.DataFrame(columns=['id', 'predicted_species', 'confidence', 'segment_id'])
    
    # Create lookup mappings
    existing_devices = dict(zip(devices_df['device_id'], devices_df['id'])) if not devices_df.empty else {}
    existing_audio = dict(zip(audio_df['filename'], audio_df['id'])) if not audio_df.empty else {}
    
    if not segments_df.empty and not audio_df.empty:
        segments_df = pd.merge(segments_df, audio_df[['id', 'filename']], left_on='audio_id', right_on='id', 
                             suffixes=('', '_audio'))
        existing_segments = dict(zip(zip(segments_df['audio_id'], segments_df['start_time']), segments_df['id']))
    else:
        existing_segments = {}

    # Prepare new records
    new_devices = []
    new_audio = []
    new_segments = []
    new_predictions = []
    
    # Generate new IDs
    next_device_id = 1 if devices_df.empty else devices_df['id'].max() + 1
    next_audio_id = 1 if audio_df.empty else audio_df['id'].max() + 1
    next_segment_id = 1 if segments_df.empty else segments_df['id'].max() + 1
    next_prediction_id = 1 if predictions_df_existing.empty else predictions_df_existing['id'].max() + 1
    
    # Device mapping
    device_map = {}
    for _, row in predictions_df.iterrows():
        if row["device_id"] not in existing_devices and row["device_id"] not in device_map:
            device_id = next_device_id
            next_device_id += 1
            new_devices.append({
                'id': device_id,
                'device_id': row["device_id"],
                'lat': row["lat"],
                'lng': row["lng"],
                'model_name': row["model"],
                'model_checkpoint': row["model_checkpoint"],
                'date_updated': datetime.now()
            })
            device_map[row["device_id"]] = device_id
        else:
            device_map[row["device_id"]] = existing_devices.get(row["device_id"])
    
    # Audio mapping
    audio_id_map = {}
    for _, row in predictions_df.iterrows():
        if row["filename"] not in existing_audio and row["filename"] not in audio_id_map:
            try:
                date_obj = datetime.strptime(row["filename"].split(".")[0], "%Y-%m-%dT%H_%M_%S")
            except:
                date_obj = None
                
            audio_id = next_audio_id
            next_audio_id += 1
            new_audio.append({
                'id': audio_id,
                'filename': row["filename"],
                'device_id': device_map[row["device_id"]],
                'date_recorded': date_obj
            })
            audio_id_map[row["filename"]] = audio_id
        else:
            audio_id_map[row["filename"]] = existing_audio.get(row["filename"])
    
    # Segment mapping
    segment_id_map = {}
    for _, row in predictions_df.iterrows():
        key = (audio_id_map[row["filename"]], row["start time"])
        if key not in existing_segments and key not in segment_id_map:
            index = int(row["start time"] / 3)

            energy = row["energy"]
            if isinstance(energy, str):
                try:
                    energy = eval(energy)
                except:
                    energy = 0

            segment_id = next_segment_id
            next_segment_id += 1
            new_segments.append({
                'id': segment_id,
                'start_time': row["start time"],
                'filename': os.path.splitext(row["filename"])[0] + f"_{row['device_id']}_{index}.wav",
                'duration': 3,
                'uncertainty': row["uncertainty"],
                'energy': energy,
                'date_processed': datetime.now(),
                'label': None,
                'notes': None,
                'audio_id': audio_id_map[row["filename"]],
                'embedding_id': 1
            })
            segment_id_map[key] = segment_id
        else:
            segment_id_map[key] = existing_segments.get(key)
    
    # Predictions
    for _, row in predictions_df.iterrows():
        audio_id = audio_id_map[row["filename"]]
        key = (audio_id, row["start time"])
        segment_id = segment_id_map[key]
        
        # Check if prediction already exists
        if not predictions_df_existing.empty:
            existing = predictions_df_existing[
                (predictions_df_existing['predicted_species'] == row["scientific name"]) & 
                (predictions_df_existing['confidence'] == row["confidence"]) & 
                (predictions_df_existing['segment_id'] == segment_id)
            ]
            if not existing.empty:
                continue
        
        prediction_id = next_prediction_id
        next_prediction_id += 1
        new_predictions.append({
            'id': prediction_id,
            'predicted_species': row["scientific name"],
            'confidence': row["confidence"],
            'segment_id': segment_id
        })
    
    try:
        # Convert lists of dictionaries to DataFrames
        if new_devices:
            new_devices_df = pd.DataFrame(new_devices)
            devices_df = pd.concat([devices_df, new_devices_df], ignore_index=True)
            
        if new_audio:
            new_audio_df = pd.DataFrame(new_audio)
            audio_df = pd.concat([audio_df, new_audio_df], ignore_index=True)
            
        if new_segments:
            new_segments_df = pd.DataFrame(new_segments)
            segments_df = pd.concat([segments_df, new_segments_df], ignore_index=True)
            
        if new_predictions:
            new_predictions_df = pd.DataFrame(new_predictions)
            predictions_df_existing = pd.concat([predictions_df_existing, new_predictions_df], ignore_index=True)
        
        # Write DataFrames back to Parquet files
        if not devices_df.empty:
            devices_dir = f"{db.parquet_dir}/devices"
            os.makedirs(devices_dir, exist_ok=True)
            devices_df.to_parquet(f"{devices_dir}/devices.parquet")
        
        if not audio_df.empty:
            audio_dir = f"{db.parquet_dir}/audio"
            os.makedirs(audio_dir, exist_ok=True)
            audio_df.to_parquet(f"{audio_dir}/audio.parquet")
            
        if not segments_df.empty:
            segments_dir = f"{db.parquet_dir}/segments"
            os.makedirs(segments_dir, exist_ok=True)
            segments_df.to_parquet(f"{segments_dir}/segments.parquet")
            
        if not predictions_df_existing.empty:
            predictions_dir = f"{db.parquet_dir}/predictions"
            os.makedirs(predictions_dir, exist_ok=True)
            predictions_df_existing.to_parquet(f"{predictions_dir}/predictions.parquet")
        
        # Refresh DuckDB views to see the new data
        db._register_views()
        
        end = time.time()
        print(f"Write successful in {end - start} seconds")
        return {"status": "data added successfully"}
    except Exception as error:
        print(f"Write failed: {error}")
        return None

def segments_with_predictions_duckdb(segments_df, predictions_df):
    """Join segments with their predictions using DataFrames."""
    results = []
    
    # For each segment, find its predictions
    for _, segment in segments_df.iterrows():
        segment_dict = segment.to_dict()
        segment_predictions = predictions_df[predictions_df['segment_id'] == segment_dict['id']]
        segment_dict["predictions"] = [Prediction.from_dict(row.to_dict()) 
                                      for _, row in segment_predictions.iterrows()]
        results.append(segment_dict)
        
    return results

def apply_filters(parameters, db):
    """Apply filters using DuckDB SQL instead of SQLAlchemy ORM."""
    # Build the SQL query
    query_parts = ["SELECT s.*, a.filename as audio_filename, a.date_recorded, d.device_id, d.country"]
    query_parts.append("FROM segments s")
    query_parts.append("JOIN audio a ON s.audio_id = a.id")
    query_parts.append("JOIN devices d ON a.device_id = d.id")
    
    where_conditions = []
    
    if parameters.country:
        where_conditions.append(f"d.country = '{parameters.country}'")
        
    if parameters.device_id:
        where_conditions.append(f"d.device_id = '{parameters.device_id}'")
        
    if parameters.start_date:
        where_conditions.append(f"a.date_recorded >= '{parameters.start_date}'")
        
    if parameters.end_date:
        where_conditions.append(f"a.date_recorded <= '{parameters.end_date}'")
        
    if parameters.uncertainty is not None:
        where_conditions.append(f"s.uncertainty >= {parameters.uncertainty}")
        
    if parameters.energy is not None and parameters.indice is not None:
        # DuckDB has different JSON handling than SQLite
        where_conditions.append(f"s.energy['{parameters.indice}'] >= {parameters.energy}")
        
    if parameters.annotated:
        where_conditions.append(f"s.label IS NOT NULL")
    
    pred_join = False
    if parameters.predicted_species or parameters.confidence is not None:
        query_parts.append("JOIN predictions p ON s.id = p.segment_id")
        pred_join = True
        
        if parameters.predicted_species:
            where_conditions.append(f"p.predicted_species = '{parameters.predicted_species}'")
            
        if parameters.confidence is not None:
            where_conditions.append(f"p.confidence >= {parameters.confidence}")
    
    if where_conditions:
        query_parts.append("WHERE " + " AND ".join(where_conditions))
        
    # Add order by and limit
    query_parts.append("ORDER BY s.id DESC")
    query_parts.append(f"LIMIT {parameters.query_limit}")
    
    # Execute query
    query = " ".join(query_parts)
    try:
        segments_df = db.execute_query(query)
    except Exception as e:
        print(f"Query error: {e}")
        segments_df = pd.DataFrame()
        
    # Get predictions for these segments
    if not segments_df.empty:
        segment_ids = segments_df['id'].unique().tolist()
        segment_ids_str = ','.join(map(str, segment_ids))
        
        try:
            if pred_join:
                # If we already joined with predictions, extract the prediction columns
                predictions_query = f"""
                SELECT p.* FROM predictions p
                WHERE p.segment_id IN ({segment_ids_str})
                """
                predictions_df = db.execute_query(predictions_query)
            else:
                # Otherwise just query the predictions
                predictions_query = f"""
                SELECT * FROM predictions
                WHERE segment_id IN ({segment_ids_str})
                """
                predictions_df = db.execute_query(predictions_query)
        except Exception as e:
            print(f"Predictions query error: {e}")
            predictions_df = pd.DataFrame()
            
        segments_with_preds = segments_with_predictions_duckdb(segments_df, predictions_df)
        return segments_with_preds, segments_df
    else:
        return [], pd.DataFrame()

def flatten(data, BASE_DIR, timestamp):
    """Flatten the data structure for CSV export - mostly unchanged."""
    flattened_data = []

    for segment in data:
        for prediction in segment["predictions"]:
            # Convert prediction object to dict if it's not already
            if hasattr(prediction, '__dict__'):
                prediction_dict = prediction.__dict__
                if '_sa_instance_state' in prediction_dict:
                    del prediction_dict['_sa_instance_state']
            else:
                prediction_dict = prediction
                
            row = {
                "segment_id": segment["id"],
                "duration": segment["duration"],
                "uncertainty": segment["uncertainty"],
                "energy": segment["energy"],
                "start_time": segment["start_time"],
                "filename": segment["filename"],
                "date_processed": segment["date_processed"],
                "audio_id": segment["audio_id"],
                "predicted_species": prediction_dict["predicted_species"],
                "confidence": prediction_dict["confidence"],
                "notes": segment["notes"],
                "label": segment["label"],
            }
            flattened_data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(flattened_data)
    filename = os.path.join(BASE_DIR, f"prediction_{timestamp}.csv")
    df.to_csv(filename, index=False)
    return filename