import torch
import faiss
import os
from app.database import SessionLocal, Device, Audio, Segment, Predictions
from types import SimpleNamespace
from sqlalchemy import and_, cast, JSON
import pandas as pd
from datetime import datetime

def normalise(predictions, db):
    """
    This function normalised the database i.e. splits one large table into smaller linked tables so that data can be more efficiently queried.
    """
    device_map = {}
    device = predictions[["device_id", "lat", "lng", "model", "model_checkpoint"]].drop_duplicates()
    for _,row in predictions.iterrows():
        existing_device = db.query(Device).filter(Device.device_id == row["device_id"]).first()
        if not existing_device:
            device = Device(
                device_id = row["device_id"],
                lat = row["lat"],
                lng = row["lng"],
                model_name = row["model"],
                model_checkpoint = row["model_checkpoint"],
                date_updated = datetime.now()
            )
            db.add(device)
            db.flush()
            device_map[row["device_id"]] = device.id
        else:
            device_map[row["device_id"]] = existing_device.id

    audio_id_map = {}
    audio_data = predictions[["filename", "device_id"]]
    for _,row in audio_data.iterrows():
        existing_audio = db.query(Audio).filter(Audio.filename == row["filename"]).first()
        if not existing_audio:
            audio = Audio(
                filename=row["filename"],
                device_id=device_map[row["device_id"]],
            )
            db.add(audio)
            db.flush()
            audio_id_map[row["filename"]] = audio.id
        else: 
            audio_id_map[row["filename"]] = existing_audio.id

    segment_id_map = {}
    segment_data = predictions[["filename", "device_id", "start time", "uncertainty", "energy"]]
    for _,row in segment_data.iterrows():
        existing_segment = db.query(Segment).filter(Segment.audio_id == audio_id_map[row["filename"]], Segment.start_time == row['start time']).first()
        index = int(row["start time"]/3)
        
        if not existing_segment:
            segment = Segment(
                start_time=row["start time"],
                filename=os.path.splitext(row["filename"])[0] + f"_{row['device_id']}_{index}.wav",
                duration=3,
                uncertainty=row["uncertainty"],
                energy=row["energy"],
                date_processed=datetime.now(),
                label=None,
                notes=None,
                audio_id=audio_id_map[row["filename"]],
                embedding_id=1
            )
            db.add(segment)
            db.flush()
            segment_id_map[(row["filename"], row["start time"])] = segment.id
        else: 
            segment_id_map[(row["filename"], row["start time"])] = existing_segment.id

    prediction_data = predictions[["filename", "scientific name", "confidence", "start time"]]
    for _, row in prediction_data.iterrows():
        existing_prediction = db.query(Predictions).filter(
            Predictions.predicted_species == row["scientific name"],
            Predictions.confidence == row["confidence"],
            Predictions.segment_id == segment_id_map[(row["filename"], row["start time"])]
            ).first()
        
        if not existing_prediction:
            prediction = Predictions(
                predicted_species=row["scientific name"],
                confidence=row["confidence"],
                segment_id=segment_id_map[(row["filename"], row["start time"])],  # Link to the segment
            )
            db.add(prediction)
    try:
        db.commit()
        print("Write successful")
        return {"status": "data added successfully"}
    except Exception as error:
        db.rollback()
        print(f"Write failed: {error}")
        return None

def segmentsWithPredictions(segments, db):
    results = [{
        **segment.__dict__,
        "predictions": db.query(Predictions).filter(Predictions.segment_id == segment.id).all()
        }
        for segment in segments
    ]
    return results

def apply_filters_body(parameters, db):
    query = db.query(Predictions).join(Segment).join(Audio)

    filters = []
    if parameters.country:
        filters.append(Device.country == parameters.country)
    if parameters.device_id:
        filters.append(Device.device_id == parameters.device_id)
    if parameters.start_date:
        filters.append(Audio.date_recorded >= parameters.start_date)
    if parameters.end_date:
        filters.append(Audio.date_recorded <= parameters.end_date)
    if parameters.energy is not None and parameters.indice is not None:
        indice = parameters.indice
        filters.append(cast(Segment.energy[indice], JSON) >= parameters.energy)
    if parameters.uncertainty is not None:
        filters.append(Segment.uncertainty >= parameters.uncertainty)
    if parameters.confidence is not None:
        filters.append(Predictions.confidence >= parameters.confidence)
    if parameters.annotated:
        filters.append(Segment.label != None)
    if parameters.predicted_species:
        filters.append(Predictions.predicted_species in parameters.predicted_species)

    # Apply filters if any
    if filters:
        query = query.filter(and_(*filters))

    segment_ids = [prediction.segment_id for prediction in query]
    segment_ids = list(dict.fromkeys(segment_ids)) # deduplicate
    segments = [db.query(Segment).filter(Segment.id == segment_id).first()  for segment_id in segment_ids]
    
    if parameters.query_limit is not None:
        segments = segments[:parameters.query_limit]
    return segmentsWithPredictions(segments, db), segments

def apply_filters(filters):
    parameters = SimpleNamespace(**filters)
    query = parameters.db.query(Predictions).join(Segment).join(Audio)

    filters = []
    if parameters.country:
        filters.append(Device.country == parameters.country)
    if parameters.device_id:
        filters.append(Device.device_id == parameters.device_id)
    if parameters.start_date:
        filters.append(Audio.date_recorded >= parameters.start_date)
    if parameters.end_date:
        filters.append(Audio.date_recorded <= parameters.end_date)
    if parameters.energy is not None and parameters.indice is not None:
        indice = parameters.indice
        filters.append(cast(Segment.energy[indice], JSON) >= parameters.energy)
    if parameters.uncertainty is not None:
        filters.append(Segment.uncertainty >= parameters.uncertainty)
    if parameters.confidence is not None:
        filters.append(Predictions.confidence >= parameters.confidence)
    if parameters.annotated:
        filters.append(Segment.label != None)
    if parameters.predicted_species:
        filters.append(Predictions.predicted_species == parameters.predicted_species)

    # Apply filters if any
    if filters:
        query = query.filter(and_(*filters))

    segment_ids = [prediction.segment_id for prediction in query]
    segment_ids = list(dict.fromkeys(segment_ids)) # deduplicate
    segments = [parameters.db.query(Segment).filter(Segment.id == segment_id).limit(parameters.limit).first()  for segment_id in segment_ids]
    return segmentsWithPredictions(segments, parameters.db), segments

def flatten(data, BASE_DIR, timestamp):
        # Flatten data
        flattened_data = []

        for segment in data:
            for prediction in segment["predictions"]:
                row = {
                    "segment_id": segment["id"],
                    "duration": segment["duration"],
                    "uncertainty": segment["uncertainty"],
                    "energy": segment["energy"],
                    "start_time": segment["start_time"],
                    "filename": segment["filename"],
                    "date_processed": segment["date_processed"],
                    "audio_id": segment["audio_id"],
                    "predicted_species": prediction.predicted_species,
                    "confidence": prediction.confidence,
                    "notes": segment["notes"],
                    "label": segment["label"],
                }
                flattened_data.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(flattened_data)
        filename = os.path.join(BASE_DIR, f"prediction_{timestamp}.csv")
        df.to_csv(filename, index=False)
        return filename

embeddings_path = "./embeddings.bin"

def add_embedding(embedding:torch.Tensor, dimension:int=2):
    if os.path.exists(embeddings_path):
        index = faiss.read_index(embeddings_path)
    else:
        index = faiss.IndexFlatL2(dimension) 

    index.add(embedding)
    embedding_id = index.ntotal - 1
    return index, embedding_id

def delete_embedding():
    index = faiss.read_index(embeddings_path)
    faiss.write_index(index, embeddings_path)
    return index

def get_embedding(index, embedding_id:int):
    index = faiss.read_index(embeddings_path)
    embedding = index.reconstruct(embedding_id)
    return embedding

