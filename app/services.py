import os
from app.database import Device, Audio, Segment, Predictions
from sqlalchemy import and_, cast, JSON
from sqlalchemy.orm import selectinload
import pandas as pd
from datetime import datetime
import time

def normalise(predictions, db):
    """
    Normalize the database by splitting it into smaller linked tables.
    """
    start = time.time()
    # Fetch existing devices, audio, and segments in bulk
    existing_devices = {d.device_id: d.id for d in db.query(Device).all()}
    existing_audio = {a.filename: a.id for a in db.query(Audio).all()}
    existing_segments = {
        (s.audio_id, s.start_time): s.id for s in db.query(Segment).all()
    }

    # Prepare bulk insert lists
    new_devices = []
    new_audio = []
    new_segments = []
    new_predictions = []

    # Device mapping
    device_map = {}
    for _, row in predictions.iterrows():
        if row["device_id"] not in existing_devices and row["device_id"] not in device_map:
            new_device = Device(
                device_id=row["device_id"],
                lat=row["lat"],
                lng=row["lng"],
                model_name=row["model"],
                model_checkpoint=row["model_checkpoint"],
                date_updated=datetime.now(),
            )
            new_devices.append(new_device)
            device_map[row["device_id"]] = None  # Placeholder for ID
        else:
            device_map[row["device_id"]] = existing_devices.get(row["device_id"])

    # Bulk insert devices and update device_map
    db.add_all(new_devices)
    db.flush()  # Assign IDs to new devices
    for device in new_devices:
        device_map[device.device_id] = device.id

    # Audio mapping
    audio_id_map = {}
    for _, row in predictions.iterrows():
        if row["filename"] not in existing_audio and row["filename"] not in audio_id_map:
            try:
                date_obj = datetime.strptime(row["filename"].split(".")[0], "%Y-%m-%dT%H_%M_%S")
            except:
                date_obj = None
            new_audio.append(
                Audio(
                    filename=row["filename"],
                    device_id=device_map[row["device_id"]],
                    date_recorded=date_obj,
                )
            )
            audio_id_map[row["filename"]] = None  # Placeholder for ID
        else:
            audio_id_map[row["filename"]] = existing_audio.get(row["filename"])

    # Bulk insert audio and update audio_id_map
    db.add_all(new_audio)
    db.flush()  # Assign IDs to new audio
    for audio in new_audio:
        audio_id_map[audio.filename] = audio.id

    # Segment mapping
    segment_id_map = {}
    for _, row in predictions.iterrows():
        key = (audio_id_map[row["filename"]], row["start time"])
        if key not in existing_segments and key not in segment_id_map:
            index = int(row["start time"] / 3)

            if type(row["energy"]) == str:
                row["energy"] = eval(row["energy"])

            new_segments.append(
                Segment(
                    start_time=row["start time"],
                    filename=os.path.splitext(row["filename"])[0]
                    + f"_{row['device_id']}_{index}.wav",
                    duration=3,
                    uncertainty=row["uncertainty"],
                    energy=row["energy"],
                    date_processed=datetime.now(),
                    label=None,
                    notes=None,
                    audio_id=audio_id_map[row["filename"]],
                    embedding_id=1,
                )
            )
            segment_id_map[key] = None  # Placeholder for ID
        else:
            segment_id_map[key] = existing_segments.get(key)

    # Bulk insert segments and update segment_id_map
    db.add_all(new_segments)
    db.flush()
    for segment in new_segments:
        segment_id_map[(segment.audio_id, segment.start_time)] = segment.id

    # Predictions
    for _, row in predictions.iterrows():
        audio_id = audio_id_map[row["filename"]]
        key = (audio_id, row["start time"])
        segment_id = segment_id_map[key]
        existing_prediction = db.query(Predictions).filter(
            Predictions.predicted_species == row["scientific name"],
            Predictions.confidence == row["confidence"],
            Predictions.segment_id == segment_id,
        ).first()
        if not existing_prediction:
            new_predictions.append(
                Predictions(
                    predicted_species=row["scientific name"],
                    confidence=row["confidence"],
                    segment_id=segment_id,
                )
            )

    # Bulk insert predictions
    db.add_all(new_predictions)

    try:
        db.commit()
        end = time.time()
        print(f"Write successful in {end - start} seconds")
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

def apply_filters(parameters, db):
    # Start query from Segments instead of Predictions
    query = db.query(Segment)\
        .join(Audio)\
        .join(Device)\
        .options(selectinload(Segment.predictions))  # Optimized relationship loading

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
        filters.append(cast(Segment.energy[parameters.indice], JSON) >= parameters.energy)
    if parameters.uncertainty is not None:
        filters.append(Segment.uncertainty >= parameters.uncertainty)
    if parameters.annotated:
        filters.append(Segment.label != None)

    # Filtering Predictions within the same query
    if parameters.predicted_species or parameters.confidence is not None:
        query = query.join(Predictions)  # Join Predictions table
        if parameters.predicted_species:
            filters.append(Predictions.predicted_species == parameters.predicted_species)
        if parameters.confidence is not None:
            filters.append(Predictions.confidence >= parameters.confidence)

    # Apply filters
    if filters:
        query = query.filter(and_(*filters))

    # Apply ordering if needed (e.g., newest segments first)
    query = query.order_by(Segment.id.desc())

    # Apply limit for performance
    query = query.limit(parameters.query_limit)

    segments = query.all()  # Single optimized DB query
    return segmentsWithPredictions(segments, db), segments

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
