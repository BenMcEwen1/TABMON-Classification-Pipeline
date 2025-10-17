import os
import pandas as pd
from datetime import datetime
import time
from typing import Optional
from fastapi import Depends

from app.schema import Filter
from app.database import ParquetDatabase


db_instance = ParquetDatabase()

def get_db():
    return db_instance

def stats(db: ParquetDatabase = Depends(get_db)):
    filters = Filter(start_date=None,
                    end_date=None,
                    country=None,
                    deployment_id=None,
                    device_id=None,
                    confidence=None,
                    predicted_species=None,
                    uncertainty=None,
                    energy=None,
                    annotated=None,
                    query_limit=100)
    
    results = db.get_segments_with_predictions(filters)
    return results

def normalise(predictions_df, db):
    """Simplified: Just save predictions as a parquet file for DuckDB to read"""
    if predictions_df is None or predictions_df.empty:
        print("No predictions to normalize")
        return None
        
    # Generate a timestamped filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(db.parquet_dir, f"predictions_{timestamp}.parquet")
    
    print(f"Saving {len(predictions_df)} predictions to {output_path}")
    
    # Ensure the output directory exists
    os.makedirs(db.parquet_dir, exist_ok=True)
    
    # Save the raw predictions directly to parquet
    predictions_df.to_parquet(output_path, index=False)
    
    # Refresh the database views to include the new data
    db._register_views()
    
    return {
        "status": "success",
        "file": output_path,
        "records": len(predictions_df)
    }

if __name__ == "__main__":
    confidence = stats(db_instance)['confidence_list']
    print(confidence)