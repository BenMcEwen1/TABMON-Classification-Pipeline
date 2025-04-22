import os
import pandas as pd
from datetime import datetime
import time

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