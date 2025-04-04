from pipeline.config import *
from app.services import normalise
from app.database import SessionLocal
from sqlalchemy.exc import OperationalError
import os
import pandas as pd

def reattempt_write():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Find all .csv files in the current directory
    csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv')]

    for file in csv_files:
        print(f"Processing file: {file}")
        predictions = pd.read_csv(os.path.join(current_dir, file))
    
        status = None
        attempts = 1
        while status is None and attempts <= 20:
            try:
                session = SessionLocal()
                db = session()
                status = normalise(predictions, db)
            except OperationalError as e:  # Specifically catch SQLite lock errors
                print(e)
                print(f"[Database locked] attempt {attempts}, retrying...")
                time.sleep(5)  # Short delay before retrying
            except Exception as e:
                print(f"[Unexpected error]: {e}")  # Catch other unexpected errors
                break  # Exit loop on unknown errors
            finally:
                if "db" in locals():
                    db.close()  # Always close the session properly
            attempts += 1
        db.close()
    return status

if __name__ == "__main__":
    reattempt_write()