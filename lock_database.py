from app.database import SessionLocal, initialize_database
from sqlalchemy import text
import time

def lock_database():
    """Acquire an exclusive lock on the database for 60 seconds to simulate contention."""
    initialize_database()  # Ensure tables are created before locking

    create_session = SessionLocal()
    session = create_session()
    try:
        session.execute(text("BEGIN EXCLUSIVE TRANSACTION;"))  # Lock DB for writes
        print("Lock acquired. Holding for 60 seconds...")
        time.sleep(60)  # Simulate a long-running write operation
    finally:
        session.commit()  # Release lock
        session.close()
        print("Lock released.")

if __name__ == "__main__":
    lock_database()