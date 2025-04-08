import os
import duckdb
import pandas as pd
from datetime import datetime

current_dir = "./pipeline/outputs" #os.path.dirname(os.path.abspath(__file__))

# Add these model classes to replicate the SQLAlchemy models structure
class Device:
    """Device model for compatibility with existing code."""
    @classmethod
    def from_dict(cls, data):
        device = cls()
        for key, value in data.items():
            setattr(device, key, value)
        return device

class Audio:
    """Audio model for compatibility with existing code."""
    @classmethod
    def from_dict(cls, data):
        audio = cls()
        for key, value in data.items():
            setattr(audio, key, value)
        return audio

class Segment:
    """Segment model for compatibility with existing code."""
    @classmethod
    def from_dict(cls, data):
        segment = cls()
        for key, value in data.items():
            setattr(segment, key, value)
        return segment

class Prediction:
    """Prediction model for compatibility with existing code."""
    @classmethod
    def from_dict(cls, data):
        prediction = cls()
        for key, value in data.items():
            setattr(prediction, key, value)
        return prediction

class ParquetDatabase:
    def __init__(self, parquet_dir=f"{current_dir}/"):
        """Initialize DuckDB connection to read Parquet files from a directory."""
        self.parquet_dir = parquet_dir
        self.con = duckdb.connect(database=":memory:")
        
        # Create directory if it doesn't exist
        os.makedirs(parquet_dir, exist_ok=True)
        
        # Register views for expected parquet files
        self._register_views()
    
    def _register_views(self):
        """Register SQL views with filters to simulate a relational structure"""
        parquet_path = f"{self.parquet_dir}/*.parquet"
        
        try:
            # Register the base view with all data
            self.con.execute(f"DROP VIEW IF EXISTS all_data")
            self.con.execute(f"CREATE VIEW all_data AS SELECT * FROM parquet_scan('{parquet_path}')")
            
            # 1. Create a devices view - MUCH more selective
            self.con.execute("DROP VIEW IF EXISTS devices")
            self.con.execute("""
                CREATE VIEW devices AS
                SELECT
                    device_id,
                    'Unknown' as country,
                    MAX(lat) as lat,
                    MAX(lng) as lng,
                    MAX(model) as model_name,
                    MAX(model_checkpoint) as model_checkpoint,
                    MAX(datetime) as date_updated
                FROM all_data
                GROUP BY device_id
            """)
            
            # 2. Create an audio view - WITH PROPER GROUPING
            self.con.execute("DROP VIEW IF EXISTS audio")
            self.con.execute("""
                CREATE VIEW audio AS
                SELECT
                    row_number() OVER() as id,
                    filename,
                    device_id,
                    MAX(datetime) as date_recorded
                FROM all_data
                GROUP BY filename, device_id
            """)
            
            # 3. Create a segments view - DISTINCT BY FILENAME+TIME
            self.con.execute("DROP VIEW IF EXISTS segments")
            self.con.execute("""
                CREATE VIEW segments AS
                SELECT
                    row_number() OVER() as id,
                    ad.filename,
                    ad."start time" as start_time,
                    3 as duration,
                    MAX(ad.uncertainty) as uncertainty,
                    MAX(ad.energy) as energy,
                    MAX(ad.datetime) as date_processed,
                    NULL as label,
                    NULL as notes,
                    a.id as audio_id,
                    1 as embedding_id
                FROM all_data ad
                JOIN audio a ON ad.filename = a.filename
                GROUP BY ad.filename, ad."start time", a.id
            """)
            
            # 4. Create predictions view - ONE PER SEGMENT+SPECIES
            self.con.execute("DROP VIEW IF EXISTS predictions")
            self.con.execute("""
                CREATE VIEW predictions AS
                SELECT
                    row_number() OVER() as id,
                    ad."scientific name" as predicted_species,
                    MAX(ad.confidence) as confidence,
                    s.id as segment_id
                FROM all_data ad
                JOIN segments s ON 
                    ad.filename = s.filename AND 
                    ad."start time" = s.start_time
                WHERE ad."scientific name" IS NOT NULL
                GROUP BY ad."scientific name", s.id
            """)
            
        except Exception as e:
            print(f"Error registering views: {e}")
            import traceback
            traceback.print_exc()
    
    def get_devices(self, filters=None):
        """Query devices data from parquet files."""
        try:
            query = "SELECT * FROM devices"
            if filters:
                # Add WHERE clauses for filters
                conditions = []
                for key, value in filters.items():
                    conditions.append(f"{key} = '{value}'")
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            return self.con.execute(query).fetchdf()
        except Exception as e:
            print(f"Error querying devices: {e}")
            # Return empty dataframe if table doesn't exist yet
            return self.con.execute("SELECT * FROM range(0) LIMIT 0").fetchdf()
    
    # All other methods remain unchanged
    
    # Add compatibility methods that return objects instead of dataframes
    def query(self, model_class):
        """Compatibility method to mimic SQLAlchemy query interface."""
        if model_class == Device:
            df = self.get_devices()
            return DataFrameQuery(df, Device)
        elif model_class == Audio:
            df = self.get_audio_files()
            return DataFrameQuery(df, Audio)
        elif model_class == Segment:
            df = self.get_segments()
            return DataFrameQuery(df, Segment)
        elif model_class == Prediction:
            df = self.get_predictions()
            return DataFrameQuery(df, Prediction)
        return None
    
    def scan_directory(self):
        """Scan the parquet directory to refresh views and return record counts"""
        # Refresh views
        self._register_views()
        
        # Get counts for each table
        stats = {}
        tables = ["all_data", "devices", "audio", "segments", "predictions"]
        
        for table in tables:
            try:
                # Use DuckDB's information_schema instead of sqlite_master
                view_exists = self.con.execute(
                    f"SELECT table_name FROM information_schema.views WHERE table_name='{table}'"
                ).fetchdf()
                
                if not view_exists.empty:
                    # If view exists, count records
                    count_df = self.con.execute(f"SELECT COUNT(*) as count FROM {table}").fetchdf()
                    stats[table] = int(count_df['count'].iloc[0])
                else:
                    stats[table] = 0
            except Exception as e:
                print(f"Error counting records in {table}: {e}")
                stats[table] = 0
        
        # Get file stats
        try:
            import glob
            files = glob.glob(f"{self.parquet_dir}/*.parquet")
            stats["parquet_files"] = len(files)
        except Exception as e:
            print(f"Error counting parquet files: {e}")
            stats["parquet_files"] = 0
        
        return stats
    
    def execute_query(self, query):
        """Execute a custom SQL query against the DuckDB connection."""
        try:
            print(f"Executing query: {query}")
            result = self.con.execute(query).fetchdf()
            print(f"Query returned {len(result)} rows")
            return result
        except Exception as e:
            print(f"Error executing query: {e}")
            print(f"Failed query: {query}")
            return pd.DataFrame()

# Helper class to mimic SQLAlchemy query interface
class DataFrameQuery:
    def __init__(self, df, model_class):
        self.df = df
        self.model_class = model_class
        
    def filter(self, *args):
        # This is a simplified filter that would need to be expanded
        # to handle actual SQLAlchemy filter conditions
        return self
        
    def all(self):
        return [self.model_class.from_dict(row) for _, row in self.df.iterrows()]
        
    def first(self):
        if len(self.df) > 0:
            return self.model_class.from_dict(self.df.iloc[0].to_dict())
        return None
    

def initialize_database(parquet_dir=None):
    """Initialize the database by creating a ParquetDatabase instance."""
    if parquet_dir is None:
        parquet_dir = f"{current_dir}/"
    
    db = ParquetDatabase(parquet_dir)
    stats = db.scan_directory()
    print(f"Database initialized. Found: {stats}")
    return db