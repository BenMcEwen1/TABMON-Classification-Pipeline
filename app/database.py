import os
import duckdb
import pandas as pd
from datetime import datetime

class ParquetDatabase:
    
    def __init__(self, parquet_dir="./pipeline/outputs/predictions"):
        self.parquet_dir = parquet_dir
        self.con = duckdb.connect(database=":memory:")
        os.makedirs(parquet_dir, exist_ok=True)
        self._register_views()
        
    def _register_views(self):
        try:
            self.con.execute(f"""
                CREATE OR REPLACE VIEW all_data AS 
                SELECT * FROM read_parquet('{self.parquet_dir}/**/*.parquet', hive_partitioning=True)
            """)
            
            self.con.execute("""
                CREATE OR REPLACE VIEW devices AS
                SELECT DISTINCT
                    device_id,
                    COALESCE(MAX(country), 'unknown') as country,
                    MAX(lat) as lat,
                    MAX(lng) as lng,
                    MAX(model) as model_name,
                    MAX(model_checkpoint) as model_checkpoint,
                    MAX(datetime) as date_updated
                FROM all_data
                GROUP BY device_id
            """)
            
            # audio view
            self.con.execute("""
                CREATE OR REPLACE VIEW audio AS
                SELECT DISTINCT
                    row_number() OVER() as id,
                    filename,
                    device_id,
                    MAX(datetime) as date_recorded
                FROM all_data
                GROUP BY filename, device_id
            """)
            
            # segments view
            self.con.execute("""
                CREATE OR REPLACE VIEW segments AS
                SELECT DISTINCT
                    row_number() OVER() as id,
                    filename,
                    "start time" as start_time,
                    3 as duration,
                    MAX(uncertainty) as uncertainty,
                    MAX(energy) as energy,
                    MAX(datetime) as date_processed,
                    NULL as label,
                    NULL as notes
                FROM all_data
                GROUP BY filename, "start time"
            """)
            
            # predictions view
            self.con.execute("""
                CREATE OR REPLACE VIEW predictions AS
                SELECT
                    row_number() OVER() as id,
                    s.id as segment_id,
                    "scientific name" as predicted_species,
                    MAX(confidence) as confidence  -- Take highest confidence if duplicates
                FROM all_data a
                JOIN segments s ON 
                    a.filename = s.filename AND 
                    a."start time" = s.start_time
                WHERE "scientific name" IS NOT NULL
                GROUP BY s.id, "scientific name"  -- Group by segment and species
            """)
            
        except Exception as e:
            print(f"Error registering views: {e}")
            import traceback
            traceback.print_exc()
    
    def execute_query(self, query):
        """Execute a SQL query."""
        try:
            result = self.con.execute(query).fetchdf()
            return result
        except Exception as e:
            print(f"Error executing query: {e}")
            print(f"Query was: {query}")
            return pd.DataFrame()
    
    def get_status(self):
        """Get database status."""
        tables = ["all_data", "devices", "audio", "segments", "predictions"]
        stats = {}
        
        for table in tables:
            try:
                count_df = self.con.execute(f"SELECT COUNT(*) as count FROM {table}").fetchdf()
                stats[table] = int(count_df['count'].iloc[0])
            except:
                stats[table] = 0
                
        return stats
        
    # Queries

    def get_countries(self):
        return self.execute_query("SELECT DISTINCT country FROM all_data")

    def get_devices(self):
        return self.execute_query("SELECT DISTINCT device_id FROM all_data")
        
    def get_audio_files(self):
        return self.execute_query("SELECT * FROM audio")
        
    def get_segments(self, segment_id=None):
        if segment_id:
            return self.execute_query(f"SELECT * FROM segments WHERE id = {segment_id}")
        return self.execute_query("SELECT * FROM segments")
        
    def get_predictions(self, segment_id=None):
        if segment_id:
            return self.execute_query(f"SELECT * FROM predictions WHERE segment_id = {segment_id}")
        return self.execute_query("SELECT * FROM predictions") 

    def get_segments_with_predictions(self, filters=None):
        
        # HAVE TO START WITH PARTITION FILTERS
        partition_filters = []
        regular_filters = []

        print(filters)
        
        if filters:
            # Partition filters so that queries go faster
            if hasattr(filters, 'device_id') and filters.device_id:
                partition_filters.append(f"d.device_id = '{filters.device_id}'")
            if hasattr(filters, 'country') and filters.country:
                partition_filters.append(f"d.country = '{filters.country}'")
            
            # The other filters (the regular ones)
            if hasattr(filters, 'predicted_species') and filters.predicted_species:
                regular_filters.append(f"p.predicted_species = '{filters.predicted_species}'")
            if hasattr(filters, 'device_id') and filters.device_id:
                regular_filters.append(f"d.device_id = '{filters.device_id}'")
            if hasattr(filters, 'confidence') and filters.confidence:
                regular_filters.append(f"p.confidence > {filters.confidence}")
            if hasattr(filters, 'uncertainty') and filters.uncertainty:
                regular_filters.append(f"s.uncertainty > {filters.uncertainty}")
            if hasattr(filters, 'start_date') and filters.start_date:
                regular_filters.append(f"a.date_recorded >= '{filters.start_date}'")
            if hasattr(filters, 'end_date') and filters.end_date:
                regular_filters.append(f"a.date_recorded <= '{filters.end_date}'")
        
        # start of the query
        segment_id_query = """
            WITH matching_segments AS (
                SELECT DISTINCT s.id
                FROM segments s
                JOIN audio a ON s.filename = a.filename
                JOIN devices d ON a.device_id = d.device_id
                LEFT JOIN predictions p ON p.segment_id = s.id
        """

        print(partition_filters)
        
        # add the filters (partition e.g. country and device + the other filters)
        if partition_filters:
            segment_id_query += " WHERE " + " AND ".join(partition_filters)
            
            if regular_filters:
                segment_id_query += " AND " + " AND ".join(regular_filters)
        elif regular_filters:
            segment_id_query += " WHERE " + " AND ".join(regular_filters)
        
        segment_id_query += """
            )
            SELECT id FROM matching_segments
        """

        
        if filters and hasattr(filters, 'query_limit') and filters.query_limit:
            segment_id_query += f" LIMIT {filters.query_limit}"
        
        # MAIN QUERY
        main_query = f"""
            SELECT 
                s.*, 
                a.filename as audio_filename, 
                a.date_recorded,
                d.device_id, 
                d.country,
                ARRAY_AGG(p.predicted_species) FILTER (WHERE p.predicted_species IS NOT NULL) as predicted_species_list,
                ARRAY_AGG(p.confidence) FILTER (WHERE p.confidence IS NOT NULL) as confidence_list
            FROM segments s
            JOIN audio a ON s.filename = a.filename
            JOIN devices d ON a.device_id = d.device_id
            LEFT JOIN predictions p ON p.segment_id = s.id
            WHERE s.id IN ({segment_id_query})
            GROUP BY s.id, s.filename, s.start_time, s.duration, s.uncertainty, 
                    s.energy, s.date_processed, s.label, s.notes,
                    a.filename, a.date_recorded, d.device_id, d.country
        """
        
        return self.execute_query(main_query)