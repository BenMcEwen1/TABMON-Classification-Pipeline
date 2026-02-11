import os
import duckdb
import pandas as pd
from datetime import datetime

class ParquetDatabase:
    
    def __init__(self, parquet_dir="./pipeline/outputs/demo"):
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
            
            # device view
            #self.con.execute("""
            #    CREATE OR REPLACE VIEW devices AS
            #    SELECT DISTINCT
            #        device_id,
            #        COALESCE(MAX(country), 'unknown') as country,
            #        MAX(lat) as lat,
            #        MAX(lng) as lng,
            #        MAX(model) as model_name,
            #        MAX(model_checkpoint) as model_checkpoint,
            #        MAX(datetime) AS date_updated
            #    FROM all_data
            #    GROUP BY device_id
            #""")
            

            # device view
            self.con.execute("""
                CREATE OR REPLACE VIEW devices AS
                SELECT DISTINCT
                    device_id,
                    COALESCE(MAX(country), 'unknown') as country,
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
                    MAX(REPLACE(SUBSTRING(filename, 1, 24), '_', ':')) as date_recorded
                FROM all_data
                GROUP BY filename, device_id
            """)

            # segments view            
            #self.con.execute("""
            #    CREATE OR REPLACE VIEW segments AS
            #    SELECT DISTINCT
            #        CONCAT(filename, '_', CAST("start time" AS VARCHAR)) as segment_id,
            #        filename,
            #        "start time" as start_time,
            #        3 as duration,
            #        MAX(uncertainty) as uncertainty,
            #        MAX(energy) as energy,
            #        MAX(datetime) as date_processed,
            #        NULL as label,
            #        NULL as notes
            #    FROM all_data
            #    GROUP BY filename, "start time"
            #""")

            # segments view            
            self.con.execute("""
                CREATE OR REPLACE VIEW segments AS
                SELECT DISTINCT
                    CONCAT(filename, '_', CAST("start time" AS VARCHAR)) as segment_id,
                    filename,
                    "start time" as start_time,
                    3 as duration,
                    MAX("max uncertainty") as uncertainty,
                FROM all_data
                GROUP BY filename, "start time"
            """)

            # predictions view
            self.con.execute("""
                CREATE OR REPLACE VIEW predictions AS
                SELECT
                    CONCAT(a.filename, '_', CAST(a."start time" AS VARCHAR)) AS segment_id,
                    "scientific name" as predicted_species,
                    MAX(confidence) as confidence  -- Take highest confidence if duplicates
                FROM all_data a
                JOIN segments s ON 
                    a.filename = s.filename AND 
                    a."start time" = s.start_time
                WHERE "scientific name" IS NOT NULL
                GROUP BY a.filename, a."start time", "scientific name"  -- Group by segment and species
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
        partition_filters = []
        regular_filters = []
        having_clause = ""
        not_having_clause = ""

        if filters:
            if getattr(filters, 'device_id', None):
                partition_filters.append(f"d.device_id = '{filters.device_id}'")
            if getattr(filters, 'country', None):
                partition_filters.append(f"d.country = '{filters.country}'") 
            if getattr(filters, 'predicted_species', None):
                species_filter = filters.predicted_species
                having_clause = f"HAVING LIST_HAS(ARRAY_AGG(p.predicted_species), '{species_filter}')"
            if getattr(filters, 'filtered_species', None):
                filtered_species_filter = filters.filtered_species
                not_having_clause = f"HAVING NOT LIST_HAS(ARRAY_AGG(p.predicted_species), '{filtered_species_filter}')"   
            if getattr(filters, 'confidence', None):
                regular_filters.append(f"p.confidence > {filters.confidence}")
            if getattr(filters, 'uncertainty', None):
                regular_filters.append(f"s.uncertainty > {filters.uncertainty}")
            if getattr(filters, 'start_date', None):
                regular_filters.append(f"a.date_recorded >= '{filters.start_date}'")
            if getattr(filters, 'end_date', None):
                regular_filters.append(f"a.date_recorded <= '{filters.end_date}'")

        # Build WHERE clause
        all_filters = partition_filters + regular_filters
        where_clause = ""
        if all_filters:
            where_clause = "WHERE " + " AND ".join(all_filters)

        # Apply LIMIT separately (if given)
        order_clause = ""
        limit_clause = ""
        if getattr(filters, 'query_limit', None):
            order_clause = "ORDER BY RANDOM()"
            if filters.stratified:
                limit_clause = f"LIMIT {5*filters.query_limit}"
            else:
                limit_clause = f"LIMIT {filters.query_limit}"

        print(filters)

        main_query = f"""
            SELECT 
                s.*, 
                a.filename AS audio_filename, 
                a.date_recorded,
                d.device_id, 
                d.country,
                ARRAY_AGG(p.predicted_species) FILTER (WHERE p.predicted_species IS NOT NULL) AS predicted_species_list,
                ARRAY_AGG(p.confidence) FILTER (WHERE p.confidence IS NOT NULL) AS confidence_list
            FROM segments s
            JOIN audio a ON s.filename = a.filename
            JOIN devices d ON a.device_id = d.device_id
            LEFT JOIN predictions p ON p.segment_id = s.segment_id
            {where_clause}
            GROUP BY 
                s.segment_id, s.filename, s.start_time, s.duration, s.uncertainty, 
                a.filename, a.date_recorded,
                d.device_id, d.country
            {having_clause}
            {not_having_clause}
            {order_clause}
            {limit_clause}
        """

        return self.execute_query(main_query)

