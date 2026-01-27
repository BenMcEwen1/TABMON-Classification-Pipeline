"""
This script:
- Merges multilabel predictions and confidence scores to a single row 
- Updates full audio path for prediction
- Adjusts output format required for Listening Lab 

Specify audio/dataset path, path to current parquets, path to new parquets, then run:
```
python merge_partitioned.py
```
"""

import argparse
import os
import time

import duckdb
import numpy as np
import pandas as pd

DEFAULT_DATASET_PATH = "/DYNI/tabmon/tabmon_data"                   # To check if audio exists
DEFAULT_INPUT_PATH = "pipeline/outputs/test_db_small"    # Directory for current parquet files
DEFAULT_OUTPUT_PATH = "pipeline/outputs/Listening_Lab"              # Directory for new merged parquet files


COUNTRY_TO_FOLDER = {
    "France": "proj_tabmon_NINA_FR",
    "Norway": "proj_tabmon_NINA",
    "Netherlands": "proj_tabmon_NINA_NL",
    "Spain": "proj_tabmon_NINA_ES",
}


def bugg_id_to_folder(id_str):
    """Convert a device_id string to the bugg directory name."""
    padded_id = id_str.rjust(15, "0")
    return f"bugg_RPiID-1{padded_id}"




def aggregate_parquet(parquet_path):
    """Load a single parquet file and aggregate to one row per segment."""
    con = duckdb.connect(database=":memory:")
    con.execute(f"""
        CREATE OR REPLACE VIEW data AS
        SELECT * FROM read_parquet('{parquet_path}')
    """)

    count = con.execute("SELECT COUNT(*) FROM data").fetchone()[0]
    if count == 0:
        con.close()
        return None, 0

    query = """
        SELECT
            filename,
            deployment_id,
            "start time",
            ARRAY_AGG("scientific name") AS "scientific name",
            ARRAY_AGG(confidence) AS confidence,
            MAX("max uncertainty") AS "max uncertainty"
        FROM data
        GROUP BY filename, deployment_id, "start time"
        ORDER BY filename, "start time"
    """
    df = con.execute(query).fetchdf()
    con.close()
    return df, count


def resolve_conf_folder(dataset_path, country, device_id):
    """Resolve the conf_folder for a given country/device_id pair."""
    country_folder = COUNTRY_TO_FOLDER.get(country)
    if country_folder is None:
        return None

    bugg_folder = bugg_id_to_folder(device_id)
    bugg_path = os.path.join(dataset_path, country_folder, bugg_folder)

    if not os.path.exists(bugg_path):
        print(f"  Warning: Path not found: {bugg_path}")
        return None

    conf_folders = [
        f
        for f in os.listdir(bugg_path)
        if os.path.isdir(os.path.join(bugg_path, f))
    ]

    if len(conf_folders) == 0:
        return None
    elif len(conf_folders) == 1:
        return conf_folders[0]
    else:
        tabmon_folders = [f for f in conf_folders if "TABMON" in f.upper()]
        if tabmon_folders:
            return sorted(tabmon_folders)[0]
        return sorted(conf_folders)[0]


def add_full_paths(df, country_folder, bugg_folder, conf_folder):
    """Vectorised fullPath construction (no row-by-row apply)."""
    if conf_folder is None:
        df["fullPath"] = None
    else:
        prefix = f"{country_folder}/{bugg_folder}/{conf_folder}/"
        df["fullPath"] = prefix + df["filename"]
    return df


def walk_parquet_files(input_dir):
    """Yield (country, device_id, parquet_path, filename) for each parquet file"""
    for country_dir in sorted(os.listdir(input_dir)):
        if not country_dir.startswith("country="):
            continue
        country = country_dir.split("=", 1)[1]
        country_path = os.path.join(input_dir, country_dir)

        for device_dir in sorted(os.listdir(country_path)):
            if not device_dir.startswith("device_id="):
                continue
            device_id = device_dir.split("=", 1)[1]
            device_path = os.path.join(country_path, device_dir)

            for fname in sorted(os.listdir(device_path)):
                if fname.endswith(".parquet"):
                    yield country, device_id, os.path.join(device_path, fname), fname


def main():
    parser = argparse.ArgumentParser(
        description="Export annotation database from merged predictions (file-by-file)."
    )
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET_PATH,
        help="Root path to the audio dataset (e.g., /DYNI/tabmon/tabmon_data)",
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_PATH,
        help="Input parquet directory (default: pipeline/outputs/merged_predictions_light)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_PATH,
        help="Output directory mirroring input structure (default: pipeline/outputs/annotation_db)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files. If not set, existing files are skipped (default: False)",
    )
    args = parser.parse_args()

    total_start = time.time()
    total_rows_in = 0
    total_segments_out = 0
    files_processed = 0
    files_skipped = 0

    # Cache conf_folder lookups per (country, device_id)
    conf_cache = {}

    for country, device_id, parquet_path, parquet_name in walk_parquet_files(args.input_dir):
        # Check if output already exists
        out_dir = os.path.join(
            args.output_dir, f"country={country}", f"device_id={device_id}"
        )
        out_path = os.path.join(out_dir, parquet_name)

        if os.path.exists(out_path) and not args.overwrite:
            files_skipped += 1
            continue

        file_start = time.time()

        # Aggregate this single file
        df, rows_in = aggregate_parquet(parquet_path)
        if df is None or df.empty:
            print(f"  [{country}/{device_id}/{parquet_name}] empty, skipping")
            continue

        # Resolve conf_folder (cached per device)
        cache_key = (country, device_id)
        if cache_key not in conf_cache:
            conf_cache[cache_key] = resolve_conf_folder(
                args.dataset_path, country, device_id
            )
        conf_folder = conf_cache[cache_key]

        # Build fullPath column
        country_folder = COUNTRY_TO_FOLDER.get(country, country)
        bugg_folder = bugg_id_to_folder(device_id)
        df = add_full_paths(df, country_folder, bugg_folder, conf_folder)

        # Add placeholder userID
        df["userID"] = np.nan

        # Reorder columns
        df = df[
            [
                "filename",
                "deployment_id",
                "fullPath",
                "start time",
                "confidence",
                "scientific name",
                "max uncertainty",
                "userID",
            ]
        ]

        # Write to mirrored output path
        os.makedirs(out_dir, exist_ok=True)
        df.to_parquet(out_path, index=False, engine="pyarrow")

        elapsed = time.time() - file_start
        total_rows_in += rows_in
        total_segments_out += len(df)
        files_processed += 1

        print(
            f"  [{country}/{device_id}/{parquet_name}] "
            f"{rows_in:,} rows -> {len(df):,} segments [{elapsed:.2f}s]"
        )

    total_elapsed = time.time() - total_start
    print(f"\nDone. Processed {files_processed} files, skipped {files_skipped} in {total_elapsed:.1f}s")
    print(f"  Total rows in:     {total_rows_in:,}")
    print(f"  Total segments out: {total_segments_out:,}")
    print(f"  Output directory:  {args.output_dir}/")


if __name__ == "__main__":
    main()
