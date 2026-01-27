"""
Generate a synthetic merged_predictions_light database for performance testing.

Produces rows (unmerged) matching the real parquet schema,
with Hive partitioning by country and device_id.

Usage:
    python generate_test_db.py --target-rows 1000000 --output-dir pipeline/outputs/test_db_small
"""

import argparse
import os
import time

import numpy as np
import pandas as pd


# --- Schema constants ---

COUNTRIES = ["Norway", "Netherlands", "France", "Spain"]

# Realistic device counts per country
DEVICES_PER_COUNTRY = {
    "Norway": 15,
    "Netherlands": 12,
    "France": 10,
    "Spain": 8,
}

# Species pool (real BirdNET species names)
SPECIES_POOL = [
    "Cyanistes caeruleus",
    "Parus major",
    "Erithacus rubecula",
    "Turdus merula",
    "Fringilla coelebs",
    "Phylloscopus collybita",
    "Sylvia atricapilla",
    "Troglodytes troglodytes",
    "Prunella modularis",
    "Aegithalos caudatus",
    "Sitta europaea",
    "Certhia brachydactyla",
    "Dendrocopos major",
    "Columba palumbus",
    "Sturnus vulgaris",
    "Passer domesticus",
    "Carduelis carduelis",
    "Chloris chloris",
    "Corvus corone",
    "Garrulus glandarius",
    "Motacilla alba",
    "Anthus trivialis",
    "Alauda arvensis",
    "Emberiza citrinella",
    "Luscinia megarhynchos",
    "Ficedula hypoleuca",
    "Phoenicurus phoenicurus",
    "Muscicapa striata",
    "Hirundo rustica",
    "Delichon urbicum",
]

MONTHS = [
    "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
    "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
]


def binary_entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def generate_device_id():
    """Generate a realistic 8-char hex device ID."""
    return format(np.random.randint(0, 0x7FFFFFFF), "08x")


def generate_filename(month, day, hour, minute, second, ms):
    """Generate a realistic Bugg audio filename."""
    return f"{month}-{day:02d}T{hour:02d}_{minute:02d}_{second:02d}.{ms:03d}Z.mp3"


def generate_device_data(device_id, country, deployment_id, target_rows, rng):
    """Generate all prediction rows for one device."""
    rows_generated = 0
    all_chunks = []

    # Each recording is ~5 minutes = 300s, segments every 3s = ~100 segments
    segment_duration = 3
    recording_length = 300  # seconds
    segments_per_recording = recording_length // segment_duration

    while rows_generated < target_rows:
        # Generate a batch of recordings
        batch_recordings = min(200, max(1, (target_rows - rows_generated) // (segments_per_recording * 2)))

        for _ in range(batch_recordings):
            if rows_generated >= target_rows:
                break

            month = rng.choice(MONTHS)
            day = rng.integers(1, 29)
            hour = rng.integers(0, 24)
            minute = rng.integers(0, 60)
            second = rng.integers(0, 60)
            ms = rng.integers(0, 1000)
            fname = generate_filename(month, day, hour, minute, second, ms)

            # Generate segments for this recording
            n_segments = rng.integers(50, segments_per_recording + 1)
            start_times = np.arange(0, n_segments * segment_duration, segment_duration)

            # For each segment, generate 1-5 predictions
            preds_per_segment = rng.integers(1, 6, size=len(start_times))

            total_preds = int(preds_per_segment.sum())
            if rows_generated + total_preds > target_rows:
                # Trim to fit
                cumsum = np.cumsum(preds_per_segment)
                remaining = target_rows - rows_generated
                keep_segments = int(np.searchsorted(cumsum, remaining, side="right")) + 1
                keep_segments = min(keep_segments, len(start_times))
                start_times = start_times[:keep_segments]
                preds_per_segment = preds_per_segment[:keep_segments]
                total_preds = int(preds_per_segment.sum())

            # Build arrays
            filenames = np.repeat(fname, total_preds)
            deployment_ids = np.repeat(deployment_id, total_preds)
            starts = np.repeat(start_times, preds_per_segment)

            # Species: pick without replacement within each segment
            species = []
            for n_pred in preds_per_segment:
                species.extend(rng.choice(SPECIES_POOL, size=n_pred, replace=False))

            # Confidence: between 0.1 and 1.0 (already filtered in real data)
            confidences = rng.uniform(0.1, 1.0, size=total_preds).astype(np.float32)

            # Max uncertainty: computed per segment as max binary entropy
            h_values = binary_entropy(confidences)
            segment_indices = np.repeat(np.arange(len(start_times)), preds_per_segment)
            max_h_per_segment = np.zeros(len(start_times))
            np.maximum.at(max_h_per_segment, segment_indices, h_values)
            max_uncertainty = max_h_per_segment[segment_indices].astype(np.float32)

            chunk = pd.DataFrame({
                "filename": filenames,
                "deployment_id": deployment_ids,
                "start time": starts.astype(np.int32),
                "scientific name": species,
                "confidence": confidences,
                "max uncertainty": max_uncertainty,
            })

            all_chunks.append(chunk)
            rows_generated += total_preds

    if not all_chunks:
        return pd.DataFrame()

    return pd.concat(all_chunks, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic merged_predictions_light database for testing."
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=70_000_000,
        help="Target number of prediction rows (default: 70,000,000)",
    )
    parser.add_argument(
        "--output-dir",
        default="pipeline/outputs/test_db",
        help="Output directory (default: pipeline/outputs/test_db)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(seed=42)

    # Calculate total devices and rows per device
    total_devices = sum(DEVICES_PER_COUNTRY.values())
    base_rows_per_device = args.target_rows // total_devices

    print(f"Target: {args.target_rows:,} rows across {total_devices} devices")
    print(f"~{base_rows_per_device:,} rows per device")
    print(f"Output: {args.output_dir}/")
    print()

    total_rows = 0
    total_start = time.time()

    for country, n_devices in DEVICES_PER_COUNTRY.items():
        device_ids = [generate_device_id() for _ in range(n_devices)]

        for i, device_id in enumerate(device_ids):
            # Vary rows per device (+/- 20%)
            variation = rng.uniform(0.8, 1.2)
            device_target = int(base_rows_per_device * variation)
            deployment_id = f"deploy_{country[:2]}_{i:03d}_{device_id}"

            print(f"  Generating {country}/{device_id} ({device_target:,} rows)...", end="", flush=True)
            start = time.time()

            df = generate_device_data(device_id, country, deployment_id, device_target, rng)

            if df.empty:
                print(" skipped (empty)")
                continue

            # Write with Hive partitioning
            output_path = os.path.join(
                args.output_dir,
                f"country={country}",
                f"device_id={device_id}",
            )
            os.makedirs(output_path, exist_ok=True)

            # Split into monthly files like the real data
            df["_month"] = df["filename"].str[:7]
            for month, month_df in df.groupby("_month"):
                month_df = month_df.drop(columns=["_month"])
                out_file = os.path.join(output_path, f"{month}_{device_id}.parquet")
                month_df.to_parquet(out_file, index=False, engine="pyarrow")

            device_rows = len(df)
            total_rows += device_rows
            elapsed = time.time() - start
            print(f" {device_rows:,} rows [{elapsed:.1f}s]")

    total_elapsed = time.time() - total_start
    print(f"\nDone. Generated {total_rows:,} rows in {total_elapsed:.1f}s")
    print(f"Output directory: {args.output_dir}/")


if __name__ == "__main__":
    main()
