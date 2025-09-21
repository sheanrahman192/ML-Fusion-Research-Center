import pickle
import gzip
import time
import math
import pandas as pd

"""Unpickle compressed data, write CSV with progress, and report stats."""

# Unpickle the compressed data once
with gzip.open('plasma_data.pkl.gz', 'rb') as f:
    data = pickle.load(f)

df = pd.DataFrame(data)

# Stream to CSV in chunks with a simple progress indicator
output_csv = 'plasma_data.csv'
chunk_size = 200_000  # rows per chunk (tune as needed)

total_rows = len(df)
num_chunks = math.ceil(total_rows / chunk_size) if total_rows else 0
start_time = time.time()
print(f"Writing {total_rows:,} rows to {output_csv} in {num_chunks} chunks...")

if total_rows:
    with open(output_csv, 'w') as out:
        for chunk_index, start in enumerate(range(0, total_rows, chunk_size)):
            end = min(start + chunk_size, total_rows)
            header_flag = (chunk_index == 0)
            df.iloc[start:end].to_csv(out, index=False, header=header_flag)
            rows_done = end
            elapsed = time.time() - start_time
            percent = (rows_done / total_rows) * 100
            print(
                f"[{chunk_index + 1}/{num_chunks}] {rows_done:,}/{total_rows:,} rows ({percent:.1f}%) - {elapsed:.1f}s",
                flush=True,
            )

total_elapsed = time.time() - start_time
print(f"Done writing {output_csv} in {total_elapsed:.1f}s")

# Count unique shots and distribution
print(f"Total records: {len(df)}")
if 'shot' in df.columns:
    try:
        min_shot = df['shot'].min()
        max_shot = df['shot'].max()
        print(f"Shot range: {min_shot} to {max_shot}")
    except Exception:
        pass
    print(f"Total unique shots: {df['shot'].nunique()}")
if 'state' in df.columns:
    print(f"State distribution: {df['state'].value_counts().to_dict()}")