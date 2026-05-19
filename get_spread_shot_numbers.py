#!/usr/bin/env python3
"""
Get 20 shot numbers spread evenly across LABEL_PROPAGATED_DATABASE.csv.
Uses line-based sampling to handle the large file without loading it fully.
"""

import csv
import os
import subprocess

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LABEL_PROPAGATED_DATABASE.csv")
NUM_SHOTS = 20


def _count_lines(path):
    """Fast line count using wc -l."""
    out = subprocess.run(["wc", "-l", path], capture_output=True, text=True, check=True)
    return int(out.stdout.split()[0])


def get_spread_shot_numbers(csv_path=CSV_PATH, n=NUM_SHOTS):
    """Return n shot numbers sampled at evenly spaced rows across the CSV."""
    total_lines = _count_lines(csv_path)

    data_lines = total_lines - 1  # exclude header
    if data_lines < n:
        # If fewer data rows than requested, read all and take shot from each
        target_indices = list(range(1, data_lines + 1))
    else:
        # 0-based row indices (1 = first data row) spread evenly
        target_indices = [
            1 + int((data_lines - 1) * i / (n - 1))
            for i in range(n)
        ]

    shots = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        shot_col = header.index("shot") if "shot" in header else 0

        current_line = 1
        next_target = 0
        for row in reader:
            if next_target < len(target_indices) and current_line == target_indices[next_target]:
                if row:
                    shots.append(int(row[shot_col]))
                next_target += 1
            current_line += 1
            if next_target >= len(target_indices):
                break

    return shots


def main():
    if not os.path.isfile(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    shots = get_spread_shot_numbers()
    print(f"20 shot numbers spread across {CSV_PATH}:")
    print(shots)


if __name__ == "__main__":
    main()
