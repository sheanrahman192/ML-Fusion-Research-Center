#!/usr/bin/env python3
"""
Plot selected plasma signals vs time for one shot from LABEL_PROPAGATED_DATABASE.csv,
plus state_binary (training-style binary label).

Default shot 206894; signals: iln3iamp, betan, density, fs_sum, Ip.
Optional: dusbradial — not present in LABEL_PROPAGATED_DATABASE.csv; supply a merge CSV
with columns (shot, time, <radial column>) via --dusbradial-csv, or pass --dusbradial-column
to plot an existing column (e.g. dR_sep) instead.

Usage:
  python plot_shot_signals_multichannel.py
  python plot_shot_signals_multichannel.py --shot 206894 --out shot_206894.png
  python plot_shot_signals_multichannel.py --shots 206894 206332 --out-dir ./figures
  python plot_shot_signals_multichannel.py --dusbradial-column dR_sep
  python plot_shot_signals_multichannel.py --dusbradial-csv /path/to/extra.csv --dusbradial-column dusbradial
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_CSV = Path("/mnt/homes/sr4240/my_folder/LABEL_PROPAGATED_DATABASE.csv")
DEFAULT_SHOT = 206894
CHUNK_SIZE = 400_000


def _format_time_ms(x: float) -> str:
    """Format a time value in ms for figure text; preserve decimals when needed."""
    if not np.isfinite(x):
        return "nan"
    xr = round(x)
    if abs(x - xr) < 1e-6:
        return str(int(xr))
    return f"{x:.10g}"


BASE_COLS = (
    "shot",
    "time",
    "iln3iamp",
    "betan",
    "density",
    "fs_sum",
    "Ip",
    "state_binary",
)


def load_shot_rows(csv_path: Path, shot: int, extra_cols: list[str]) -> pd.DataFrame:
    cols = list(dict.fromkeys(list(BASE_COLS) + extra_cols))
    missing = []
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    for c in cols:
        if c not in header:
            missing.append(c)
    if missing:
        raise ValueError(
            f"CSV missing columns {missing}. Available include: {header[:30]}..."
        )

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        csv_path, usecols=cols, chunksize=CHUNK_SIZE, low_memory=False
    ):
        sub = chunk[chunk["shot"] == shot]
        if len(sub):
            chunks.append(sub)
    if not chunks:
        raise ValueError(f"No rows found for shot {shot} in {csv_path}")
    df = pd.concat(chunks, ignore_index=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.sort_values("time").reset_index(drop=True)


def load_many_shots(
    csv_path: Path, shots: list[int], extra_cols: list[str]
) -> dict[int, pd.DataFrame]:
    """Single pass through CSV; returns one dataframe per shot (sorted by time)."""
    cols = list(dict.fromkeys(list(BASE_COLS) + extra_cols))
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = [c for c in cols if c not in header]
    if missing:
        raise ValueError(
            f"CSV missing columns {missing}. Available include: {header[:30]}..."
        )

    shot_set = set(shots)
    buckets: dict[int, list[pd.DataFrame]] = {s: [] for s in shots}

    for chunk in pd.read_csv(
        csv_path, usecols=cols, chunksize=CHUNK_SIZE, low_memory=False
    ):
        sub = chunk[chunk["shot"].isin(shot_set)]
        if sub.empty:
            continue
        for sid, grp in sub.groupby("shot", sort=False):
            sid = int(sid)
            if sid in buckets:
                buckets[sid].append(grp)

    out: dict[int, pd.DataFrame] = {}
    for s in shots:
        parts = buckets[s]
        if not parts:
            raise ValueError(f"No rows found for shot {s} in {csv_path}")
        df = pd.concat(parts, ignore_index=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        out[s] = df.sort_values("time").reset_index(drop=True)
    return out


def merge_dusbradial(
    df: pd.DataFrame,
    merge_path: Path,
    shot: int,
    col_name: str,
) -> pd.DataFrame:
    mcols = ["shot", "time", col_name]
    m = pd.read_csv(merge_path, usecols=lambda c: c in set(mcols))
    for c in mcols:
        if c not in m.columns:
            raise ValueError(f"{merge_path} must contain columns {mcols}; missing {c}")
    m = m[m["shot"] == shot].sort_values("time")
    m = m.drop_duplicates(subset=["time"], keep="last")
    out = df.merge(m[["time", col_name]], on="time", how="left")
    return out


def plot_shot_figure(
    df: pd.DataFrame,
    shot: int,
    out_path: Path,
    *,
    dusbradial_column: str,
    dusbradial_csv: Path | None,
) -> None:
    radial_series: pd.Series | None = None
    radial_label = dusbradial_column

    if dusbradial_csv is not None:
        df = merge_dusbradial(df, dusbradial_csv, shot, dusbradial_column)
        radial_series = df[dusbradial_column]
    elif dusbradial_column in df.columns:
        radial_series = df[dusbradial_column]
    else:
        radial_series = None

    t = df["time"].to_numpy()
    t_start = float(np.min(t))
    t_end = float(np.max(t))

    signal_specs: list[tuple[str, np.ndarray, str]] = [
        ("iln3iamp", df["iln3iamp"].to_numpy(), "iln3iamp"),
        ("betan", df["betan"].to_numpy(), r"$\beta_N$"),
        ("density", df["density"].to_numpy(), "density"),
        ("fs_sum", df["fs_sum"].to_numpy(), "fs_sum"),
        ("Ip", df["Ip"].to_numpy(), r"$I_\mathrm{p}$"),
    ]

    if radial_series is not None:
        signal_specs.append((dusbradial_column, radial_series.to_numpy(), radial_label))
    else:
        signal_specs.append(
            (
                "__missing_radial__",
                np.full(len(t), np.nan),
                f"{radial_label} (not in CSV — use --dusbradial-csv or --dusbradial-column)",
            )
        )

    y_bin = df["state_binary"].to_numpy()

    nrows = len(signal_specs) + 1
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 2.4 * nrows), sharex=True)

    range_line = (
        f"Time interval (same units as x-axis): "
        f"start = {_format_time_ms(t_start)} ms  |  "
        f"end = {_format_time_ms(t_end)} ms  |  "
        f"span = {_format_time_ms(t_end - t_start)} ms"
    )
    fig.text(0.5, 0.93, range_line, ha="center", fontsize=11, transform=fig.transFigure)
    fig.suptitle(
        f"Shot {shot} — LABEL_PROPAGATED_DATABASE",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for ax, (key, y, ylab) in zip(axes[:-1], signal_specs):
        if key == "__missing_radial__":
            ax.text(
                0.5,
                0.5,
                "No dusbradial column in database.\n"
                "Pass --dusbradial-column <existing_col> or --dusbradial-csv <path>",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
            )
            ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.3)
        else:
            ax.plot(t, y, lw=0.8, color="C0")
            ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.3)
        ax.axvline(t_start, color="0.35", ls="--", lw=1.1, alpha=0.85, zorder=4)
        ax.axvline(t_end, color="0.35", ls="--", lw=1.1, alpha=0.85, zorder=4)

    axb = axes[-1]
    axb.step(t, y_bin, where="post", color="C3", lw=1.0)
    axb.fill_between(t, y_bin, step="post", alpha=0.25, color="C3")
    axb.set_ylabel("state_binary")
    axb.set_xlabel(
        f"time (ms)   —   [{_format_time_ms(t_start)}, {_format_time_ms(t_end)}] ms"
    )
    axb.set_yticks([0, 1])
    axb.grid(True, alpha=0.3)
    axb.axvline(t_start, color="0.35", ls="--", lw=1.1, alpha=0.85, zorder=4)
    axb.axvline(t_end, color="0.35", ls="--", lw=1.1, alpha=0.85, zorder=4)

    for ax in axes:
        ax.set_xlim(t_start, t_end)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Label propagated CSV")
    p.add_argument(
        "--shot",
        type=int,
        default=None,
        help="Single shot number (default 206894 if neither --shot nor --shots given)",
    )
    p.add_argument(
        "--shots",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help="Multiple shots; scans CSV once. Overrides --shot when non-empty.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (single-shot only; default shot_<n>_signals.png)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for PNGs when using multiple shots (default: script directory)",
    )
    p.add_argument(
        "--dusbradial-column",
        type=str,
        default="dusbradial",
        help="Column name for radial signal (must exist in main CSV, or in merge file)",
    )
    p.add_argument(
        "--dusbradial-csv",
        type=Path,
        default=None,
        help="Optional CSV with shot,time,<dusbradial-column> to left-merge on time",
    )
    args = p.parse_args()

    if args.shots:
        shots = list(args.shots)
    elif args.shot is not None:
        shots = [args.shot]
    else:
        shots = [DEFAULT_SHOT]

    base_dir = Path(__file__).resolve().parent
    if len(shots) > 1:
        if args.out is not None:
            raise SystemExit(
                "Cannot use --out with multiple shots; use --out-dir instead "
                "(files will be shot_<n>_signals.png)."
            )
        out_dir = args.out_dir or base_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_paths = {s: out_dir / f"shot_{s}_signals.png" for s in shots}
    else:
        if args.out is not None:
            out_paths = {shots[0]: args.out}
        else:
            out_paths = {shots[0]: base_dir / f"shot_{shots[0]}_signals.png"}

    hdr = pd.read_csv(args.csv, nrows=0).columns.tolist()
    extra: list[str] = []
    if args.dusbradial_csv is None and args.dusbradial_column in hdr:
        extra.append(args.dusbradial_column)

    if len(shots) > 1:
        dfs = load_many_shots(args.csv, shots, extra)
        for s in shots:
            plot_shot_figure(
                dfs[s],
                s,
                out_paths[s],
                dusbradial_column=args.dusbradial_column,
                dusbradial_csv=args.dusbradial_csv,
            )
            print(f"Wrote {out_paths[s]}")
    else:
        df = load_shot_rows(args.csv, shots[0], extra)
        plot_shot_figure(
            df,
            shots[0],
            out_paths[shots[0]],
            dusbradial_column=args.dusbradial_column,
            dusbradial_csv=args.dusbradial_csv,
        )
        print(f"Wrote {out_paths[shots[0]]}")


if __name__ == "__main__":
    main()
