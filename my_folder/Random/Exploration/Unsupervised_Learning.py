import argparse
import logging
import time
from typing import Dict, List, Optional, Tuple

import dask.array as da
import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, progress
from tqdm import tqdm

try:
    # Try to use RAPIDS cuML for GPU acceleration first
    import cupy as cp
    from cuml.cluster import KMeans as cuKMeans
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    from cuml.impute import SimpleImputer as cuSimpleImputer
    GPU_AVAILABLE = True
    print("🚀 GPU acceleration available with RAPIDS cuML")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ GPU acceleration not available, falling back to CPU")

try:
    # Fallback to Dask-ML for CPU processing
    from dask_ml.preprocessing import StandardScaler
    from dask_ml.impute import SimpleImputer
    from dask_ml.cluster import KMeans
except Exception as import_error:  # pragma: no cover
    raise SystemExit(
        "This script requires either dask-ml or cuml. Install with: pip install dask-ml cuml"
    ) from import_error


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=level,
    )


def show_progress(future, description: str):
    """Show progress for a Dask computation with a progress bar."""
    print(f"\n{description}...")
    progress(future)
    print(f"✓ {description} completed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unsupervised clustering into two states (0 vs 1) without using the"
            " 'state' column for training. Designed for large datasets using Dask."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input dataset path. Supports CSV (wildcards OK) or Parquet."
            " Examples: '/path/data-*.csv', '/path/data.parquet'"
        ),
    )
    parser.add_argument(
        "--file-format",
        choices=["csv", "parquet"],
        default="csv",
        help="Input file format (default: csv)",
    )
    parser.add_argument(
        "--blocksize",
        default="256MB",
        help="Blocksize/row-group hint for CSV ingestion (default: 256MB)",
    )
    parser.add_argument(
        "--id-columns",
        nargs="*",
        default=None,
        help=(
            "Optional list of identifier columns to exclude from training in addition"
            " to 'state' (e.g., timestamps, IDs)."
        ),
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help=(
            "Persist intermediate Dask collections in memory to speed up repeated"
            " access (requires sufficient RAM)."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="If provided, start a local Dask distributed Client with this many workers.",
    )
    parser.add_argument(
        "--report-out",
        default=None,
        help=(
            "Optional path to write a small metrics report (CSV). If omitted, only"
            " prints to stdout."
        ),
    )
    parser.add_argument(
        "--predictions-out",
        default=None,
        help=(
            "Optional path to write per-row predicted labels (Parquet). Contains"
            " columns: ['predicted_state', 'cluster_label']."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="For testing: use only this many rows (e.g., 100000 for 100k rows)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can repeat: -v, -vv)",
    )
    return parser.parse_args()


def load_dataset(
    path: str,
    file_format: str,
    blocksize: str,
) -> dd.DataFrame:
    if file_format == "csv":
        df = dd.read_csv(
            path,
            blocksize=blocksize,
            assume_missing=True,
            na_values=["", "na", "NA", "null", "None"],
        )
    else:
        df = dd.read_parquet(path)
    logging.info("Loaded dataset with columns: %s", list(df.columns))
    return df


def build_feature_target_frames(
    df: dd.DataFrame,
    id_columns: Optional[List[str]] = None,
) -> Tuple[dd.DataFrame, dd.Series]:
    if "state" not in df.columns:
        raise ValueError("Expected a 'state' column in the dataset.")

    # Map raw state to binary label: 1/2/3 -> 0, 4 -> 1
    state_mapping = {1: 0, 2: 0, 3: 0, 4: 1}
    binary_state = df["state"].map(state_mapping)

    # Exclude columns not to be used for training
    columns_to_exclude: List[str] = ["state", "binary_state"]
    if id_columns:
        columns_to_exclude.extend(id_columns)

    # Auto-select numeric feature columns
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.columns.size == 0:
        raise ValueError("No numeric columns found for clustering.")

    # Attach binary_state for filtering and later evaluation
    numeric_df["binary_state"] = binary_state

    # Drop rows where binary_state is missing
    numeric_df = numeric_df.dropna(subset=["binary_state"])

    # Remove excluded columns safely if present
    feature_columns = [
        c for c in numeric_df.columns if c not in set(columns_to_exclude)
    ]
    if "binary_state" in feature_columns:
        feature_columns.remove("binary_state")

    X = numeric_df[feature_columns]
    y = numeric_df["binary_state"]

    if X.columns.size == 0:
        raise ValueError(
            "No remaining feature columns after exclusions; cannot train."
        )

    logging.info("Using %d feature columns for clustering.", X.columns.size)
    return X, y


def preprocess_features(X: dd.DataFrame) -> da.Array:
    # Convert to float64 to avoid pandas extension dtype issues
    X_float = X.astype(float)
    
    # Impute missing values with median, then standardize features
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler(with_mean=False)  # memory-friendly for large data

    X_imputed = imputer.fit_transform(X_float)
    X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled


def cluster_two_groups(
    X_scaled: da.Array,
    random_state: int,
) -> KMeans:
    # 2-cluster KMeans; Dask-ML implementation scales well to large datasets
    model = KMeans(
        n_clusters=2,
        random_state=random_state,
        init_max_iter=5,
        oversampling_factor=2,
    )
    model.fit(X_scaled)
    return model


def map_clusters_to_binary(
    cluster_labels: dd.Series,
    true_binary: dd.Series,
) -> Dict[int, int]:
    # Compute mean positive rate per cluster using Dask aggregation to avoid memory issues
    # Create a temporary DataFrame for grouping
    df_tmp = cluster_labels.to_frame("cluster").assign(y=true_binary)
    
    # Use Dask groupby to compute means efficiently
    grouped = df_tmp.groupby("cluster")["y"].mean().compute()

    mapping: Dict[int, int] = {}
    for cluster_label, mean_value in grouped.items():
        mapping[int(cluster_label)] = 1 if float(mean_value) >= 0.5 else 0

    logging.info("Cluster->binary mapping based on positive rate: %s", mapping)
    return mapping


def compute_confusion_counts(
    y_true: dd.Series,
    y_pred: dd.Series,
) -> Tuple[int, int, int, int]:
    # Use Dask aggregation to compute confusion matrix counts efficiently
    df_eval = y_true.to_frame("y_true").assign(y_pred=y_pred)
    counts = df_eval.groupby(["y_true", "y_pred"]).size().compute()

    # Extract TP, TN, FP, FN for binary {0,1}
    def get_count(y_t: int, y_p: int) -> int:
        try:
            return int(counts.loc[(y_t, y_p)])
        except Exception:
            return 0

    tn = get_count(0, 0)
    fp = get_count(0, 1)
    fn = get_count(1, 0)
    tp = get_count(1, 1)
    return tp, tn, fp, fn


def compute_binary_metrics(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def write_report(path: str, metrics: Dict[str, float]) -> None:
    import pandas as pd

    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)
    logging.info("Wrote metrics report to %s", path)


def write_predictions(path: str, cluster_labels: dd.Series, predicted: dd.Series) -> None:
    df = predicted.to_frame("predicted_state").assign(cluster_label=cluster_labels)
    df.to_parquet(path, write_index=False)
    logging.info("Wrote predictions to %s (Parquet)", path)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    print("🚀 Starting Unsupervised Learning Analysis")
    print("=" * 50)

    client: Optional[Client] = None
    if args.n_workers:
        print(f"🔧 Starting Dask Client with {args.n_workers} workers...")
        client = Client(n_workers=args.n_workers, memory_limit="8GB")
        logging.info("Started local Dask Client: %s", client)

    print("📂 Loading dataset...")
    start_time = time.time()
    df = load_dataset(args.input, args.file_format, args.blocksize)
    
    if args.sample_size:
        print(f"🔬 Using sample of {args.sample_size:,} rows for testing...")
        df = df.head(args.sample_size, npartitions=-1)
    
    print(f"✓ Dataset loaded in {time.time() - start_time:.2f}s")

    print("🔧 Building feature and target frames...")
    start_time = time.time()
    X, y_true = build_feature_target_frames(df, id_columns=args.id_columns)
    print(f"✓ Feature frames built in {time.time() - start_time:.2f}s")

    if args.persist:
        print("💾 Persisting data in memory...")
        start_time = time.time()
        X_future = X.persist()
        y_true_future = y_true.persist()
        show_progress([X_future, y_true_future], "Persisting data")
        X = X_future
        y_true = y_true_future

    print("⚙️ Preprocessing features (imputation and scaling)...")
    start_time = time.time()
    X_scaled = preprocess_features(X)
    print(f"✓ Features preprocessed in {time.time() - start_time:.2f}s")

    if args.persist:
        print("💾 Persisting scaled features...")
        start_time = time.time()
        X_scaled_future = X_scaled.persist()
        show_progress([X_scaled_future], "Persisting scaled features")
        X_scaled = X_scaled_future

    print("🎯 Running K-means clustering...")
    start_time = time.time()
    model = cluster_two_groups(X_scaled, random_state=args.random_state)
    print(f"✓ Clustering completed in {time.time() - start_time:.2f}s")

    # Predict cluster labels lazily
    print("🔮 Predicting cluster labels...")
    start_time = time.time()
    cluster_labels_da = model.predict(X_scaled)
    cluster_labels = dd.from_dask_array(cluster_labels_da, columns=["cluster"])  # type: ignore[arg-type]
    cluster_labels = cluster_labels["cluster"]
    print(f"✓ Predictions generated in {time.time() - start_time:.2f}s")

    # Map clusters -> binary predictions using majority mapping
    print("🔄 Mapping clusters to binary states...")
    start_time = time.time()
    mapping = map_clusters_to_binary(cluster_labels, y_true)
    predicted_binary = cluster_labels.map(mapping)
    print(f"✓ Mapping completed in {time.time() - start_time:.2f}s")

    # Evaluate
    print("📊 Computing evaluation metrics...")
    start_time = time.time()
    tp, tn, fp, fn = compute_confusion_counts(y_true, predicted_binary)
    metrics = compute_binary_metrics(tp, tn, fp, fn)
    print(f"✓ Metrics computed in {time.time() - start_time:.2f}s")

    # Emphasize F1 in output
    print("\n" + "=" * 50)
    print("📈 RESULTS")
    print("=" * 50)
    print("Confusion Matrix:")
    print(f"  True Negatives: {tn:,}")
    print(f"  False Positives: {fp:,}")
    print(f"  False Negatives: {fn:,}")
    print(f"  True Positives: {tp:,}")
    print()
    print("Performance Metrics:")
    print(f"  F1 Score: {metrics['f1']:.6f}")
    print(f"  Precision: {metrics['precision']:.6f}")
    print(f"  Recall: {metrics['recall']:.6f}")
    print(f"  Accuracy: {metrics['accuracy']:.6f}")
    print(f"  Total Samples: {metrics['support']:,}")
    print("=" * 50)

    if args.report_out:
        write_report(args.report_out, metrics)
        print(f"📄 Report saved to: {args.report_out}")

    if args.predictions_out:
        write_predictions(args.predictions_out, cluster_labels, predicted_binary)
        print(f"💾 Predictions saved to: {args.predictions_out}")

    if client is not None:
        client.close()
        print("🔧 Dask Client closed")

    print("✅ Analysis completed successfully!")


if __name__ == "__main__":
    main()


