import pandas as pd
import sys


def load_dataset(csv_path: str) -> pd.DataFrame:
    print("=== Missing Values Report ===")
    print(f"Loading CSV dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def print_missing_by_column(df: pd.DataFrame) -> None:
    print("\nMissing values per column:")
    missing_series = df.isna().sum().sort_values(ascending=False)
    print(missing_series)


def report_missing_kappa_shots(df: pd.DataFrame) -> None:
    if "kappa" not in df.columns:
        print("\n[kappa] column not found in dataset.")
        return
    if "shot" not in df.columns:
        print("\n[shot] column not found in dataset; cannot summarize shots with missing kappa.")
        return

    kappa_missing = df[df["kappa"].isna()]  # rows where kappa is missing
    total_missing = len(kappa_missing)
    unique_shots = sorted(kappa_missing["shot"].dropna().unique().tolist())

    print("\nMissing 'kappa' overview:")
    print(f"Total rows with missing kappa: {total_missing}")
    print(f"Unique shots with missing kappa: {len(unique_shots)}")
    if unique_shots:
        print("Shot list (sorted):")
        print(unique_shots)

        print("\nCounts of rows with missing kappa by shot:")
        counts = kappa_missing.groupby("shot").size().sort_values(ascending=False)
        print(counts)
    else:
        print("No shots have missing kappa.")


def main():
    csv_path = "plasma_data.csv"
    if len(sys.argv) > 1 and sys.argv[1].strip():
        csv_path = sys.argv[1]

    df = load_dataset(csv_path)

    # Per the CNN script, the missing report there is printed before any filtering.
    print_missing_by_column(df)

    # Additionally, list shots with missing values for kappa.
    report_missing_kappa_shots(df)

    # If you also want to see the view after excluding a known problematic shot (as in CNN script):
    if "shot" in df.columns:
        df_excluded = df[df["shot"] != 191675].copy()
        if len(df_excluded) != len(df):
            print("\n(After excluding shot 191675, consistent with the CNN script):")
            print_missing_by_column(df_excluded)
            report_missing_kappa_shots(df_excluded)


if __name__ == "__main__":
    main()


