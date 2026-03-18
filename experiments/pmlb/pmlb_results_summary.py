import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = ("Feynman", "Strogatz", "Black-box")
OUTPUT_COLUMNS = [
    "group",
    "r2_mean",
    "r2_var",
    "r2_valid_count",
    "total_count",
    "recovery_rate",
    "complexity_mean",
    "complexity_var",
    "complexity_count",
    "seconds_mean",
    "seconds_var",
    "seconds_count",
]
SUCCESS_STATUSES = {"ok", "success"}


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize batch PMLB inference results.")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="experiments/pmlb/results/pmlb_batch_inference.csv",
        help="Batch inference CSV to summarize.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="experiments/pmlb/results/pmlb_batch_inference_summary.csv",
        help="Where to save the grouped summary CSV.",
    )
    return parser


def assign_group(dataset_name):
    dataset_name = str(dataset_name)
    if dataset_name.startswith("feynman_"):
        return "Feynman"
    if dataset_name.startswith("strogatz_"):
        return "Strogatz"
    return "Black-box"


def finite_numeric(series):
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric[np.isfinite(numeric)]


def safe_mean(values):
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def safe_var(values):
    if len(values) == 0:
        return 0.0
    return float(np.var(values, ddof=0))


def summarize_group(df_group):
    total_count = int(len(df_group))

    raw_r2 = pd.to_numeric(df_group["r2"], errors="coerce")
    valid_r2_mask = np.isfinite(raw_r2) & (raw_r2 >= 0)
    clipped_r2 = raw_r2.where(np.isfinite(raw_r2), 0.0)
    clipped_r2 = clipped_r2.clip(lower=0).fillna(0.0)

    status_series = df_group["status"].astype(str).str.lower()
    success_mask = status_series.isin(SUCCESS_STATUSES)

    complexity_values = finite_numeric(df_group.loc[success_mask, "complexity"])
    seconds_values = finite_numeric(df_group.loc[success_mask, "seconds"])

    return {
        "r2_mean": safe_mean(clipped_r2.to_numpy()),
        "r2_var": safe_var(clipped_r2.to_numpy()),
        "r2_valid_count": int(valid_r2_mask.sum()),
        "total_count": total_count,
        "recovery_rate": float(((raw_r2 > 0.9) & np.isfinite(raw_r2)).sum() / total_count) if total_count else 0.0,
        "complexity_mean": safe_mean(complexity_values.to_numpy()),
        "complexity_var": safe_var(complexity_values.to_numpy()),
        "complexity_count": int(len(complexity_values)),
        "seconds_mean": safe_mean(seconds_values.to_numpy()),
        "seconds_var": safe_var(seconds_values.to_numpy()),
        "seconds_count": int(len(seconds_values)),
    }


def summarize_results(df):
    df = df.copy()
    df["group"] = df["dataset"].map(assign_group)

    rows = []
    for group in GROUPS:
        df_group = df[df["group"] == group]
        summary = summarize_group(df_group)
        rows.append({"group": group, **summary})

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def validate_columns(df):
    required = {"dataset", "status", "r2", "complexity", "seconds"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def main():
    args = build_parser().parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    df = pd.read_csv(input_path)
    validate_columns(df)
    summary_df = summarize_results(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    print(summary_df.to_string(index=False))
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
