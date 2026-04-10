import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = ("Feynman", "Strogatz", "Black-box")
DEFAULT_RESULTS_DIR = Path("experiments/pmlb/results")
DEFAULT_NOISE_LEVELS = (0.0, 0.001, 0.01, 0.1)
OUTPUT_COLUMNS = [
    "noise_strength",
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
        "--input_csvs",
        nargs="*",
        default=None,
        help="Batch inference CSVs to summarize together. If omitted, use the default four noise files.",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="",
        help="Deprecated single-input alias kept for backward compatibility.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="experiments/pmlb/results/pmlb_results_summary.csv",
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


def summarize_single_file(df, noise_strength):
    df = df.copy()
    df["group"] = df["dataset"].map(assign_group)

    rows = []
    for group in GROUPS:
        df_group = df[df["group"] == group]
        summary = summarize_group(df_group)
        rows.append({"noise_strength": noise_strength, "group": group, **summary})
    return rows


def validate_columns(df):
    required = {"dataset", "status", "r2", "complexity", "seconds"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def infer_noise_strength(input_path):
    file_name = input_path.name
    if file_name == "pmlb_results.csv":
        return 0.0

    match = re.fullmatch(r"pmlb_batch_inference_noise_([0-9eE.+-]+)\.csv", file_name)
    if match is None:
        raise ValueError(f"Cannot infer noise_strength from filename: {input_path}")
    return float(match.group(1))


def resolve_default_input_paths():
    zero_noise_path = DEFAULT_RESULTS_DIR / "pmlb_batch_inference_noise_0.csv"
    legacy_zero_noise_path = DEFAULT_RESULTS_DIR / "pmlb_results.csv"
    input_paths = []

    if zero_noise_path.is_file():
        input_paths.append(zero_noise_path)
    elif legacy_zero_noise_path.is_file():
        input_paths.append(legacy_zero_noise_path)
    else:
        raise FileNotFoundError(
            f"Missing zero-noise results: {zero_noise_path} or {legacy_zero_noise_path}"
        )

    for noise_strength in DEFAULT_NOISE_LEVELS[1:]:
        input_path = DEFAULT_RESULTS_DIR / f"pmlb_batch_inference_noise_{format(noise_strength, 'g')}.csv"
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing default results CSV: {input_path}")
        input_paths.append(input_path)

    return input_paths


def resolve_input_paths(args):
    if args.input_csv and args.input_csvs:
        raise ValueError("Use either --input_csvs or --input_csv, not both.")
    if args.input_csv:
        return [Path(args.input_csv)]
    if args.input_csvs:
        return [Path(path) for path in args.input_csvs]
    return resolve_default_input_paths()


def summarize_inputs(input_paths):
    rows = []
    seen_noise_strengths = set()

    for input_path in input_paths:
        noise_strength = infer_noise_strength(input_path)
        if noise_strength in seen_noise_strengths:
            raise ValueError(f"Duplicate noise_strength detected: {noise_strength}")
        seen_noise_strengths.add(noise_strength)

        df = pd.read_csv(input_path)
        validate_columns(df)
        rows.extend(summarize_single_file(df, noise_strength))

    summary_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    summary_df["group"] = pd.Categorical(summary_df["group"], categories=GROUPS, ordered=True)
    summary_df = summary_df.sort_values(["noise_strength", "group"]).reset_index(drop=True)
    summary_df["group"] = summary_df["group"].astype(str)
    return summary_df.loc[:, OUTPUT_COLUMNS]


def main():
    args = build_parser().parse_args()
    input_paths = resolve_input_paths(args)
    output_path = Path(args.output_csv)

    summary_df = summarize_inputs(input_paths)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
