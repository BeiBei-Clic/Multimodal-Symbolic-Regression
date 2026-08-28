import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT_DIR = Path("experiments/pmlb/SNIP_results")
DEFAULT_NOISE_LEVELS = (0.0, 0.001, 0.01, 0.1)
DEFAULT_OUTPUT_CSV = DEFAULT_INPUT_DIR / "strogatz_results.csv"
R2_THRESHOLD = 0.99


def build_parser():
    parser = argparse.ArgumentParser(description="Collect Strogatz rows from batch PMLB inference CSVs.")
    parser.add_argument(
        "--input_csvs",
        nargs="*",
        default=None,
        help="Batch inference CSVs to collect. If omitted, use the default four noise files in SNIP_results.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help="Where to save the merged Strogatz CSV.",
    )
    return parser


def resolve_input_paths(args):
    if args.input_csvs:
        return [Path(path) for path in args.input_csvs]
    input_paths = []
    for noise_strength in DEFAULT_NOISE_LEVELS:
        input_path = DEFAULT_INPUT_DIR / f"pmlb_batch_inference_noise_{format(noise_strength, 'g')}.csv"
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing default results CSV: {input_path}")
        input_paths.append(input_path)
    return input_paths


def main():
    args = build_parser().parse_args()
    input_paths = resolve_input_paths(args)
    output_path = Path(args.output_csv)

    frames = []
    for input_path in input_paths:
        df = pd.read_csv(input_path)
        missing = sorted({"dataset", "r2", "noise_strength"} - set(df.columns))
        if missing:
            raise ValueError(f"{input_path} is missing required columns: {', '.join(missing)}")
        frames.append(df[df["dataset"].str.startswith("strogatz_")])

    strogatz_df = pd.concat(frames, ignore_index=True)
    strogatz_df = strogatz_df.sort_values(["noise_strength", "dataset"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    strogatz_df.to_csv(output_path, index=False)

    print(f"Strogatz rows merged: {len(strogatz_df)} from {len(input_paths)} files")
    for noise_strength, group in strogatz_df.groupby("noise_strength"):
        count = int((group["r2"] > R2_THRESHOLD).sum())
        total = len(group)
        print(
            f"noise={noise_strength:g}: r2>{R2_THRESHOLD} = {count}/{total} = {count / total:.4f}"
        )
    all_count = int((strogatz_df["r2"] > R2_THRESHOLD).sum())
    print(
        f"overall: r2>{R2_THRESHOLD} = {all_count}/{len(strogatz_df)} = {all_count / len(strogatz_df):.4f}"
    )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
