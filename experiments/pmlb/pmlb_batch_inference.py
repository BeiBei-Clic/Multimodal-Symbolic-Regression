import csv
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from LSO_eval import load_pmlb_summary_stats
from experiments.pmlb.pmlb_inference import (
    RESULT_COLUMNS,
    build_inference_parser,
    configure_params,
    create_inference_components,
    infer_dataset_result,
)


def build_batch_parser():
    parser = build_inference_parser()
    parser.set_defaults(
        dataset="",
        dataset_root="./pmlb/datasets",
        output_csv="",
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default="./pmlb/datasets",
        help="Directory containing local PMLB dataset folders.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./weights/snip-e2e-sr.pth",
        help="Path to the pretrained checkpoint.",
    )
    parser.add_argument(
        "--dataset_limit",
        type=int,
        default=0,
        help="Optional limit on the number of datasets to process. 0 means all.",
    )
    return parser


def iter_regression_datasets(datasets_dir, dataset_limit):
    summary_stats = load_pmlb_summary_stats(datasets_dir)
    regression_datasets = summary_stats[summary_stats["task"] == "regression"]
    regression_datasets = regression_datasets[regression_datasets["n_categorical_features"] == 0]
    regression_datasets = regression_datasets[regression_datasets["n_features"] < 11]
    dataset_names = regression_datasets["dataset"].tolist()

    if dataset_limit and dataset_limit > 0:
        dataset_names = dataset_names[:dataset_limit]
    return dataset_names


def write_header(csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        handle.flush()


def append_result(csv_path, result):
    row = {column: result.get(column, "") for column in RESULT_COLUMNS}
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writerow(row)
        handle.flush()


def main():
    parser = build_batch_parser()
    params = configure_params(parser.parse_args())
    params.reload_model = params.model_path

    datasets_dir = Path(params.datasets_dir)
    if not datasets_dir.is_dir():
        raise FileNotFoundError(f"PMLB datasets directory not found: {datasets_dir}")

    dataset_names = iter_regression_datasets(datasets_dir, params.dataset_limit)
    if not dataset_names:
        raise RuntimeError(f"No regression datasets found under {datasets_dir}")

    if not params.output_csv:
        noise_tag = format(params.noise_strength, "g")
        params.output_csv = f"./experiments/pmlb/results/pmlb_batch_inference_noise_{noise_tag}.csv"

    csv_path = Path(params.output_csv)
    completed_datasets = set()
    if csv_path.is_file() and csv_path.stat().st_size > 0:
        existing_results = pd.read_csv(csv_path)
        if "dataset" not in existing_results.columns:
            raise ValueError(f"Existing output CSV is missing 'dataset' column: {csv_path}")
        completed_datasets = set(existing_results["dataset"].dropna().astype(str))
    else:
        write_header(csv_path)

    pending_datasets = [name for name in dataset_names if name not in completed_datasets]
    if not pending_datasets:
        print(f"All datasets already processed in {csv_path}")
        return

    if completed_datasets:
        print(f"Skipping {len(dataset_names) - len(pending_datasets)} completed datasets from {csv_path}")

    env, model = create_inference_components(params)
    for dataset_name in pending_datasets:
        result = infer_dataset_result(datasets_dir, dataset_name, env, params, model)
        append_result(csv_path, result)
        print(
            f"[{result['status']}] {dataset_name} r2={result['r2']} rmse={result['rmse']} seconds={result['seconds']:.2f}"
        )

    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
