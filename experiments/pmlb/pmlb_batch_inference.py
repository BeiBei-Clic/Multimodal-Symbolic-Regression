import csv
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.pmlb.pmlb_inference import (
    DIRECT_REFINEMENT_TYPE,
    build_inference_parser,
    configure_params,
    create_inference_components,
    infer_dataset_result,
)

CSV_COLUMNS = [
    "dataset",
    "status",
    "n_features",
    "refinement_type",
    "r2",
    "rmse",
    "complexity",
    "seconds",
    "error",
    "expr",
]


def build_batch_parser():
    parser = build_inference_parser()
    parser.set_defaults(
        dataset="",
        dataset_root="./pmlb/datasets",
        output_csv="./experiments/pmlb/results/pmlb_batch_inference.csv",
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


def is_regression_dataset(dataset_dir):
    metadata_path = dataset_dir / "metadata.yaml"
    if not metadata_path.is_file():
        return False

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle) or {}
    return metadata.get("task") == "regression"


def iter_regression_datasets(datasets_dir, dataset_limit):
    dataset_names = []
    for dataset_dir in sorted(datasets_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if is_regression_dataset(dataset_dir):
            dataset_names.append(dataset_dir.name)

    if dataset_limit and dataset_limit > 0:
        dataset_names = dataset_names[:dataset_limit]
    return dataset_names


def write_header(csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        handle.flush()


def append_result(csv_path, result):
    row = {column: result.get(column, "") for column in CSV_COLUMNS}
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
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

    csv_path = Path(params.output_csv)
    write_header(csv_path)

    env, model = create_inference_components(params)
    for dataset_name in dataset_names:
        result = infer_dataset_result(datasets_dir, dataset_name, env, params, model)
        result["refinement_type"] = DIRECT_REFINEMENT_TYPE
        append_result(csv_path, result)
        print(
            f"[{result['status']}] {dataset_name} r2={result['r2']} rmse={result['rmse']} seconds={result['seconds']:.2f}"
        )

    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
