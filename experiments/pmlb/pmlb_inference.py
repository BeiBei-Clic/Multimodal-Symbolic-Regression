import time
from pathlib import Path
import sys
from collections import defaultdict
import copy
import random

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import symbolicregression
from LSO_eval import read_file, reload_model, resolve_pmlb_dataset_root
from LSO_fit import gen2eq
from model import SNIPSymbolicRegressor
from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
import symbolicregression.model.utils_wrapper as utils_wrapper
from symbolicregression.trainer import Trainer


def build_inference_parser():
    parser = get_parser()
    parser.set_defaults(
        beam_size=2,
        max_input_points=200,
        lso_optimizer="gwo",
        lso_pop_size=50,
        lso_max_iteration=80,
        lso_stop_r2=0.99,
        validation_metrics="r2_zero,r2,_rmse,_complexity",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="strogatz_barmag2",
        help="Name of the local PMLB dataset directory to run inference on.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="",
        help="Optional override for the local PMLB datasets root.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=200,
        help="Maximum number of rows to load from the dataset.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.25,
        help="Fraction of rows used for evaluation.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./experiments/pmlb/results/pmlb_inference.csv",
        help="Where to save the inference result CSV.",
    )
    return parser


def configure_params(params):
    params.batch_size = 1
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)

    params.n_steps_per_epoch = 100
    params.max_input_dimension = 10
    params.env_base_seed = 2023
    params.n_dec_layers = 16
    params.local_rank = -1
    params.master_port = -1
    params.num_workers = 1
    params.random_state = 14423
    params.max_number_bags = 10
    params.eval_verbose_print = True
    params.rescale = True
    params.eval_only = True
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(params.seed)

    if not params.cpu:
        assert torch.cuda.is_available()
    symbolicregression.utils.CUDA = not params.cpu
    return params


def load_dataset(dataset_root, dataset_name, max_rows):
    dataset_file = dataset_root / dataset_name / f"{dataset_name}.tsv.gz"
    X, y, feature_names = read_file(str(dataset_file))
    if max_rows > 0:
        X = X[:max_rows]
        y = y[:max_rows]
    return X, y, feature_names, dataset_file


def build_direct_sample(sample_to_learn, max_input_points):
    sub_sample = copy.deepcopy(sample_to_learn)

    seq_len = len(sample_to_learn["x_to_fit"][0])
    if seq_len >= max_input_points:
        random_indices = random.sample(list(range(seq_len)), max_input_points)
        sub_sample["X_scaled_to_fit"][0] = np.array([sample_to_learn["X_scaled_to_fit"][0][i] for i in random_indices])
        sub_sample["Y_scaled_to_fit"][0] = np.array([sample_to_learn["Y_scaled_to_fit"][0][i] for i in random_indices])
        sub_sample["x_to_fit"][0] = np.array([sample_to_learn["x_to_fit"][0][i] for i in random_indices])
        sub_sample["y_to_fit"][0] = np.array([sample_to_learn["y_to_fit"][0][i] for i in random_indices])

    seq_len = len(sample_to_learn["x_to_predict"][0])
    if seq_len >= max_input_points:
        random_indices = random.sample(list(range(seq_len)), max_input_points)
        sub_sample["x_to_predict"][0] = np.array([sample_to_learn["x_to_predict"][0][i] for i in random_indices])
        sub_sample["y_to_predict"][0] = np.array([sample_to_learn["y_to_predict"][0][i] for i in random_indices])

    return sub_sample


def run_direct_inference(sample_to_learn, env, params, model):
    sub_sample = build_direct_sample(sample_to_learn, params.max_input_points)
    encoded_y, generations, _ = model(sub_sample, max_len=params.max_target_len)
    return gen2eq(env, params, encoded_y, generations, sample_to_learn, stored_skeletons=[])


def main():
    parser = build_inference_parser()
    params = configure_params(parser.parse_args())

    dataset_root = Path(params.dataset_root) if params.dataset_root else resolve_pmlb_dataset_root()
    X, y, feature_names, dataset_file = load_dataset(dataset_root, params.dataset, params.max_rows)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=params.test_size, shuffle=True, random_state=params.random_state
    )

    env = build_env(params)
    env.rng = np.random.RandomState(0)
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    trainer.modules = reload_model(trainer.modules, params.reload_model)

    model = SNIPSymbolicRegressor(params=params, env=env, modules=trainer.modules)
    model.to(params.device)
    model.eval()

    scaler = utils_wrapper.StandardScaler() if params.rescale else None
    if scaler is not None:
        x_train_scaled = scaler.fit_transform(x_train)
    else:
        x_train_scaled = x_train

    sample_to_learn = {
        "X_scaled_to_fit": [x_train_scaled],
        "Y_scaled_to_fit": [y_train.reshape(-1, 1)],
        "x_to_fit": [x_train],
        "y_to_fit": [y_train.reshape(-1, 1)],
        "x_to_predict": [x_test],
        "y_to_predict": [y_test.reshape(-1, 1)],
    }

    start_time = time.time()
    with torch.inference_mode():
        try:
            (
                success,
                _skeleton_candidate,
                predicted_tree,
                complexity,
                _y_fit,
                _y_pred,
                _mse_fit,
                _mse_pred,
                _results_fit,
                results_predict,
            ) = run_direct_inference(sample_to_learn, env, params, model)
        except Exception:
            success = False
            predicted_tree = "NaN"
            complexity = np.nan
            results_predict = {}
    runtime_sec = time.time() - start_time

    expression = predicted_tree.infix() if success and predicted_tree != "NaN" else None
    rmse = results_predict.get("_rmse", [np.nan])[0] if success else np.nan
    r2 = results_predict.get("r2", [np.nan])[0] if success else np.nan

    result = pd.DataFrame(
        [
            {
                "dataset": params.dataset,
                "dataset_file": str(dataset_file),
                "rows_loaded": len(X),
                "n_features": len(feature_names),
                "train_size": len(x_train),
                "test_size": len(x_test),
                "mode": "direct_e2e",
                "expression": expression,
                "r2": r2,
                "rmse": rmse,
                "complexity": complexity,
                "runtime_sec": runtime_sec,
            }
        ]
    )

    output_path = Path(params.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(result.to_string(index=False))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
