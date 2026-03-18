import copy
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import symbolicregression
import symbolicregression.model.utils_wrapper as utils_wrapper
from LSO_eval import read_file, resolve_pmlb_dataset_root
from LSO_fit import gen2eq
from model import SNIPSymbolicRegressor
from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from symbolicregression.trainer import Trainer


DEFAULT_VALIDATION_METRICS = "r2_zero,r2,_rmse,_complexity"
DIRECT_REFINEMENT_TYPE = "direct_e2e"


def build_inference_parser():
    parser = get_parser()
    parser.set_defaults(
        beam_size=2,
        max_input_points=200,
        validation_metrics=DEFAULT_VALIDATION_METRICS,
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
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use: cpu, cuda, cuda:0, cuda:1, ...",
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
    params.validation_metrics = DEFAULT_VALIDATION_METRICS

    np.random.seed(params.seed)
    random.seed(params.seed)
    torch.manual_seed(params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(params.seed)

    params.device = resolve_device(params.device)
    params.cpu = params.device.type == "cpu"
    symbolicregression.utils.CUDA = not params.cpu
    return params


def resolve_device(device_arg):
    device_str = str(device_arg).strip().lower()
    if device_str == "cpu":
        return torch.device("cpu")

    if not device_str.startswith("cuda"):
        raise ValueError(f"Unsupported device '{device_arg}'. Use cpu, cuda, or cuda:N.")

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device '{device_arg}', but CUDA is not available in this environment."
        )

    if device_str == "cuda":
        torch.cuda.set_device(0)
        return torch.device("cuda:0")

    if device_str.startswith("cuda:"):
        index_str = device_str.split(":", 1)[1]
        if not index_str.isdigit():
            raise ValueError(f"Invalid CUDA device '{device_arg}'. Expected cuda:N.")
        index = int(index_str)
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested device '{device_arg}', but only {torch.cuda.device_count()} CUDA device(s) are available."
            )
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")

    raise ValueError(f"Unsupported device '{device_arg}'. Use cpu, cuda, or cuda:N.")


def load_pretrained_modules(modules, checkpoint_path, device):
    if not checkpoint_path:
        raise ValueError("--reload_model must point to a pretrained checkpoint.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    for name, module in modules.items():
        weights = checkpoint[name]
        if all(key.startswith("module.") for key in weights.keys()):
            weights = {key[len("module."):]: value for key, value in weights.items()}
        module.load_state_dict(weights)
        module.requires_grad = False
    return modules


def create_inference_components(params):
    env = build_env(params)
    env.rng = np.random.RandomState(0)
    checkpoint_path = params.reload_model
    params.reload_model = ""
    modules = build_modules(env, params)
    params.reload_model = checkpoint_path
    trainer = Trainer(modules, env, params)
    trainer.modules = load_pretrained_modules(trainer.modules, params.reload_model, params.device)
    model = SNIPSymbolicRegressor(params=params, env=env, modules=trainer.modules)
    model.to(params.device)
    model.eval()
    return env, model


def load_dataset(dataset_root, dataset_name, max_rows):
    dataset_file = dataset_root / dataset_name / f"{dataset_name}.tsv.gz"
    X, y, feature_names = read_file(str(dataset_file))
    if max_rows > 0:
        X = X[:max_rows]
        y = y[:max_rows]
    return X, y, feature_names, dataset_file


def prepare_sample(X, y, test_size, random_state, rescale):
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
    )

    scaler = utils_wrapper.StandardScaler() if rescale else None
    x_train_scaled = scaler.fit_transform(x_train) if scaler is not None else x_train

    sample_to_learn = {
        "X_scaled_to_fit": [x_train_scaled],
        "Y_scaled_to_fit": [y_train.reshape(-1, 1)],
        "x_to_fit": [x_train],
        "y_to_fit": [y_train.reshape(-1, 1)],
        "x_to_predict": [x_test],
        "y_to_predict": [y_test.reshape(-1, 1)],
    }
    return sample_to_learn


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


def infer_dataset_result(dataset_root, dataset_name, env, params, model):
    start_time = time.time()
    result = {
        "dataset": dataset_name,
        "status": "error",
        "n_features": np.nan,
        "refinement_type": DIRECT_REFINEMENT_TYPE,
        "r2": np.nan,
        "rmse": np.nan,
        "complexity": np.nan,
        "expr": None,
        "seconds": np.nan,
        "error": "",
    }

    try:
        X, y, feature_names, dataset_file = load_dataset(dataset_root, dataset_name, params.max_rows)
        sample_to_learn = prepare_sample(
            X=X,
            y=y,
            test_size=params.test_size,
            random_state=params.random_state,
            rescale=params.rescale,
        )
        result["n_features"] = len(feature_names)

        with torch.inference_mode():
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

        if success and predicted_tree != "NaN":
            result["status"] = "success"
            result["expr"] = predicted_tree.infix()
            result["complexity"] = complexity
            result["r2"] = results_predict.get("r2", [np.nan])[0]
            result["rmse"] = results_predict.get("_rmse", [np.nan])[0]
        else:
            result["error"] = f"Failed to decode expression for {dataset_file.name}."
    except Exception as exc:
        result["error"] = str(exc)

    result["seconds"] = time.time() - start_time
    return result


def write_single_result(output_csv, result):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(output_path, index=False)


def main():
    parser = build_inference_parser()
    params = configure_params(parser.parse_args())

    dataset_root = Path(params.dataset_root) if params.dataset_root else resolve_pmlb_dataset_root()
    env, model = create_inference_components(params)
    result = infer_dataset_result(dataset_root, params.dataset, env, params, model)
    write_single_result(params.output_csv, result)

    print(pd.DataFrame([result]).to_string(index=False))
    print(f"Saved results to {params.output_csv}")


if __name__ == "__main__":
    main()
