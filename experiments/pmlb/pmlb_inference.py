import random
import sys
from collections import defaultdict
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
from LSO_fit import LSOFitNeverGrad, lso_fit
from model import SNIPSymbolicRegressor
from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from symbolicregression.trainer import Trainer


DEFAULT_VALIDATION_METRICS = "r2_zero,r2,_rmse,_complexity"
DEFAULT_REFINEMENT_TYPE = "lso"
RESULT_COLUMNS = [
    "dataset",
    "status",
    "n_features",
    "refinement_type",
    "r2",
    "rmse",
    "complexity",
    "seconds",
    "error",
    "noise_strength",
    "expr",
]


def build_inference_parser():
    parser = get_parser()
    parser.set_defaults(
        beam_size=2,
        max_input_points=200,
        lso_optimizer="gwo",
        lso_pop_size=50,
        lso_max_iteration=80,
        lso_stop_r2=0.99,
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
    parser.add_argument(
        "--noise_strength",
        type=float,
        default=0.0,
        help="Multiplicative Gaussian noise strength applied to training targets.",
    )
    parser.add_argument(
        "--noise_seed",
        type=int,
        default=0,
        help="Random seed used when injecting target noise.",
    )
    return parser


def configure_params(params):
    if params.noise_strength < 0:
        raise ValueError(f"noise_strength must be non-negative, got {params.noise_strength}.")

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
    params.n_trees_to_refine = params.beam_size
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


def prepare_sample(X, y, test_size, random_state, rescale, noise_strength, noise_seed):
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
    )

    if noise_strength > 0:
        rng = np.random.RandomState(noise_seed)
        noise = rng.normal(0, noise_strength, size=y_train.shape)
        y_train = y_train * (1 + noise)

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


def run_lso_inference(sample_to_learn, env, params, model):
    batch_results = defaultdict(list)
    if params.lso_optimizer == "gwo":
        return lso_fit(sample_to_learn, env, params, model, batch_results, 1)
    return LSOFitNeverGrad(env, params, model, sample_to_learn, batch_results, 1).fit_func()


def infer_dataset_result(dataset_root, dataset_name, env, params, model):
    result = {
        "dataset": dataset_name,
        "status": "error",
        "n_features": np.nan,
        "refinement_type": f"{DEFAULT_REFINEMENT_TYPE}_{params.lso_optimizer}",
        "r2": np.nan,
        "rmse": np.nan,
        "complexity": np.nan,
        "seconds": np.nan,
        "error": "",
        "noise_strength": params.noise_strength,
        "expr": "",
    }

    try:
        X, y, feature_names, _ = load_dataset(dataset_root, dataset_name, params.max_rows)
        sample_to_learn = prepare_sample(
            X=X,
            y=y,
            test_size=params.test_size,
            random_state=params.random_state,
            rescale=params.rescale,
            noise_strength=params.noise_strength,
            noise_seed=params.noise_seed,
        )

        with torch.no_grad():
            batch_results = run_lso_inference(sample_to_learn, env, params, model)

        final_tree = batch_results["final_predicted_tree"][0]
        result["status"] = "success"
        result["n_features"] = len(feature_names)
        result["r2"] = batch_results["r2_final_predict"][0]
        result["rmse"] = batch_results["_rmse_final_predict"][0]
        result["complexity"] = len(final_tree.prefix().split(","))
        result["seconds"] = batch_results["time"][0]
        result["expr"] = final_tree.infix()
    except Exception as exc:
        result["error"] = str(exc)

    return result


def write_single_result(output_csv, result):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result], columns=RESULT_COLUMNS).to_csv(output_path, index=False)


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
