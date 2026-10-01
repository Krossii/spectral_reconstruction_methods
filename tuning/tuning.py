import argparse
import copy
import os
import sys

from ray import tune


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPERVISED_ML_DIR = os.path.join(REPO_ROOT, "supervised_ml")
DEFAULT_PARAMS_PATH = os.path.join(SUPERVISED_ML_DIR, "params.json")
if SUPERVISED_ML_DIR not in sys.path:
    sys.path.insert(0, SUPERVISED_ML_DIR)

from supervisedml import FitRunner, ParameterHandler, paramsDefaultDict


def ray_tune_wrapper(config, params_path=DEFAULT_PARAMS_PATH):
    """Train one supervised model and report its held-out test loss."""
    params_path = os.path.abspath(params_path)
    params = copy.deepcopy(paramsDefaultDict)
    parameter_handler = ParameterHandler(params)
    parameter_handler.load_from_json(params_path)

    params = parameter_handler.get_params()
    correlator_path = params["correlatorFile"]
    if not os.path.isabs(correlator_path):
        params["correlatorFile"] = os.path.abspath(
            os.path.join(os.path.dirname(params_path), correlator_path)
        )

    params["networkStructure"] = config["networkStructure"]
    params["learning_rate"] = [float(config["learning_rate"])]
    params["batch_size"] = int(config["batch_size"])
    params["epochs"] = [int(config["epochs"])]

    fit_runner = FitRunner(parameter_handler)
    old_working_directory = os.getcwd()
    try:
        try:
            trial_directory = tune.get_context().get_trial_dir()
        except AttributeError:
            trial_directory = tune.get_trial_dir()
        os.makedirs(trial_directory, exist_ok=True)
        os.chdir(trial_directory)

        _, _, _, validation_loss = fit_runner.fitter.fitCorrelator(
            fit_runner.x,
            fit_runner.error,
            fit_runner.mean,
            fit_runner.finiteT_kernel,
            fit_runner.Nt,
            fit_runner.omega,
            data_noise=fit_runner.data_noise,
            extractedQuantity=fit_runner.extractedQuantity,
            verbose=False,
            samples_per_epoch=int(config["samples_per_epoch"]),
            data_seed=0,
            return_eval=True,
            save_test_plots=False,
        )
    finally:
        os.chdir(old_working_directory)

    tune.report({"loss": validation_loss})


def run_hyperparameter_search(
        params_path=DEFAULT_PARAMS_PATH,
        num_samples=10,
        gpu_per_trial=0,
        ):
    search_space = {
        "networkStructure": tune.choice(["SupervisedNN", "KadesFC", "KadesConv"]),
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([32, 64, 128]),
        "epochs": tune.choice([3, 5, 10]),
        "samples_per_epoch": tune.choice([10000, 50000]),
    }
    trainable = tune.with_parameters(
        ray_tune_wrapper,
        params_path=os.path.abspath(params_path),
    )
    return tune.run(
        trainable,
        config=search_space,
        num_samples=num_samples,
        metric="loss",
        mode="min",
        resources_per_trial={"cpu": 2, "gpu": gpu_per_trial},
    )


def main():
    parser = argparse.ArgumentParser(
        description="Tune supervised spectral reconstruction networks with Ray."
    )
    parser.add_argument("--config", default=DEFAULT_PARAMS_PATH,
                        help="Path to the supervisedml parameter JSON file.")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of Ray Tune trials.")
    parser.add_argument("--gpu-per-trial", type=float, default=0,
                        help="GPUs reserved per trial (use 0 for CPU-only tuning).")
    args = parser.parse_args()
    run_hyperparameter_search(
        params_path=args.config,
        num_samples=args.num_samples,
        gpu_per_trial=args.gpu_per_trial,
    )


if __name__ == "__main__":
    main()
