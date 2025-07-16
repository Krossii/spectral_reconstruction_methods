#from train import run_training

import tensorflow as tf
from tensorflow import keras
from ray import tune


import subprocess
import json
import os


def ray_tune_wrapper(config):
    loss = run_training(config, return_eval=True)
    tune.report(loss=loss)

def run_hyperparameter_search():
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        "epochs": 5,
    }
    tune.run(
        ray_tune_wrapper,
        config=search_space,
        num_samples=10,
        metric="loss",
        mode="min",
        resources_per_trial={"cpu": 2, "gpu": 1},
    )

