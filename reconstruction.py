#run this with a .json parameter file to reconstruct a correlator
import json
import argparse
import pprint
from typing import List, Tuple, Callable
import subprocess
import os

def initializeArgumentParser(
        paramsDefaultDict: dict
        ) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reconstruction",
        description="Fit spectral functions to provided correlators using specified reconstruction methods."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="",
        help="Path to JSON configuration file"
    )
    
    for name, default in paramsDefaultDict.items():
        typeArg=type(default)
        typeString=typeArg.__name__
        if typeArg==list:
            nargsArg='+'
            typeArg=type(default[0])
            typeString=f"List of {typeArg.__name__}"
        else:
            nargsArg=None
        parser.add_argument(
            f"--{name}",
            type=typeArg,
            nargs=nargsArg,
            # default=default,
            help=f"Value for parameter '{name}'"
        )
    return parser

class ParameterHandler:
    def __init__(self, paramsDefaultDict: dict):
        self.allowed_params = paramsDefaultDict.keys()
        self.params = paramsDefaultDict

    def load_from_json(self, config_path: str) -> None:
        if config_path:
            with open(config_path, 'r') as f:
                data = json.load(f)
            for name in self.allowed_params:
                if name in data:
                    self.params[name] = data[name]

    def override_with_args(self, args: argparse.Namespace) -> None:
        for name in self.allowed_params:
            val = getattr(args, name, None)
            if val is not None:
                self.params[name] = val

    def check_parameters(self) -> None:
        for name in self.allowed_params:
            if name == "outputFile" and self.params[name] is None:
                continue
            if name not in self.params or self.params[name] is None:
                raise ValueError(f"Parameter '{name}' is not set.")
    
    def load_params(self, config_path: str, args: argparse.Namespace) -> None:
        self.load_from_json(config_path)
        self.override_with_args(args)
        self.check_parameters()

    def get_params(self) -> dict:
        return self.params

    def get_verbose(self) -> bool:
        return self.params["verbose"]

    def get_blacklisted_dict(self, bvals) -> dict:
        cleaned_dict = {
                k:v for k,v in self.params.items() if k not in bvals
        }
        return cleaned_dict

    def write_new_json(self) -> None:
        if self.get_params()["cluster"]:
            cluster_path = self.get_params()["clusterpath"]
        if self.params["Method"] == "UnsupervisedNN":
            black_list_vals = set((
                "Method", "batch_size", "create_data","data_noise", "trainingFile", 
                "validationFile", "optimizer", "variance", "model_file",
                "lengthscale", "alpha_min", "alpha_max",
                "alpha_points", "default_model","eval_model"
                ))
            cleaned_dict = self.get_blacklisted_dict(black_list_vals)
            cleaned_dict["networkStructure"] = "SpectralNN"
            subpath = "neuralFit/params.json"
            if self.get_params()["cluster"]:
                subpath = os.path.join(cluster_path, subpath)
            with open(subpath, "w") as f:
                json.dump(cleaned_dict, f, indent=4)
        if self.params["Method"] == "SupervisedNN" or self.params["Method"] == "KadesFC" or self.params["Method"] == "KadesConv":
            black_list_vals = set((
                "Method", "width", "create_data","data_noise", "optimizer",
                "variance", "lengthscale", "alpha_min", "alpha_max",
                "alpha_points", "default_model", "multiFit"
                ))
            cleaned_dict = self.get_blacklisted_dict(black_list_vals)
            cleaned_dict["networkStructure"] = self.params["Method"]
            subpath = "supervised_ml/params.json"
            if self.get_params()["cluster"]:
                subpath = os.path.join(cluster_path, subpath)
            with open(subpath, "w") as f:
                json.dump(cleaned_dict, f, indent=4)
        if self.params["Method"] == "Gaussian":
            black_list_vals = set((
                "lambda_s", "lambda_l2", "epochs","eval_model",
                "learning_rate", "errorWeighting", "width", "model_file",
                "batch_size", "create_data","data_noise", "trainingFile", "validationFile",
                "saveLossHistory", "alpha_min", "alpha_max",
                "alpha_points", "default_model"
                ))
            cleaned_dict = self.get_blacklisted_dict(black_list_vals)
            subpath = "gaussian/params.json"
            if self.get_params()["cluster"]:
                subpath = os.path.join(cluster_path, subpath)
            with open(subpath, "w") as f:
                json.dump(cleaned_dict, f, indent=4)
        if self.params["Method"] == "MEM":
            black_list_vals = set((
                "lambda_s", "lambda_l2", "epochs","eval_model",
                "learning_rate", "errorWeighting", "width", "model_file",
                "batch_size", "create_data","data_noise", "trainingFile", "validationFile",
                "saveLossHistory", "optimizer", "variance", "lengthscale"
                ))
            cleaned_dict = self.get_blacklisted_dict(black_list_vals)
            subpath = "mem/params.json"
            if self.get_params()["cluster"]:
                subpath = os.path.join(cluster_path, subpath)
            with open(subpath, "w") as f:
                json.dump(cleaned_dict, f, indent=4)

def call_create_data_program(parameterHandler: ParameterHandler):
    if parameterHandler.get_verbose():
        print("*"*40)
        print("Checking for create_data:")
        print(parameterHandler.get_params()["create_data"])

    if parameterHandler.get_params()["create_data"]:

        if parameterHandler.get_params()["cluster"]:
            working_dir = os.path.join(parameterHandler.get_params()["clusterpath"], "supervised_ml/")
        else:
            working_dir = "supervised_ml/"
        subprocess.Popen(["python", "create_data.py", "--config", "../params.json"], cwd=working_dir).communicate()
        if parameterHandler.get_verbose():
            print("*"*40)
            print("Successfully ran data creation.")


def call_method_programs(parameterHandler: ParameterHandler):
    if parameterHandler.get_verbose():
        print("*"*40)
        print("Calling program for method:")
        print(parameterHandler.get_params()["Method"])

    if parameterHandler.get_params()["Method"] == "UnsupervisedNN":
        if parameterHandler.get_params()["cluster"]:
            working_dir = os.path.join(parameterHandler.get_params()["clusterpath"], "neuralFit/")
        else:
            working_dir = "neuralFit/"
        subprocess.Popen(["python", "neuralFit.py", "--config", "params.json"], cwd=working_dir).communicate()

    if parameterHandler.get_params()["Method"] == "SupervisedNN" or parameterHandler.get_params()["Method"] == "KadesFC" or parameterHandler.get_params()["Method"] == "KadesConv":
        if parameterHandler.get_params()["cluster"]:
            working_dir = os.path.join(parameterHandler.get_params()["clusterpath"], "supervised_ml/")
        else:
            working_dir = "supervised_ml/"
        subprocess.Popen(["python", "supervisedml.py", "--config", "params.json"], cwd=working_dir).communicate()

    if parameterHandler.get_params()["Method"] == "Gaussian":
        if parameterHandler.get_params()["cluster"]:
            working_dir = os.path.join(parameterHandler.get_params()["clusterpath"], "gaussian/")
        else:
            working_dir = "gaussian/"
        subprocess.Popen(["python", "gpr_rec.py", "--config", "params.json"], cwd=working_dir).communicate()
    
    if parameterHandler.get_params()["Method"] == "MEM":
        if parameterHandler.get_params()["cluster"]:
            working_dir = os.path.join(parameterHandler.get_params()["clusterpath"], "mem/")
        else:
            working_dir = "mem/"
        subprocess.Popen(["python", "mem.py", "--config", "params.json"], cwd=working_dir).communicate()

def main(paramsDefaultDict):
    parser=initializeArgumentParser(paramsDefaultDict)
    args = parser.parse_args()
    parameterHandler = ParameterHandler(paramsDefaultDict)
    parameterHandler.load_params(args.config, args)
    call_create_data_program(parameterHandler)
    parameterHandler.write_new_json()
    call_method_programs(parameterHandler)


paramsDefaultDict = {
    #choice of SupervisedNN, KadesFC, UnsupervisedNN, Gaussian, MEM
    "Method": "UnsupervisedNN",
    #NetworkParams (Ai specrec)
    "lambda_s": [1e-5],
    "lambda_l2": [1e-8],
    "epochs": [100],
    "learning_rate": [1e-4],
    "errorWeighting": True,
    "saveLossHistory": False,
    #Unsupervised specific
    "width": [32,32,32],
    #Supervised specific
    "batch_size": 128,
    "create_data": False,
    "data_noise": 10e-5,
    "trainingFile": "",
    "validationFile": "",
    "eval_model": False,
    "model_file": "",
    #Gaussian specific
    "optimizer": False,
    "variance": 0.3,
    "lengthscale": 0.4,
    #MEM specific; default model: choice of "quadratic" or "constant"
    "alpha_min": 0,
    "alpha_max": 10,
    "alpha_points": 64,
    "default_model": "constant",
    #Correlator/Rho params
    "omega_min": 0,
    "omega_max": 10,
    "omega_points": 500,
    "Nt": 0,
    "extractedQuantity": "RhoOverOmega",
    "FiniteT_kernel": True,
    "multiFit": False,
    "correlatorFile": "",
    "xCol": 0,
    "meanCol": 1,
    "errorCol": 2,
    "correlatorCols": "3:",
    "errormethod": "jackknife",
    #General Params
    "saveParams": False,
    "verbose": False,
    "outputFile": "",
    "outputDir": "",
    "cluster": False,
    "clusterpath": ""
}



if __name__ == "__main__":
    main(paramsDefaultDict)