import fredipy as fp
from scipy import optimize
from dataclasses import dataclass, field

import numpy as np
import json
import argparse
import time
import pprint
import os
from typing import List, Tuple, Callable

@dataclass
class gprParameters:
    Nt: int
    variance: List = field(default_factory=list)
    lengthscale: List = field(default_factory=list)
    optimizer: bool = False

    def __post_init__(self):        
        # Ensure variance and lengthscale are floats
        if not isinstance(self.variance, float):
            raise ValueError("Variance must be float.")
        if not isinstance(self.lengthscale, float):
            raise ValueError("Lengthscale must be float.")

class gaussianFit:
    def __init__(self, gprParameters:gprParameters):
        self.Nt = gprParameters.Nt
        self.variance = gprParameters.variance
        self.lengthscale = gprParameters.lengthscale
        self.optimizer = gprParameters.optimizer

    def OptimizeKernelParameters(
        self, gp: fp.models.GaussianProcess,
        guess: List[float],
        bound_min: float = 1e-4
        ):

        def optimized_function(params):
            gp.set_kernel_params(params)
            return - gp.log_likelihood()

        num_params = len(guess)
        bounds = [(bound_min, None)]*num_params

        res = optimize.minimize(
            optimized_function, guess, bounds = bounds, 
            method = 'L-BFGS-B')

        if res.success:
            print(' Optimized parameters are: ', res.x)
            gp.set_kernel_params(res.x)
        else:
            print('No convergence in optimization!')

    def fitCorrelator(
            self, x: np.ndarray, error: np.ndarray,
            correlator: np.ndarray, finiteT_kernel: bool, 
            Nt: int, omega: np.ndarray, 
            extractedQuantity: str = "RhoOverOmega", 
            verbose: bool = True
            ) -> np.ndarray:

        data = {
            'x': x,
            'y': correlator,
            'yerr': error
        }

        rbf = fp.kernels.RadialBasisFunction(self.variance, self.lengthscale)
        integrator = fp.integrators.Riemann_1D(0,5,500)

        if extractedQuantity=="RhoOverOmega" and finiteT_kernel:
            integral_op = fp.operators.Integral(
                KL_kernel_Omega_fin_T, integrator)
        elif extractedQuantity=="RhoOverOmega" and finiteT_kernel==False:
            integral_op = fp.operators.Integral(
                KL_kernel_Omega_Vacuum, integrator)
        elif extractedQuantity=="Rho" and finiteT_kernel:
            integral_op = fp.operators.Integral(
                KL_kernel_Position_FiniteT, integrator)
        elif extractedQuantity=="Rho" and finiteT_kernel==False:
            integral_op = fp.operators.Integral(
                KL_kernel_Position_Vacuum, integrator)
        else:
            raise ValueError("Invalid choice spectral function target")
        
        constraints = [fp.constraints.LinearEquality(integral_op, data)]
        model = fp.models.GaussianProcess(rbf, constraints)
        if self.optimizer:
            if verbose:
                print("Optimizing Kernel Parameters...")
            self.OptimizeKernelParameters(model, [self.variance, self.lengthscale])

        res, res_err = model.predict(omega)
        return res, res_err

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
    
    def getgprParams(self) -> gprParameters:
        return gprParameters(
            Nt=self.params["Nt"],
            variance=self.params["variance"],
            lengthscale=self.params["lengthscale"],
            optimizer=self.params["optimizer"],
        )
    
    def get_extractedQuantity(self) -> str:
        return self.params["extractedQuantity"]
    
    def get_correlator_file(self) -> str:
        return os.path.abspath(self.params["correlatorFile"])
    
    def get_verbose(self) -> bool:
        return self.params["verbose"]
    
    def get_correlator_cols(self) -> List[int]:
        correlator_cols = self.params["correlatorCols"]
        if isinstance(correlator_cols, list):
            return correlator_cols
        if isinstance(correlator_cols, int):
            return [correlator_cols]
        if isinstance(correlator_cols, str) and ':' in correlator_cols:
            start_str, end_str = correlator_cols.split(':')
            start = int(start_str) if start_str else None
            end = int(end_str) if end_str else None
            return list(range(start if start is not None else 0, end + 1 if end is not None else len(np.loadtxt(self.params["correlatorFile"], max_rows=1))))
        if isinstance(correlator_cols, str) and correlator_cols.isdigit():
            return [int(correlator_cols)]
        if not correlator_cols:
            return []
        raise ValueError("correlator_cols must be an integer index, list of indices, or a string with a range (e.g., '6:10', '6:', ':10', ':').")

class FitRunner:
    def __init__(self, parameterHandler: ParameterHandler):
        self.parameterHandler = parameterHandler
        self.gpr_params = self.parameterHandler.getgprParams()
        self.fitter = gaussianFit(self.gpr_params)
        self.x, self.mean, self.error, self.correlators = self.extractColumns(
            self.parameterHandler.get_correlator_file(),
            self.parameterHandler.get_params()["xCol"],
            self.parameterHandler.get_params()["meanCol"],
            self.parameterHandler.get_params()["errorCol"],
            self.parameterHandler.get_correlator_cols()
        )
        self.omega = np.linspace(
            self.parameterHandler.get_params()["omega_min"],
            self.parameterHandler.get_params()["omega_max"],
            self.parameterHandler.get_params()["omega_points"]
        )
        self.finiteT_kernel = self.parameterHandler.get_params()["FiniteT_kernel"]
        self.verbose = self.parameterHandler.get_verbose()
        self.multiFit = self.parameterHandler.get_params()["multiFit"]
        self.extractedQuantity = self.parameterHandler.get_extractedQuantity()
        self.Nt = self.parameterHandler.get_params()["Nt"] or len(self.x)
        self.outputDir = os.path.abspath(self.parameterHandler.get_params()["outputDir"])
        self.outputFile = self.parameterHandler.get_params()["outputFile"] or f"{self.extractedQuantity}_{os.path.basename(self.parameterHandler.get_correlator_file())}"

    def extractColumns(self, file: str, x_col: int, mean_col: int, error_col: int, correlator_cols: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data = np.loadtxt(file)
        x = data[:, x_col]
        mean = data[:, mean_col]
        error = data[:, error_col]
        correlator = data[:, correlator_cols]
        return x, mean, error, correlator    

    def run_single_fit(self, fittedQuantity, messageString ,results: List[np.ndarray]) -> None:
        start_time = time.time()
        print("=" * 40)
        print(messageString)
        print("=" * 40)
        sf = self.fitter.fitCorrelator(
            self.x,
            self.error,
            fittedQuantity,
            self.finiteT_kernel,
            self.Nt,
            self.omega,
            extractedQuantity=self.extractedQuantity,
            verbose=self.verbose
        )
        if self.verbose:
            print("-" * 40)
            print(f"Time: {time.time() - start_time:.2f} seconds")
        results.append(sf)
    
    def run_fits(self) -> np.ndarray:
        results = []
        if self.correlators.ndim == 1:
            self.correlators = np.array([self.correlators])
        else:
            self.correlators = self.correlators.T
        n_correlators = self.correlators.shape[0]
        if self.multiFit:
            self.run_single_fit(self.correlators, f"Multifitting {n_correlators} correlators", results)
        else:
            self.run_single_fit(self.mean, "Fitting mean correlator", results)
            for i, corr in enumerate(self.correlators):
                self.run_single_fit(corr, f"Fitting correlator sample {i + 1}/{n_correlators}", results)
        return np.array(results)

    def calculate_mean_error(self, mean: np.ndarray, samples: np.ndarray, errormethod: str = "jackknife") -> np.ndarray:
        N = len(samples)
        fac = N - 1 if errormethod == "jackknife" else 1
        if errormethod not in ["jackknife", "bootstrap"]:
            raise ValueError("Invalid choice of error estimation method")
        return np.sqrt(fac / N * np.sum((samples - mean) ** 2, axis=0))

    def save_results(
            self, mean: np.ndarray, error: np.ndarray, 
            samples: np.ndarray, 
            extractedQuantity: str = "RhoOverOmega"
            ) -> None:
        header = "Omega " + self.extractedQuantity + "_mean"
        if samples is not None and error is not None:
            header += f" {self.extractedQuantity}_error"
            for i in range(len(samples)):
                header += f" {self.extractedQuantity}_sample_{i}"
            writeData = np.column_stack((self.omega, mean, error, samples.T))
        else:
            writeData = np.column_stack((self.omega, mean))
        np.savetxt(os.path.join(self.outputDir, self.outputFile + ".gpr"), writeData, header=header)
        if self.parameterHandler.get_params()["saveParams"]:
            self.save_params(self.parameterHandler.get_params(), os.path.join(self.outputDir, self.outputFile + ".params"))

    def save_params(self, params: dict, outputFile: str) -> None:
        with open(outputFile + '.json', 'w') as f:
            json.dump(params, f, indent=4)

def initializeArgumentParser(paramsDefaultDict: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpr_rec",
        description="Fit spectral functions to provided correlators using Gaussian Process Regression."
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

def main(paramsDefaultDict):
    parser=initializeArgumentParser(paramsDefaultDict)
    args = parser.parse_args()
    parameterHandler = ParameterHandler(paramsDefaultDict)
    parameterHandler.load_params(args.config, args)
    
    global Nt
    Nt = parameterHandler.get_params()["Nt"]

    if parameterHandler.get_verbose():
        print("*"*40)
        print("Running fits with the following parameters:")
        pprint.pprint(parameterHandler.get_params())

    fitRunner = FitRunner(parameterHandler)
    results = fitRunner.run_fits()
    mean = results[0]
    if len(results)>1:
        samples = results[1:]
        error = fitRunner.calculate_mean_error(samples,mean,parameterHandler.get_params()["errormethod"])
    else:
        samples = None
        error = None
    fitRunner.save_results(mean,error,samples)

def KL_kernel_Position_Vacuum(Position, Omega):
    Position = Position[:, np.newaxis]  # Reshape Position as column to allow broadcasting
    ker = np.exp(-Omega * np.abs(Position))
    return np.squeeze(ker)

def KL_kernel_Position_FiniteT(Position, Omega):
    if type(Position) == np.ndarray:
        Position = Position[:, np.newaxis]  # Reshape Position as column to allow broadcasting
        with np.errstate(divide='ignore'):
            ker = np.cosh(Omega * (Position-1/(2)*Nt)) / np.sinh(Omega*Nt/2)

            # set all entries in ker to 1 where Position is modulo 1/T and the entry is nan, because of numerical instability for large Omega
            ker[np.isnan(ker) & (Position % (Nt) == 0)] = 1
            #set all other nan entries to 0
            ker[np.isnan(ker)] = 0
        return np.squeeze(ker)
    else:
        return np.cosh(Omega*(Position - 0.5*Nt))/np.sinh(0.5*Omega*Nt)

def KL_kernel_Omega_fin_T(x,Omega):
    ret=KL_kernel_Position_FiniteT(x, Omega)
    ret=np.expand_dims(ret, axis=1)
    ret[:,Omega==0]=1
    ret=Omega * ret
    ret[:,Omega==0]=2*1/Nt
    return np.squeeze(ret)

def KL_kernel_Omega_Vacuum(x,Omega):
    ret = KL_kernel_Position_Vacuum(x,Omega)
    ret=np.expand_dims(ret, axis=1)
    ret[:,Omega==0]=1
    ret=Omega * ret
    ret[:,Omega==0]=0
    return np.squeeze(ret)


paramsDefaultDict = {
    "Method": "Gaussian",
    #Gaussian specific
    "optimizer": False,
    "variance": 0.3,
    "lengthscale": 0.4,
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
    "outputDir": ""

}


if __name__ == "__main__":
    main(paramsDefaultDict)

