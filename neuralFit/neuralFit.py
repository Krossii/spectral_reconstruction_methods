#!/usr/bin/env python

import tensorflow as tf
import keras
from dataclasses import dataclass, field

import numpy as np
import json
import argparse
import time
import pprint
import os
from typing import List, Tuple, Callable

# Define the kernel functions

def KL_kernel_Momentum(Momentum, Omega):
    Momentum = Momentum[:, np.newaxis]  # Reshape Momentum as column to allow broadcasting
    ker = Omega / (Omega**2 + Momentum**2)  # Element-wise division
    return ker / np.pi 

def KL_kernel_Position_Vacuum(Position, Omega):
    Position = Position[:, np.newaxis]  # Reshape Position as column to allow broadcasting
    ker = np.exp(-Omega * np.abs(Position))
    return ker

def KL_kernel_Position_FiniteT(Position, Omega,T):
    Position = Position[:, np.newaxis]  # Reshape Position as column to allow broadcasting
    with np.errstate(divide='ignore'):
        ker = np.cosh(Omega * (Position-1/(2*T))) / np.sinh(Omega/2/T)

        # set all entries in ker to 1 where Position is modulo 1/T and the entry is nan, because of numerical instability for large Omega
        ker[np.isnan(ker) & (Position % (1/T) == 0)] = 1
        #set all other nan entries to 0
        ker[np.isnan(ker)] = 0

    return ker

def KL_kernel_Omega(KL,x,Omega,args=[]):
    ret=KL(x, Omega, *args)
    ret[:,Omega==0]=1
    ret=Omega * ret
    # set for all Omega=0 to 1
    if len(args)==0:
        ret[:,Omega==0]=0
    else:
        ret[:,Omega==0]=2*args[0]
    return ret

def Di(KL, rhoi, delomega):
    # Ensure both tensors are of the same data type (float32)
    KL = tf.cast(KL, dtype=tf.float32)  # Cast KL to float32
    rhoi = tf.cast(rhoi, dtype=tf.float32)  # Cast rhoi to float32
    delomega = tf.cast(delomega, dtype=tf.float32)  # Cast delomega to float32
    
    # Ensure rhoi has the correct shape [500,1] for matrix multiplication
    rhoi = tf.reshape(rhoi, [-1, 1])  # Reshape to [500, 1]

    # Perform matrix multiplication
    dis = tf.matmul(KL, rhoi)  # Shape will be [25, 1]
    dis = tf.squeeze(dis, axis=-1)  # Remove the singleton dimension to get [25]
    
    dis = dis * delomega  # Multiply by delomega
    return dis



class SpectralNN(tf.keras.Model):
    def __init__(self, num_output_nodes: int, width: List[int]):
        super(SpectralNN, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(w, activation='elu', use_bias=False) for w in width]
        self.output_layer = tf.keras.layers.Dense(num_output_nodes, activation=tf.keras.activations.softplus, use_bias=False)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

# class SpectralNNP2P(tf.keras.Model):
#     def __init__(self, width=[32]):
#         super(SpectralNNP2P, self).__init__()

#         # Create hidden layers
#         self.hidden_layers = []
#         for w in width:
#             self.hidden_layers.append(tf.keras.layers.Dense(w, activation='elu'))
        
#         # Output layer that produces one output per frequency (500 outputs)
#         self.output_layer = tf.keras.layers.Dense(1, activation='softplus')
    
#     def call(self, inputs):
#         x = inputs
#         for layer in self.hidden_layers:
#             x = layer(x)
#         return self.output_layer(x)
 
class LossCalculator:
    def __init__(self, model: tf.keras.Model=None,y_true:tf.Tensor=None,std=None,kernel: tf.Tensor=None,
                 delomega: tf.Tensor =None,
                 x:tf.Tensor=None,
                 lambda_s_func: Callable[[int], float]=lambda x:0.0,
                 lambda_l2_func: Callable[[int], float]=lambda x:0.0,):
        
        self.model = model
        self.y_true = y_true
        self.x=x
        self.std = std

        if self.std is None:
            self.std = tf.constant(1.0, dtype=tf.float32)
        else:
            self.std = tf.cast(self.std, dtype=tf.float32)
        self.kernel = kernel
        self.delomega = delomega
        self.lambda_s_func = lambda_s_func
        self.lambda_l2_func = lambda_l2_func


    def get_lambda_s(self, epoch: int) -> float:
        return self.lambda_s_func(epoch)
    
    def get_lambda_l2(self, epoch: int) -> float:
        return self.lambda_l2_func(epoch)
    
    def l2_regularization(self, weights: List[tf.Tensor] = None) -> tf.Tensor:
        if weights is None:
            weights = self.model.trainable_weights
        return tf.reduce_sum([tf.nn.l2_loss(w) for w in weights])
    
    def smoothness_loss(self, rho: tf.Tensor = None) -> tf.Tensor:
        if rho is None:
            rho = self.model(self.x)
        return tf.reduce_sum(tf.square(rho[:, 1:] - rho[:, :-1]))
    
    def custom_loss(self, y_pred: tf.Tensor, y_true: tf.Tensor = None) -> tf.Tensor:
        if y_true is None:
            y_true = self.y_true
        weighting = tf.cast(self.std, dtype=tf.float32)
        weighting /= weighting.numpy()[0]
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        chi_squared = tf.square((y_true - y_pred) / weighting)
        return tf.reduce_mean(chi_squared)
    
    def total_loss(self, epoch: int, y_pred: tf.Tensor = None, rho: tf.Tensor = None, y_true: tf.Tensor = None) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        if rho is None:
            rho = self.model(self.x)
        if y_pred is None:
            y_pred = Di(self.kernel, rho, self.delomega)
        if y_true is None:
            y_true = self.y_true
        main_loss = self.custom_loss(y_pred, y_true)
        smooth_loss = self.smoothness_loss(rho)
        l2_loss = self.l2_regularization()
        total_loss_value = main_loss + self.get_lambda_s(epoch) * smooth_loss + self.get_lambda_l2(epoch) * l2_loss
        return total_loss_value, [main_loss, self.get_lambda_s(epoch)*smooth_loss, self.get_lambda_l2(epoch)*l2_loss]
    
   
class networkTrainer:
    def __init__(self, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, loss_calculator: LossCalculator):
        self.model = model
        self.optimizer = optimizer
        self.loss_calculator = loss_calculator
    
    def train_step(self, epoch: int) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        with tf.GradientTape() as tape:
            total_loss_value, individual_losses = self.loss_calculator.total_loss(epoch)

        # Compute gradients and update weights
        gradients = tape.gradient(total_loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
    
        return total_loss_value, individual_losses
    
    def train(self, num_epochs: int, verbose: bool = False, start_epoch: int = 0) -> Tuple[List[tf.Tensor], List[List[tf.Tensor]]]:
        losses = []
        individual_losses_history = []
        net_num_epochs = num_epochs - start_epoch
        if verbose:
            print(f'Training for {net_num_epochs} epochs')
            start_time = time.time()
        for epoch in range(start_epoch, start_epoch + num_epochs):
            total_loss_value, individual_losses = self.train_step(epoch)
            losses.append(total_loss_value)
            individual_losses_history.append(individual_losses)
            if verbose and (epoch == 0 or (net_num_epochs > 10 and epoch % (net_num_epochs // 10) == 0)):
                print(f'Epoch {epoch}, Loss: {total_loss_value}')
        print(f'Epoch {epoch}, Loss: {total_loss_value}')
        if verbose:
            end_time = time.time()
            print(f'Training took {end_time - start_time:.2f} seconds')
        return losses, individual_losses_history

# Interface and runner classes

@dataclass
class networkParameters:
    lambda_s: List = field(default_factory=list)
    lambda_l2: List = field(default_factory=list)
    epochs: List = field(default_factory=list)
    learning_rate: List = field(default_factory=list)
    width: List = field(default_factory=list)
    errorWeighting: bool = False
    networkStructure: str = "SpectralNN"

    def __post_init__(self):
        # Ensure lambda_s, lambda_l2, epochs, and learning_rate are lists of the same length
        if not (len(self.lambda_s) == len(self.lambda_l2) == len(self.epochs) == len(self.learning_rate)):
            raise ValueError("lambda_s, lambda_l2, epochs, and learning_rate must be lists of the same length.")
        
        # Ensure all entries in lambda_s, lambda_l2 and learning_rate are floats
        if not all(isinstance(item, float) for item in self.lambda_s):
            raise ValueError("All entries in lambda_s must be floats.")
        if not all(isinstance(item, float) for item in self.lambda_l2):
            raise ValueError("All entries in lambda_l2 must be floats.")
        if not all(isinstance(item, float) for item in self.learning_rate):
            raise ValueError("All entries in learning_rate must be floats.")
        
        # Ensure all entries in epochs are integers
        if not all(isinstance(item, int) for item in self.epochs):
            raise ValueError("All entries in epochs must be integers.")
        
        #Ensure that width is a list of integers 
        if not all(isinstance(item, int) for item in self.width):
            raise ValueError("All entries in width must be integers.")  

class neuralFit:
    def __init__(self,networkParameters:networkParameters):
        self.lambda_s=networkParameters.lambda_s
        self.lambda_l2=networkParameters.lambda_l2
        self.epochs=networkParameters.epochs
        self.learning_rate=networkParameters.learning_rate
        self.width=networkParameters.width
        self.depth=len(self.width)
        self.errorWeighting=networkParameters.errorWeighting
        self.networkStructure=networkParameters.networkStructure


    def initKernel(self,extractedQuantity:str,finiteT_kernel:bool,Nt:int,x:np.ndarray,omega:np.ndarray):
        if extractedQuantity=="RhoOverOmega" and finiteT_kernel:
            kernel=KL_kernel_Omega(KL_kernel_Position_FiniteT,x,omega,args=(1/Nt,))
        elif extractedQuantity=="RhoOverOmega" and finiteT_kernel==False:
            kernel=KL_kernel_Omega(KL_kernel_Position_Vacuum,x,omega)
        elif extractedQuantity=="Rho" and finiteT_kernel:
            kernel=KL_kernel_Position_FiniteT(x,omega,1/Nt)
        elif extractedQuantity=="Rho" and finiteT_kernel==False:
            kernel=KL_kernel_Position_Vacuum(x,omega)
        else:
            raise ValueError("Invalid choice spectral function target")
        return kernel

    def fitCorrelator(self, x: np.ndarray, error: np.ndarray, correlator: np.ndarray, finiteT_kernel: bool, Nt: int, omega: np.ndarray, extractedQuantity: str = "RhoOverOmega", verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:

        kernel = self.initKernel(extractedQuantity, finiteT_kernel, Nt, x, omega)
        del_omega = omega[1] - omega[0]
        errorWeight = error if self.errorWeighting else np.ones(len(x))
        constant_input = tf.constant([[1.0]], dtype=tf.float32)  # NN

        if self.networkStructure == "SpectralNN":
            model = SpectralNN(num_output_nodes=len(omega), width=self.width)
        else:
            raise ValueError("Invalid choice of network")

        target_output = correlator
        lossCalc = LossCalculator(
            model=model,
            y_true=target_output,
            kernel=kernel,
            delomega=del_omega,
            x=constant_input,
            std=errorWeight,
            lambda_s_func=lambda x: self.lambda_s[0],
            lambda_l2_func=lambda x: self.lambda_l2[0]
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate[0])
        trainer = networkTrainer(model, optimizer, lossCalc)
        total_loss_history = []
        loss_history = []
        for lambda_s, lambda_l2, learning_rate, epochs in zip(self.lambda_s, self.lambda_l2, self.learning_rate, self.epochs):
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            lossCalc.lambda_s_func = lambda x: lambda_s
            lossCalc.lambda_l2_func = lambda x: lambda_l2
            trainer.optimizer = optimizer
            total_loss_history_tmp, loss_history_tmp = trainer.train(epochs, verbose=verbose)
            total_loss_history.extend(total_loss_history_tmp)
            loss_history.extend(loss_history_tmp)
            if verbose:
                print("-" * 40)
        spectralFunction = model(constant_input)
        return np.squeeze(spectralFunction), np.squeeze(loss_history)
   
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
    
    def get_width(self) -> List[int]:
        width = self.params["width"]
        return [width] if isinstance(width, int) else width
    
    def getNetworkParams(self) -> networkParameters:
        return networkParameters(
            lambda_s=self.params["lambda_s"],
            lambda_l2=self.params["lambda_l2"],
            epochs=self.params["epochs"],
            learning_rate=self.params["learning_rate"],
            width=self.get_width(),
            errorWeighting=self.params["errorWeighting"],
            networkStructure=self.params["networkStructure"]
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
        self.net_params = self.parameterHandler.getNetworkParams()
        self.fitter = neuralFit(self.net_params)
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

    def run_single_fit(self, fittedQuantity, messageString ,results: List[np.ndarray], loss_histories: List[np.ndarray]) -> None:
        start_time = time.time()
        print("=" * 40)
        print(messageString)
        print("=" * 40)
        sf, loss_history = self.fitter.fitCorrelator(
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
            print(f"Training time: {time.time() - start_time:.2f} seconds")
        results.append(sf)
        loss_histories.append(loss_history)

    def run_fits(self) -> Tuple[np.ndarray, np.ndarray]:
        results = []
        loss_histories = []
        if self.correlators.ndim == 1:
            self.correlators = np.array([self.correlators])
        else:
            self.correlators = self.correlators.T
        n_correlators = self.correlators.shape[0]
        if self.multiFit:
            self.run_single_fit(self.correlators, f"Multifitting {n_correlators} correlators", results, loss_histories)            
        else:
            self.run_single_fit(self.mean, "Fitting mean correlator", results, loss_histories)
            for i, corr in enumerate(self.correlators):
                self.run_single_fit(corr, f"Fitting correlator sample {i + 1}/{n_correlators}", results, loss_histories)
        return np.array(results), np.array(loss_histories)

    def calculate_mean_error(self, mean: np.ndarray, samples: np.ndarray, errormethod: str = "jackknife") -> np.ndarray:
        N = len(samples)
        fac = N - 1 if errormethod == "jackknife" else 1
        if errormethod not in ["jackknife", "bootstrap"]:
            raise ValueError("Invalid choice of error estimation method")
        return np.sqrt(fac / N * np.sum((samples - mean) ** 2, axis=0))

    def save_results(self, mean: np.ndarray, error: np.ndarray, samples: np.ndarray, loss_history: np.ndarray, extractedQuantity: str = "RhoOverOmega") -> None:
        header = "Omega " + self.extractedQuantity + "_mean"
        if samples is not None and error is not None:
            header += f" {self.extractedQuantity}_error"
            for i in range(len(samples)):
                header += f" {self.extractedQuantity}_sample_{i}"
            writeData = np.column_stack((self.omega, mean, error, samples.T))
        else:
            writeData = np.column_stack((self.omega, mean))
        np.savetxt(os.path.join(self.outputDir, self.outputFile + ".uns"), writeData, header=header)
        if self.parameterHandler.get_params()["saveParams"]:
            self.save_params(self.parameterHandler.get_params(), os.path.join(self.outputDir, self.outputFile + ".params"))
        if self.parameterHandler.get_params()["saveLossHistory"]:
            self.save_loss_history(loss_history, os.path.join(self.outputDir, self.outputFile + ".loss.dat.uns"))

    def save_loss_history(self, loss_history: np.ndarray, outputFile: str) -> None:
        header = "mean_total_loss mean_main_loss mean_smoothness_loss mean_l2_loss"
        for i in range(len(loss_history[1:])):
            header += f" sample_{i}_total_loss sample_{i}_main_loss sample_{i}_smoothness_loss sample_{i}_l2_loss"
        reshaped_loss_history = loss_history.transpose(1, 0, 2).reshape(loss_history.shape[1], -1)
        np.savetxt(outputFile, reshaped_loss_history, header=header)

    def save_params(self, params: dict, outputFile: str) -> None:
        with open(outputFile + '.json', 'w') as f:
            json.dump(params, f, indent=4)
    
def initializeArgumentParser(paramsDefaultDict: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuralFit",
        description="Fit spectral functions to provided correlators using a neural network."
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
    parameterHandler.load_params(args.config,args)

    if parameterHandler.get_verbose():
        print("*"*40)
        print("Running fits with the following parameters:")
        pprint.pprint(parameterHandler.get_params())

    fitRunner = FitRunner(parameterHandler)
    results,loss_histories = fitRunner.run_fits()
    mean = results[0]
    if len(results)>1:
        samples = results[1:]
        error = fitRunner.calculate_mean_error(samples,mean,parameterHandler.get_params()["errormethod"])
    else:
        samples = None
        error = None
    fitRunner.save_results(mean,error,samples,loss_histories)


paramsDefaultDict = {
    #NetworkParams
    "lambda_s": [1e-5],
    "lambda_l2": [1e-8],
    "epochs": [100],
    "learning_rate": [1e-4],
    "width": [32,32,32],
    "errorWeighting": True,
    "networkStructure": "SpectralNN",
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
    "saveLossHistory": False,
    "verbose": False,
    "outputFile": "",
    "outputDir": ''

}



if __name__ == "__main__":
    main(paramsDefaultDict)
