import pickle
import matplotlib.pyplot as plt

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
    KL = np.asarray(KL, dtype=np.float32)  # Cast KL to float32
    rhoi = np.asarray(rhoi, dtype=np.float32)  # Cast rhoi to float32
    delomega = np.asarray(delomega, dtype=np.float32)  # Cast delomega to float32
    
    # Ensure rhoi has the correct shape [500,1] for matrix multiplication
    rhoi = np.reshape(rhoi, [-1, 1])  # Reshape to [500, 1]

    # Perform matrix multiplication
    dis = np.matmul(KL, rhoi)  # Shape will be [25, 1]
    dis = np.squeeze(dis, axis=-1)  # Remove the singleton dimension to get [25]
    dis = dis * delomega  # Multiply by delomega
    return dis

class spectral_functions:
    def __init__(self, w: np.ndarray, extractedQuantity: str):
        self.w = w
        self.extractedQuantity = extractedQuantity

    def breit_wigner(
            self, w: np.ndarray, a: int, m: int, g: int
            ) -> np.ndarray:
        if self.extractedQuantity=="RhoOverOmega":
            return 4*a*g/((m**2 + g**2 - w**2)**2 + 4 * g**2 * w**2)
        elif self.extractedQuantity=="Rho":
            return 4*a*g*w/((m**2 + g**2 - w**2)**2 + 4 * g**2 * w**2)
    
    def multiple_breit_wigner(
            self, w: np.ndarray, a: np.ndarray, m: np.ndarray, g: np.ndarray
            ) -> np.ndarray:
        rho = np.zeros(len(w))
        for i in range(len(a)):
            rho_temp = 4*a[i]*g[i]*w/((m[i]**2 + g[i]**2 - w**2)**2 + 4 * g[i]**2 * w**2)
            rho += rho_temp
        if self.extractedQuantity=="RhoOverOmega":
            return np.divide(rho,w, out=np.zeros_like(rho,dtype=float), where=w!=0)
        elif self.extractedQuantity=="Rho":
            return rho
    
    def step_function(
            self, w: np.ndarray, threshhold: int, height: float
        ) -> np.ndarray:
        rho = np.zeros(len(w))
        for i in range(len(w)):
            if w[i] >= threshhold:
                rho[i] += height
        if self.extractedQuantity=="RhoOverOmega":
            return np.divide(rho,w, out=np.zeros_like(rho,dtype=float), where=w!=0)
        elif self.extractedQuantity=="Rho":
            return rho
    
    def sharp_gaussian(
            self, w: np.ndarray, mu: int, sigma: float
        ) -> np.ndarray:
        rho = w**2 * np.exp(-1/2 *(w-mu)**2/(sigma**2))
        if self.extractedQuantity=="RhoOverOmega":
            return np.divide(rho,w, out=np.zeros_like(rho,dtype=float), where=w!=0)
        elif self.extractedQuantity=="Rho":
            return rho
    
    def non_zero_gaussian_at_origin(
            self, w: np.ndarray, mu: int, sigma: float
        ) -> np.ndarray:
        rho = 1/np.sqrt(2 *np.pi * sigma) *1/1.5* np.exp(-1/2 * (w-mu)**2/(sigma**2))
        if self.extractedQuantity=="RhoOverOmega":
            return np.divide(rho,w, out=np.zeros_like(rho,dtype=float), where=w!=0)
        elif self.extractedQuantity=="Rho":
            return rho
    
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
    
    def get_create_data(self) -> bool:
        return self.params["create_data"]

class correlators:
    def __init__(self, parameterHandler: ParameterHandler):
        self.parameterHandler = parameterHandler
        
    def initKernel(
            self,extractedQuantity:str,finiteT_kernel:bool,
            Nt:int,x:np.ndarray,omega:np.ndarray
            ):
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
    
    def correlator(
        self, w: np.ndarray, tau: np.ndarray, rho: np.ndarray
        ) -> np.ndarray:
        
        kernel = self.initKernel(
            self.parameterHandler.get_params()["extractedQuantity"],
            self.parameterHandler.get_params()["FiniteT_kernel"],
            len(tau), tau, w
            )
        del_omega = w[1]- w[0]
        corr = Di(kernel, rho, del_omega)
        return corr

    def noise(
        self, corr: np.ndarray
        ):
        #this error handling is problematic somehow
        if np.all(corr):
            noise = np.random.normal(
                np.zeros(len(corr)), self.parameterHandler.get_params()["data_noise"]/corr, (len(corr))
                )
        else:
            print("Zero in correlator found")
            noise = np.zeros(len(corr))
            for i in range(len(corr)):
                if corr[i] == 0:
                    noise[i] = 1
                    print("Set noise for this value of the correlator to 1")
                else:
                    noise[i] = np.random.normal(0, self.parameterHandler.get_params()["data_noise"]/corr[i])
        return noise

class create_datset:
    def __init__(
            self, w: np.ndarray, tau: np.ndarray, specfuncs: spectral_functions,
            corrs: correlators, parameterHandler: ParameterHandler
            ) -> None:
        self.breit_wigner = specfuncs.breit_wigner
        self.mbreit_wigner = specfuncs.multiple_breit_wigner
        self.non_zero = specfuncs.non_zero_gaussian_at_origin
        self.step = specfuncs.step_function
        self.peak = specfuncs.sharp_gaussian
        self.get_corr = corrs.correlator
        self.noise = corrs.noise
        self.w = w
        self.tau = tau
        self.parameterHandler = parameterHandler

    def breit_wigners(self):
        one_dat = []
        A = np.linspace(0.1, 0.7, 20)
        M = np.linspace(0.5, 3.0, 20)
        G = np.linspace(0.3, 0.8, 20)
        if self.parameterHandler.get_verbose:
            print("*"*40)
            print("Creating single peaked Breit Wigner.")
        for i in range(len(A)):
            for j in range(len(M)):
                for k in range(len(G)):
                    rho = self.breit_wigner(self.w, A[i], M[j], G[k])
                    normalizing_fac = np.trapezoid(rho, self.w)
                    normed_rho = rho/normalizing_fac
                    if np.max(normed_rho) >= 5:
                        print(A[i], M[j], G[k])
                    corr = self.get_corr(self.w, self.tau, normed_rho)
                    noise = self.noise(corr)
                    if np.any(np.isnan(normed_rho)) or np.any(np.isnan(corr)) or np.any(np.isnan(noise)):
                        print("Nan value in single peaked BW")
                    one_dat.append({
                        'fct': normed_rho,
                        'corr': corr,
                        'noise': noise,
                    })
        
        mult_dat = []
        A, M, G = [],[],[]
        N_params = 1
        for i in range(N_params):
            for j in range(N_params):
                A.append([0.1 + i*0.7/N_params, 0.1 + j*0.7/N_params])
                M.append([0.5 + i*3.0/N_params, 0.5 + j*3.0/N_params])
                G.append([0.3 + i * 0.8/N_params, 0.3 + j*0.8/N_params])
        if self.parameterHandler.get_verbose:
            print("*"*40)
            print("Creating double peaked Breit Wigner.")

        for i in range(N_params**2):
            for j in range(N_params**2):
                for k in range(N_params**2):
                    rho = self.mbreit_wigner(self.w, A[i][:], M[j][:], G[k][:])
                    normalizing_fac = np.trapezoid(rho, self.w)
                    normed_rho = rho/normalizing_fac                    
                    if np.max(normed_rho) >= 5:
                        print(A[i][:], M[j][:], G[k][:])
                    corr = self.get_corr(self.w, self.tau, normed_rho)
                    noise = self.noise(corr)
                    if np.any(np.isnan(normed_rho)) or np.any(np.isnan(corr)) or np.any(np.isnan(noise)):
                        print("Nan value in double peaked BW")
                    mult_dat.append({
                        'fct': normed_rho,
                        'corr': corr,
                        'noise': noise,
                    })
        return one_dat, mult_dat

    def non_zeros(self):
        dat = []
        mu = np.linspace(0, 0.5, 500)
        sigma = np.linspace(0.1, 0.5, 100)
        if self.parameterHandler.get_verbose:
            print("*"*40)
            print("Creating Gaussian non-zero at origin.")
        for i in range(len(mu)):
            for j in range(len(sigma)):                
                rho = self.non_zero(self.w, mu[i], sigma[j])
                normalizing_fac = np.trapezoid(rho, self.w)
                normed_rho = rho/normalizing_fac
                corr = self.get_corr(self.w, self.tau, normed_rho)
                noise = self.noise(corr)
                if np.any(np.isnan(normed_rho)) or np.any(np.isnan(corr)) or np.any(np.isnan(noise)):
                    print("Nan value in gaussian non zero")
                dat.append({
                    'fct': normed_rho,
                    'corr': corr,
                    'noise': noise,
                })
        return dat

    def steps(self):
        dat = []
        t = np.linspace(self.w[-1]/15, self.w[-1]/2, 500)
        h = np.linspace(0.1, 10, 100)
        if self.parameterHandler.get_verbose:
            print("*"*40)
            print("Creating step function.")
        for i in range(len(t)):
            for j in range(len(h)):
                rho = self.step(self.w, t[i], h[j])
                normalizing_fac = np.trapezoid(rho, self.w)
                normed_rho = rho/normalizing_fac
                corr = self.get_corr(self.w, self.tau, normed_rho)
                noise = self.noise(corr)
                if np.any(np.isnan(normed_rho)) or np.any(np.isnan(corr)) or np.any(np.isnan(noise)):
                    print("Nan value in step function")
                dat.append({
                    'fct': normed_rho,
                    'corr': corr,
                    'noise': noise,
                })
        return dat

    def peaks(self):
        dat = []
        mu = self.w
        sigma = np.linspace(0.001, 0.05, 100)
        if self.parameterHandler.get_verbose:
            print("*"*40)
            print("Creating sharp Gaussian peaks.")
        for i in range(len(mu)):
            for j in range(len(sigma)):
                rho = self.peak(self.w, mu[i], sigma[j])
                normalizing_fac = np.trapezoid(rho, self.w)
                normed_rho = rho/normalizing_fac
                corr = self.get_corr(self.w, self.tau, normed_rho)
                noise = self.noise(corr)
                if np.any(np.isnan(normed_rho)) or np.any(np.isnan(corr)) or np.any(np.isnan(noise)):
                    print("Nan value in peak function")                
                dat.append({
                    'fct': normed_rho,
                    'corr': corr,
                    'noise': noise,
                })
        return dat

    def dataset(self):
        if self.parameterHandler.get_verbose:
            print("*"*40)
            print("Creating the datasets.")
        bw, mbw = self.breit_wigners()
        #nonz = self.non_zeros()
        #steps = self.steps()
        #peaks = self.peaks()

        if self.parameterHandler.get_verbose:
            print("*"*40)
            print("Splitting into test and validation sets.")
        full_set = bw #+ mbw + nonz + peaks + steps
        print(len(full_set))
        train_dat = []
        val_dat = []
        integers_split = np.random.randint(
            len(full_set), size = int((len(full_set))/5))
        for i in range(len(full_set)):
            if i in integers_split: val_dat.append(full_set[i])
            else: train_dat.append(full_set[i])
        train_file = f'train_dat_{self.parameterHandler.get_params()["Nt"]}_{len(self.w)}_{self.parameterHandler.get_params()["extractedQuantity"]}_{self.parameterHandler.get_params()["data_noise"]}_{self.parameterHandler.get_params()["Method"]}.dat'
        val_file = f'val_dat_{self.parameterHandler.get_params()["Nt"]}_{len(self.w)}_{self.parameterHandler.get_params()["extractedQuantity"]}_{self.parameterHandler.get_params()["data_noise"]}_{self.parameterHandler.get_params()["Method"]}.dat'
        if self.parameterHandler.get_params()["cluster"]:
            cluster_path = os.path.join(ParameterHandler.get_params()["clusterpath"], "spf_datasets/")
            with open(os.path.join(cluster_path, train_file), 'wb') as f:
                np.save(f, train_dat)
            with open(os.path.join(cluster_path, val_file), 'wb') as f:
                np.save(f, val_dat)
        else:
            with open(os.path.join("spf_datasets/", train_file), 'wb') as f:
                np.save(f, train_dat)
            with open(os.path.join("spf_datasets/", val_file), 'wb') as f:
                np.save(f, val_dat)
        
        if self.parameterHandler.get_verbose:
            print("*"*40)
            print("Files created at:", train_file)

class create_Kades_datset:
    def __init__(self, w: np.ndarray, tau: np.ndarray, specfuncs: spectral_functions,
                 corrs: correlators, parameterHandler: ParameterHandler) -> None:
        self.breit_wigner = specfuncs.breit_wigner
        self.mbreit_wigner = specfuncs.multiple_breit_wigner
        self.get_corr = corrs.correlator
        self.noise = corrs.noise
        self.w = w
        self.tau = tau
        self.parameterHandler = parameterHandler


    def breit_wigners(self):
        one_dat = []
        A = np.linspace(0.1,1,30)
        M = np.linspace(0.5,3,30)
        G = np.linspace(0.1,0.4,30)
        if self.parameterHandler.get_verbose:
            print("*"*40)
            print("Creating single peaked Breit Wigner.")
        for i in range(len(A)):
            for j in range(len(M)):
                for k in range(len(G)):
                    rho = self.breit_wigner(self.w, A[i], M[j], G[k])
                    normalizing_fac = np.trapezoid(rho, self.w)
                    normed_rho = rho/normalizing_fac
                    corr = self.get_corr(self.w, self.tau, normed_rho)
                    noise = self.noise(corr)
                    if np.any(np.isnan(normed_rho)) or np.any(np.isnan(corr)) or np.any(np.isnan(noise)):
                        print("Nan value in single peaked BW")
                    one_dat.append({
                        'fct': normed_rho,
                        'corr': corr,
                        'noise': noise,
                    })

        mult_dat = []
        A, M, G = [],[],[]
        for i in range(7):
            for j in range(7):
                A.append([0.1 + i/7, 0.1 + j/7])
                M.append([0.5 + i*3/7, 0.5 + j*3/7])
                G.append([0.1 + i*0.4/7, 0.1 + j*0.4/7])
        
        if self.parameterHandler.get_verbose:
            print("*"*40)
            print("Creating double peaked Breit Wigner.")

        for i in range(7**2):
            for j in range(7**2):
                for k in range(7**2):
                    rho = self.mbreit_wigner(self.w, A[i][:], M[j][:], G[k][:])
                    normalizing_fac = np.trapezoid(rho, self.w)
                    normed_rho = rho/normalizing_fac
                    corr = self.get_corr(self.w, self.tau, normed_rho)
                    noise = self.noise(corr)
                    if np.any(np.isnan(normed_rho)) or np.any(np.isnan(corr)) or np.any(np.isnan(noise)):
                        print("Nan value in double peaked BW")
                    mult_dat.append({
                        'fct': normed_rho,
                        'corr': corr,
                        'noise': noise,
                    })
        return one_dat, mult_dat

    def dataset(self):
        if self.parameterHandler.get_verbose:
            print("*"*40)
            print("Creating the datasets.")
        bw, mbw = self.breit_wigners()

        if self.parameterHandler.get_verbose:
            print("*"*40)
            print("Splitting into test and validation sets.")
        full_set = bw + mbw
        print(len(full_set))
        train_dat = []
        val_dat = []
        integers_split = np.random.randint(
            len(full_set), size = int((len(full_set))/5))
        for i in range(len(full_set)):
            if i in integers_split: val_dat.append(full_set[i])
            else: train_dat.append(full_set[i])
        if self.parameterHandler.get_params()["cluster"]:
            cluster_path = os.path.join(ParameterHandler.get_params()["clusterpath"], "spf_datasets/")
            train_file = f'train_dat_{self.parameterHandler.get_params()["Nt"]}_{len(self.w)}_{self.parameterHandler.get_params()["extractedQuantity"]}_{self.parameterHandler.get_params()["data_noise"]}_{self.parameterHandler.get_params()["Method"]}.dat'
            val_file = f'val_dat_{self.parameterHandler.get_params()["Nt"]}_{len(self.w)}_{self.parameterHandler.get_params()["extractedQuantity"]}_{self.parameterHandler.get_params()["data_noise"]}_{self.parameterHandler.get_params()["Method"]}.dat'
            with open(os.path.join(cluster_path, train_file), 'wb') as f:
                np.save(f, train_dat)
            with open(os.path.join(cluster_path, val_file), 'wb') as f:
                np.save(f, val_dat)
        else:
            with open(f'spf_datasets/train_dat_{self.parameterHandler.get_params()["Nt"]}_{len(self.w)}_{self.parameterHandler.get_params()["extractedQuantity"]}_{self.parameterHandler.get_params()["data_noise"]}_{self.parameterHandler.get_params()["Method"]}.dat', 'wb') as f:
                np.save(f, train_dat)
            with open(f'spf_datasets/val_dat_{self.parameterHandler.get_params()["Nt"]}_{len(self.w)}_{self.parameterHandler.get_params()["extractedQuantity"]}_{self.parameterHandler.get_params()["data_noise"]}_{self.parameterHandler.get_params()["Method"]}.dat', 'wb') as f:
                np.save(f, val_dat)
        
        if self.parameterHandler.get_verbose:
            print("*"*40)
            print("Files created.")


def initializeArgumentParser(paramsDefaultDict: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="create_data",
        description="Create mock correlators and spectral functions for training of the supervised NN."
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
        print("Creating datasets for the following parameters")
        pprint.pprint(parameterHandler.get_params())

    omega = np.linspace(
        parameterHandler.get_params()["omega_min"],
        parameterHandler.get_params()["omega_max"],
        parameterHandler.get_params()["omega_points"]
    )
    tau = np.arange(parameterHandler.get_params()["Nt"])
    specfuncs = spectral_functions(omega, parameterHandler.get_params()["extractedQuantity"])
    corrs = correlators(parameterHandler)
    if parameterHandler.get_params()["Method"] == "SupervisedNN":
        createdataset = create_datset(
            omega, tau, specfuncs, corrs, parameterHandler
            )
    
        createdataset.dataset()
    if parameterHandler.get_params()["Method"] == "KadesFC":
        createdataset = create_Kades_datset(
            omega, tau, specfuncs, corrs, parameterHandler
            )
        
        createdataset.dataset()



paramsDefaultDict = {
    "Method": "UnsupervisedNN",
    #NetworkParams (Ai specrec)
    "lambda_s": [1e-5],
    "lambda_l2": [1e-8],
    "epochs": [100],
    "learning_rate": [1e-4],
    "errorWeighting": True,
    #Unsupervised specific
    "width": [32,32,32],
    #Supervised specific
    "batch_size": [128],
    "create_data": True,
    "data_noise": 10e-5,
    "trainingFile": "",
    "validationFile": "",
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
    "saveLossHistory": False,
    "verbose": False,
    "outputFile": "",
    "outputDir": "",
    "cluster": False,
    "clusterpath": ""
}



if __name__ == "__main__":
    main(paramsDefaultDict)