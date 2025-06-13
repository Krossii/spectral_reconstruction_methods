from scipy import integrate
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.optimize import check_grad
from dataclasses import dataclass, field

import numpy as np
import json
import argparse
import time
import pprint
import os
import itertools
from typing import List, Tuple, Callable

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
        if Omega[0] == 0:
            Omega[0] = 1e-8

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
    KL = KL.astype(dtype=np.float32)  # Cast KL to float32
    rhoi = rhoi.astype(dtype=np.float32)  # Cast rhoi to float32
    delomega = delomega.astype(dtype=np.float32)  # Cast delomega to float32

    rhoi = np.reshape(rhoi, [-1, 1])

    # Perform matrix multiplication
    dis = np.matmul(KL, rhoi)
    dis = np.squeeze(dis, axis=-1)  # Remove the singleton dimension to get [25]
    dis = dis * delomega  # Multiply by delomega
    return dis

def get_default_model(w: np.ndarray, defmod: str):
    def_model = np.ones(len(w))
    if defmod == "constant":
        return def_model
    if defmod == "quadratic":
        def_model = w**2 #theoretically there should be a factor here but I omit it for now
        return def_model
    if defmod == "exact":
        data = np.loadtxt("/home/Christian/Desktop/mock-data-main/BW/exact_spectral_function_BW.dat")
        def_model = data[:, 1]
        return def_model

class mem:
    def __init__(
            self, omega: np.ndarray, alpha: np.ndarray, def_model: np.ndarray,
            cov_mat_inv: np.ndarray, Nt: int):
        self.alpha = alpha
        self.def_model = def_model
        self.w = omega
        self.N_t = Nt
        self.cov_mat_inv = cov_mat_inv
        self.delomega = self.w[1] - self.w[0]

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

    def partialL_partialG(self, G_rho: np.ndarray, corr: np.ndarray)-> np.ndarray:
        return np.dot(self.cov_mat_inv, (G_rho - corr))

    def step1(self, corr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        V, xi, U = np.linalg.svd(kernel, full_matrices=False)
        U= U.T
        xi = np.array(list(itertools.takewhile(lambda x: x > 1e-10, xi)))
        s = xi.size
        V = V[:,:s]
        U = U[:,:s]
        print("Singular space dimension:", s, "down from:", min(self.N_t, len(self.w)))

        VXi = np.dot(V, np.diag(xi))

        M = np.dot(VXi.T, np.dot(self.cov_mat_inv, VXi))
        rho_min = np.zeros((len(self.alpha), len(self.w)))

        for i in range(len(self.alpha)):
            rho_min[i][:] = self.minimizer(corr, VXi, M, U, self.alpha[i])
        return rho_min

    def minimizer(self,
                corr: np.ndarray, VXi: np.ndarray, M: np.ndarray, U: np.ndarray, al: float) -> np.ndarray:
        N_s = M.shape[0]
        u = np.zeros((N_s))
        u[0] = 1
        rho = self.def_model *np.exp(np.dot(U, u))

        stoppingcondition = 100
        mu = 0
        solveccounter = 0
        while stoppingcondition >= 1e-5:
            #mucounter = 0
            """while normcondition > 0.2*np.sum(self.def_model):
                if mucounter == 0:
                    mu = 0
                elif mucounter == 1:
                    mu = al/10000
                else:
                    mu *= 10"""
            G_rho = np.dot(VXi, np.dot(U.T, rho))
            g = np.dot(VXi.T, self.partialL_partialG(G_rho, corr))
            T = np.dot(U.T, np.dot(np.diag(rho), U))
            Gamma, P = np.linalg.eigh(T)
            Gamma_safe = np.maximum(Gamma, 1e-4)
            Psqgamma = np.dot(P, np.diag(np.sqrt(Gamma_safe)))
            B = np.dot(Psqgamma.T, np.dot(M, Psqgamma))
            Lambda, R = np.linalg.eigh(B)
            Lambda_safe = np.maximum(Lambda, 1e-4)
            Yinv = np.dot(R.T, np.dot(np.diag(np.sqrt(Gamma_safe)), P.T))
            Yinv_du = -np.dot(Yinv, al*u + g) / (al + mu + Lambda_safe)
            du = (-al * u - g - np.dot(M, np.dot(Yinv.T, Yinv_du))) / (al+mu)
                #mucounter += 1

            u += du

            stoppingcondition = 2*(np.linalg.norm(-al*np.dot(T, u) - np.dot(T, g)))**2/((np.linalg.norm(-al*np.dot(T, u)) + np.linalg.norm(np.dot(T, g)))**2)

            dot_Uu = np.dot(U,u)
            dot_Uu = np.clip(dot_Uu, -50, 50) #safe range for inf/nan values
            rho = self.def_model * np.exp(dot_Uu)

            solveccounter += 1
            if solveccounter % 50000 == 0:
                print("‖g‖:", np.linalg.norm(g), "‖u‖:", np.linalg.norm(u), "‖du‖:", np.linalg.norm(du))
                print(stoppingcondition)

            if solveccounter > 100000000:
                break
        if stoppingcondition < 1e-5:
            print("Found solution vector after iteration", solveccounter)
        else:
            print("No solution found in reasonable time.")
        return rho

    def step2(self, rho: np.ndarray, corr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        P_alphaHM = 1/self.alpha
        S, L, exp, lam_mat = np.zeros(len(self.alpha)), np.zeros(len(self.alpha)), np.zeros(len(self.alpha)), np.zeros((len(self.w), len(self.w)))
        Hess_mat = np.transpose(kernel) @ np.linalg.inv(self.cov_mat_inv) @ kernel
        for i in range(len(self.alpha)):
            for j in range(len(self.w)):
                for k in range(len(self.w)):
                    lam_mat[j][k] = self.delomega*np.sqrt(rho[i][j]) * Hess_mat[j][k] * np.sqrt(rho[i][k])
            eigval, eigvec = np.linalg.eigh(lam_mat)
            self.def_model += 0.000001
            div = rho[i][:]/self.def_model
            div[0] = 1
            S[i] = self.delomega*np.sum(rho[i][:] - self.def_model - rho[i][:]*np.log(div))
            self.def_model -= 0.000001
            G = Di(kernel, rho[i][:], self.delomega)
            L[i] = 1/2 * np.sum(sum((corr - G) @ self.cov_mat_inv[:][j] * (corr[j] - G[j]) for j in range(self.N_t)))
            exp[i] = np.exp(1/2 * np.sum(np.log(self.alpha[i]/(self.alpha[i] + eigval[:])))+ self.alpha[i] * S[i] - L[i])
        P_alphaDHM = P_alphaHM * exp
        alpha_ind = []
        for i in range(len(self.alpha)):
            if P_alphaDHM[i] >= 10**(-1) * P_alphaDHM.max():
                alpha_ind.append(i)
        alpha_int = np.zeros(len(alpha_ind))
        rho_red = np.zeros((len(alpha_ind), len(self.w)))
        P_alphaDHM_red = np.zeros(len(alpha_ind))
        for i in range(len(alpha_ind)):
            alpha_int[i] = self.alpha[alpha_ind[i]]
            P_alphaDHM_red[i] = P_alphaDHM[alpha_ind[i]]
            rho_red[i][:] = rho[alpha_ind[i]][:]
        normalizing_fac = integrate.trapezoid(P_alphaDHM_red, alpha_int)
        print(alpha_int)
        print(P_alphaDHM)
        rho_out = np.zeros(len(self.w))
        if normalizing_fac < 1e-6: 
            P_alphaDHM_red = 1
            normalizing_fac = 1
            return np.mean(rho, axis=0), P_alphaDHM_red, alpha_int, Hess_mat
        for i in range(len(self.w)):
            rho_out[i] = integrate.trapezoid(np.transpose(rho_red)[i][:] * P_alphaDHM_red/normalizing_fac, alpha_int)
        return rho_out, P_alphaDHM_red/normalizing_fac, alpha_int, Hess_mat

    def step3(
            self, rho_alpha: np.ndarray, Hess_L: np.ndarray, w_region: np.ndarray, 
            P_alphaDHM: np.ndarray, alpharegion: np.ndarray) -> np.ndarray:
        Hess_Q = np.zeros((len(self.alpha), len(w_region), len(w_region)))
        Hess_Q_inv = np.zeros((len(self.alpha), len(w_region), len(w_region)))
        Hess_S = np.zeros((len(self.alpha), len(w_region), len(w_region)))
        rho_var = np.zeros((len(self.alpha)))
        rho_var_temp = np.zeros((len(self.alpha), len(w_region)))
        integrand = np.zeros((len(self.alpha))) 
        for i in range(len(self.w)):
            if w_region[0] == self.w[i]:
                w_start_index = i
                break
        norm = (w_region[len(w_region)-1] - w_region[0])**2
        for i in range(len(self.alpha)):
            for k in range(len(w_region)):
                for n in range(len(w_region)):
                    if k == n:
                        if rho_alpha[i][k+w_start_index] == 0:
                            print("A_alpha value error")
                        Hess_S[i][k][n] = - 1/(rho_alpha[i][k+w_start_index] * self.delomega)
                    Hess_Q[i][k][n] = self.alpha[i] * Hess_S[i][k][n] - Hess_L[k][n]
            Hess_Q_inv[i][:][:] = np.linalg.inv(Hess_Q[i][:][:])
        for i in range(len(self.alpha)):
            for j in range(len(w_region)):
                rho_var_temp[i][j] = integrate.trapezoid(Hess_Q_inv[i][j][:], w_region)
            rho_var[i] = integrate.trapezoid(rho_var_temp[i][:], w_region) 
            rho_var[i] *= -1/norm
            integrand[i] = rho_var[i] * P_alphaDHM[i]
        rho_out_var = integrate.trapezoid(integrand, self.alpha)
        return rho_out_var

    def fitCorrelator(
        self, x: np.ndarray, correlator: np.ndarray, finiteT_kernel: bool, 
        Nt: int, omega: np.ndarray, extractedQuantity: str = "RhoOverOmega", 
        verbose: bool = True
        ) -> Tuple[np.ndarray, np.ndarray]:

        kernel = self.initKernel(extractedQuantity, finiteT_kernel, Nt, x, omega)
        if verbose:
            print("*"*40)
            print("Starting minimization using svd")
        rho_min = self.step1(correlator, kernel)
        if verbose:
            print("*"*40)
            print("Starting calculation of probability distribution")
        rho_out, Prob_dist, alpha_reg, Hess_L = self.step2(rho_min, correlator, kernel)
        if verbose:
            print("*"*40)
            print("Starting error evaluation")
        if np.any(np.isnan(rho_min)):
            print("*"*40)
            print("Nan value in rho_out detected. Aborting error evaluation.")
            error = np.zeros(len(omega))
        if Prob_dist == 1:
            print("*"*40)
            print("Probability distribution empty. Aborting error evaluation.")
            error = np.zeros(len(omega))
        else:
            error_region = self.w[:50]
            error = self.step3(rho_min, Hess_L, error_region, Prob_dist, alpha_reg)
        return rho_out, error

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
        self.alpha = np.linspace(
            self.parameterHandler.get_params()["alpha_min"],
            self.parameterHandler.get_params()["alpha_max"],
            self.parameterHandler.get_params()["alpha_points"]
        )
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
        self.default_model = get_default_model(self.omega, self.parameterHandler.get_params()["default_model"])
        self.finiteT_kernel = self.parameterHandler.get_params()["FiniteT_kernel"]
        self.verbose = self.parameterHandler.get_verbose()
        self.multiFit = self.parameterHandler.get_params()["multiFit"]
        self.extractedQuantity = self.parameterHandler.get_extractedQuantity()
        self.Nt = self.parameterHandler.get_params()["Nt"] or len(self.x)
        self.outputDir = os.path.abspath(self.parameterHandler.get_params()["outputDir"])
        self.outputFile = self.parameterHandler.get_params()["outputFile"] or f"{self.extractedQuantity}_{os.path.basename(self.parameterHandler.get_correlator_file())}"
        self.fitter = mem(
            self.omega, self.alpha, self.default_model, 
            np.linalg.inv(np.diag(self.error**2)), self.Nt)

    def extractColumns(self, file: str, x_col: int, mean_col: int, error_col: int, correlator_cols: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data = np.loadtxt(file)
        x = data[:, x_col]
        mean = data[:, mean_col]
        error = data[:, error_col]
        correlator = data[:, correlator_cols]
        return x, mean, error, correlator    

    def run_single_fit(
            self, fittedQuantity, messageString ,results: List[np.ndarray] ,
            errors: List[np.ndarray]) -> None:
        start_time = time.time()
        print("=" * 40)
        print(messageString)
        print("=" * 40)
        rho, error = self.fitter.fitCorrelator(
            self.x,
            fittedQuantity,
            self.finiteT_kernel,
            self.Nt,
            self.omega,
            self.extractedQuantity,
            self.verbose
        )
        if self.verbose:
            print("-" * 40)
            print(f"Fitting time: {time.time() - start_time:.2f} seconds")
        results.append(rho)
        errors.append(error)

    
    def run_fits(self) -> Tuple[np.ndarray, np.ndarray]:
        results = []
        errors = []
        if self.correlators.ndim == 1:
            self.correlators = np.array([self.correlators])
        else:
            self.correlators = self.correlators.T
        n_correlators = self.correlators.shape[0]
        if self.multiFit:
            self.run_single_fit(self.correlators, f"Multifitting {n_correlators} correlators", results, errors)
        else:
            self.run_single_fit(self.mean, "Fitting mean correlator", results, errors)
            for i, corr in enumerate(self.correlators):
                self.run_single_fit(corr, f"Fitting correlator sample {i + 1}/{n_correlators}", results, errors)
        return np.array(results), np.array(errors)

    def calculate_mean_error(self, mean: np.ndarray, samples: np.ndarray, errormethod: str = "jackknife") -> np.ndarray:
        N = len(samples)
        fac = N - 1 if errormethod == "jackknife" else 1
        if errormethod not in ["jackknife", "bootstrap"]:
            raise ValueError("Invalid choice of error estimation method")
        return np.sqrt(fac / N * np.sum((samples - mean) ** 2, axis=0))

    def save_results(
            self, mean: np.ndarray, error: np.ndarray, samples: np.ndarray, 
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
        np.savetxt(os.path.join(self.outputDir, self.outputFile), writeData, header=header)
        if self.parameterHandler.get_params()["saveParams"]:
            self.save_params(self.parameterHandler.get_params(), os.path.join(self.outputDir, self.outputFile + ".params"))

    def save_params(self, params: dict, outputFile: str) -> None:
        with open(outputFile + '.json', 'w') as f:
            json.dump(params, f, indent=4)


def initializeArgumentParser(paramsDefaultDict: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mem",
        description="Fit spectral functions toprovided correlator with the Maximum Entropy Method."
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
    results, error = fitRunner.run_fits()
    mean = results[0]
    if len(results)>1:
        samples = results[1:]
        error = fitRunner.calculate_mean_error(samples,mean,parameterHandler.get_params()["errormethod"])
    else:
        samples = None
        error = None
    fitRunner.save_results(mean,error,samples)



    
paramsDefaultDict = {
    #choice of SupervisedNN, UnsupervisedNN, Gaussian, MEM
    "Method": "MEM",
    #MEM specific; default model: choose from quadratic, constant currently (ai to be implemented)
    #should not give 0 as a minimum for alpha, just a small value
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
    "outputDir": ""

}



if __name__ == "__main__":
    main(paramsDefaultDict)
