import scipy
from scipy import integrate
from scipy.linalg import solve
from scipy.optimize import root
from scipy.optimize import minimize
from scipy.optimize import check_grad
from dataclasses import dataclass, field

import matplotlib.pyplot as plt

from latqcdtools.base.check import ignoreDivideByZero, ignoreInvalidValue, ignoreUnderflow, ignoreOverflow
from latqcdtools.base.speedify import parallel_function_eval
ignoreDivideByZero()
ignoreInvalidValue()
ignoreUnderflow()
ignoreOverflow()


import imageio
import imageio.v2 as imageio

import numpy as np
import json
import argparse
import time
import pprint
import os
import itertools
from typing import List, Tuple, Callable

def trunc(
        values, 
        dec = 0
        ):
    return np.trunc(values * 10**dec) / 10**dec # truncate to dec decimal places - primarily for Hessian in step 2 of mem

def KL_kernel_Momentum(
        Momentum, 
        Omega
        ):
    Momentum = Momentum[:, np.newaxis]  # Reshape Momentum as column to allow broadcasting
    ker = Omega / (Omega**2 + Momentum**2)  # Element-wise division
    return ker / np.pi 

def KL_kernel_Position_Vacuum(
        Position, 
        Omega
        ):
    Position = Position[:, np.newaxis]  # Reshape Position as column to allow broadcasting
    ker = np.exp(-Omega * np.abs(Position)) + np.exp(-Omega*(len(Position)-Position))
    return ker

def KL_kernel_Position_FiniteT(
        Position, 
        Omega,
        T
        ):
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

def KL_kernel_Omega(
        KL,
        x,
        Omega,
        args=[]
        ):
    ret=KL(x, Omega, *args)
    ret[:,Omega==0]=1
    ret=Omega * ret
    # set for all Omega=0 to 1
    if len(args)==0:
        ret[:,Omega==0]=0
    else:
        ret[:,Omega==0]=2*args[0]
    return ret

def Di(
        KL, 
        rhoi, 
        delomega
        ) -> np.ndarray:
    # Ensure both tensors are of the same data type (float32)
    KL = KL.astype(dtype=np.float32)  # Cast KL to float32
    rhoi = rhoi.astype(dtype=np.float32)  # Cast rhoi to float32
    delomega = delomega.astype(dtype=np.float32)  # Cast delomega to float32

    rhoi = np.reshape(rhoi, [-1, 1])

    # Perform matrix multiplication
    dis = np.matmul(KL, rhoi)
    dis = np.squeeze(dis, axis=-1)  # Remove the singleton dimension
    dis = dis * delomega  # Multiply by delomega
    return dis

def get_default_model(
        w: np.ndarray, 
        defmod: str, 
        file: str = ""
        ) -> np.ndarray:
    def_model = np.ones(len(w))
    if defmod == "constant":
        if file != "":
            data = np.loadtxt(file)
            exact = data[:,1]
            m_0 = np.trapezoid(exact, x=w)/np.trapezoid(np.ones(len(exact)), x=w)
            def_model = np.ones(len(exact))
            return def_model*m_0
        else:
            return def_model
    if defmod == "quadratic":
        if file != "":
            data = np.loadtxt(file)
            exact = data[:,1]
            m_0 = np.trapezoid(exact, x=w) / (np.trapezoid(w**2, x=w))
            def_model = m_0* w**2
            return def_model
        else:
            def_model = w**2
            return def_model
    if defmod == "exact" or defmod == "file":
        data = np.loadtxt(file)
        def_model = data[:, 1]
        return def_model + 0.1*w
    raise ValueError("Invalid choice of default model")

class mem:
    def __init__(
            self, 
            omega: np.ndarray, 
            alpha: np.ndarray, 
            def_model: np.ndarray,
            cov_mat_inv: np.ndarray, 
            Nt: int
            ):
        self.alpha = alpha
        self.def_model = def_model
        self.w = omega
        self.N_t = Nt
        self.cov_mat_inv = cov_mat_inv
        self.delomega = self.w[1] - self.w[0]
        self.tau = np.arange(Nt)

    def initKernel(
        self,
        extractedQuantity: str,
        finiteT_kernel: bool,
        Nt: int,
        x: np.ndarray,
        omega: np.ndarray
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

    def step1(
            self, 
            corr: np.ndarray, 
            kernel: np.ndarray
            ) -> np.ndarray:
        """Minimization of Q = alpha S - L for all alpha as per MEM paper step 1"""
        U, xi, Vt = np.linalg.svd(kernel.T, full_matrices=False)
        xi = np.array(list(itertools.takewhile(lambda x: x > 1e-10, xi)))
        s = xi.size
        Vt = Vt[:s,:]
        U = U[:,:s]
        print("Singular space dimension:", s, "down from:", min(self.N_t, len(self.w)))

        VXi = Vt.T @ np.diag(xi)

        M = VXi.T @ self.cov_mat_inv @ VXi
        rho_min = np.zeros((len(self.alpha), len(self.w)))
        S, L, Q = np.zeros(len(self.alpha)),np.zeros(len(self.alpha)),np.zeros(len(self.alpha))
        #u_g = np.zeros((M.shape[0]))
        #u_g[0]=1
        u_g = np.random.rand(M.shape[0]) 

        def getRhoMin(al):
            return self.minimizer(corr, VXi, M, U, al, u_g, kernel)
        

        rho_min_array = parallel_function_eval(getRhoMin, list(range(len(self.alpha))))


        for i in range(len(self.alpha)):
            rho_min_array = self.minimizer(corr, VXi, M, U, self.alpha[i], u_g, kernel)
            rho_min[i][:] = rho_min_array#[i]
            G_rho = Di(kernel, rho_min[i][:]/(2*np.pi), self.delomega)

        plt.figure(1, figsize=(12,4))
        plt.subplot(1,2,1)
        for i in range(len(self.alpha)):
            plt.plot(self.w, rho_min[i][:], label = f"alpha = {self.alpha[i]}")
        plt.plot(self.w, self.def_model, label = "Default model", color = "black", linestyle = "--")
        plt.ylim(-0.1,3)
        plt.subplot(1,2,2)
        plt.scatter(self.tau, corr, label = "Data", marker = "x")
        plt.scatter(self.tau, G_rho, label = "Prediction", marker = "x")
        plt.legend()
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig("mem_alpha_scan.png")
        return rho_min
    
    def minimizer(
        self,
        corr: np.ndarray, 
        VXi: np.ndarray, 
        M: np.ndarray, 
        U: np.ndarray, 
        al: float, 
        u_guess: np.ndarray, 
        kernel: np.ndarray
        ) -> np.ndarray:
        N_s = M.shape[0]
        def_model = self.def_model
            
        def func(b):
            rho = self.def_model *np.exp(U @ b)
            G_rho = Di(kernel, rho/(2*np.pi), self.delomega)
            g = VXi.T @ self.cov_mat_inv @ (G_rho - corr)
            f = -al * b - g
            return f
        
        def jac(b):
            rho = self.def_model *np.exp(U @ b)
            diag_rho_U = np.diag(rho /(2*np.pi)) @ U
            A = (kernel @ diag_rho_U) * self.delomega
            J_nonlinear = VXi.T @ self.cov_mat_inv @ A
            J = -al * np.eye(N_s) - J_nonlinear
            return J

        # --- diagnostics ---
        rho_guess = self.def_model * np.exp(U @ u_guess)
        G_rho_guess = Di(kernel, rho_guess/(2*np.pi), self.delomega)
        diff_guess = G_rho_guess - corr
        g_guess = VXi.T @ self.cov_mat_inv @ diff_guess
        print(f"alpha: {al:.3e}")
        print(f"||G-r|| = {np.linalg.norm(diff_guess):.3e}, ||VXi^T cov_inv (G-r)|| = {np.linalg.norm(g_guess):.3e}")
        # --- end diagnostics ---
        print("Residual norm before solve", np.linalg.norm(func(u_guess)))
        sol = root(func, u_guess, jac=jac, method='lm')
        res_norm = np.linalg.norm(func(sol.x))
        print("Solver message:", sol.message)
        print("Number of function evaluations:", getattr(sol, "nfev", None))
        print("Residual norm after solve:", res_norm)

        u = sol.x
        rho = self.def_model * np.exp(U @ u)
        G_pred = Di(kernel, rho/(2*np.pi), self.delomega)
        S = np.sum(rho - def_model - rho * np.nan_to_num(np.log(rho/def_model), neginf = -1e300))
        L = 0.5 * (corr - G_pred) @ self.cov_mat_inv @ (corr - G_pred)
        print("S, L, Q:", S, L, al*S-L)
        return rho

    def step2(
            self, 
            rho: np.ndarray, 
            corr: np.ndarray, 
            kernel: np.ndarray
            ) -> np.ndarray:
        """Calculation of P[alpha|D,H,M] as per MEM paper step 2"""
        P_alphaHM = 1 # Laplace's rule
        S, L, exp, prefactor = np.zeros(len(self.alpha)), np.zeros(len(self.alpha)), np.zeros(len(self.alpha)), np.zeros(len(self.alpha))
        evs = np.zeros((len(self.alpha), len(self.w)))
        Hess_mat = kernel.T @ self.cov_mat_inv @ kernel
        Hess_mat = 0.5 * (Hess_mat + Hess_mat.T) # ensure symmetry
        #Hess_mat = trunc(Hess_mat, dec=1) # truncate to 1 decimal place to avoid numerical instability

        for i in range(len(self.alpha)):
            rho_vec = np.maximum(rho[i][:].astype(np.float64), 1e-300) # avoid exact zeros for eigenvalue function
            S_mat = np.sqrt(np.diag(rho_vec))
            lam_mat = S_mat @ Hess_mat @ S_mat
            lam_mat = 0.5 * (lam_mat + lam_mat.T) #ensure symmetry

            eigval, eigvec = np.linalg.eigh(lam_mat)

            machine_eps = np.finfo(float).eps
            scale = max(1.0, np.max(np.abs(eigval)))
            tol = lam_mat.shape[0] * machine_eps * scale # relative tolerance
            abs_tol = 1e-12 # absolute floor (relaxable to 1e-10)
            neg_tol = max(tol, abs_tol)

            # clamp tiny negative eigenvlaues to zero: warn for substantial negatives
            small_neg_mask = eigval < 0
            if np.any((eigval < 0) & (np.abs(eigval) <= neg_tol)):
                eigval[eigval < 0] = 0.0
            if np.any(eigval < -neg_tol):
                print("Warning: significant negative eigenvalues (numerical?) for alpha = ", self.alpha[i])
                print("min eigenval:", eigval.min())
                eigval = np.maximum(eigval, -self.alpha[i] + abs_tol)
            
            
            evs[i][:] = eigval
            S[i] = np.sum(rho[i][:] - self.def_model) - np.nansum(rho[i][:]*np.log(rho[i][:]/self.def_model))
            G = Di(kernel, rho[i][:]/(2*np.pi), self.delomega)
            diff = corr - G
            L[i] = 0.5 * diff @ self.cov_mat_inv @ diff 
            prefactor[i] = np.prod(np.sqrt(self.alpha[i]/(self.alpha[i] + eigval)))
            exp[i] = prefactor[i] * np.exp(self.alpha[i] * S[i] - L[i]) 
        P_alphaDHM = P_alphaHM * exp

        # --- diagnostics ---
        plt.figure(2, figsize=(15,5))
        plt.subplot(1,3,1)
        plt.plot(np.log(self.alpha), np.log(evs), alpha = 0.3, linestyle = "--")
        plt.plot(np.log(self.alpha), np.log(self.alpha), label = "log(alpha)")
        plt.xlabel("log(alpha)")
        plt.ylabel("log(mean eigenvalue)")
        plt.legend()
        plt.subplot(1,3,2)
        plt.plot(self.alpha, P_alphaDHM, label = "P[alpha|D,H,M]")
        plt.plot(self.alpha, prefactor, label = "prefactor")
        plt.plot(self.alpha, np.exp(self.alpha*S - L), label = "exp")
        plt.legend()
        plt.xlabel("alpha")
        plt.ylabel("P[alpha|D,H,M]")
        plt.subplot(1,3,3)
        plt.plot(self.alpha, self.alpha*S, label = "alpha*S")
        plt.plot(self.alpha, L, label = "L")
        plt.plot(self.alpha, self.alpha*S - L, label = "Q")
        plt.xlabel("alpha")
        plt.ylabel("Q, L, alpha*S")
        plt.legend()
        plt.tight_layout()
        plt.savefig("mem_prob_dist.png")
        # --- end diagnostics ---

        alpha_ind = []
        for i in range(len(self.alpha)):
            if P_alphaDHM[i] >= 10e-8 * P_alphaDHM.max():
                alpha_ind.append(i)
        alpha_int = np.zeros(len(alpha_ind))
        rho_red = np.zeros((len(alpha_ind), len(self.w)))
        P_alphaDHM_red = np.zeros(len(alpha_ind))
        for i in range(len(alpha_ind)):
            alpha_int[i] = self.alpha[alpha_ind[i]]
            P_alphaDHM_red[i] = P_alphaDHM[alpha_ind[i]]
            rho_red[i][:] = rho[alpha_ind[i]][:]
        normalizing_fac = integrate.trapezoid(P_alphaDHM_red, alpha_int)
        rho_out = np.zeros(len(self.w))
        if len(alpha_int) <= 2: 
            P_alphaDHM_red = 1
            normalizing_fac = 1
            print("Warning: Probability distribution too narrow, using maximum only for averaging")
            return np.mean(rho, axis=0), P_alphaDHM, P_alphaDHM_red, alpha_int, Hess_mat
        for i in range(len(self.w)):
            rho_out[i] = integrate.trapezoid(np.transpose(rho_red)[i][:] * P_alphaDHM_red/normalizing_fac, alpha_int)
        return rho_out, P_alphaDHM, P_alphaDHM_red/normalizing_fac, alpha_int, Hess_mat

    def step3(
            self, 
            rho_alpha: np.ndarray, 
            Hess_L: np.ndarray, 
            w_region: np.ndarray, 
            P_alphaDHM_normed: np.ndarray, 
            alpharegion: np.ndarray
            ) -> np.ndarray:
        """Error estimation as per MEM paper step 3"""
        print("Calculating error in region:", w_region[0], w_region[-1])
        Hess_Q = np.zeros((len(self.alpha), len(w_region), len(w_region)))
        Hess_Q_inv = np.zeros((len(self.alpha), len(w_region), len(w_region)))
        Hess_S = np.zeros((len(self.alpha), len(w_region), len(w_region)))
        n_alpha = len(alpharegion)
        rho_var = np.zeros((n_alpha))
        rho_var_temp = np.zeros((n_alpha, len(w_region)))
        integrand = np.zeros((n_alpha)) 
        for i in range(len(self.w)):
            if w_region[0] == self.w[i]:
                w_start_index = i
                break
        norm = (w_region[len(w_region)-1] - w_region[0])**2
        for i in range(n_alpha):
            for k in range(len(w_region)):
                for n in range(len(w_region)):
                    if k == n:
                        if rho_alpha[i][k+w_start_index] == 0:
                            print("rho_alpha value error")
                        Hess_S[i][k][n] = - 1/(rho_alpha[i][k+w_start_index] * self.delomega)
                    Hess_Q[i][k][n] = alpharegion[i] * Hess_S[i][k][n] - Hess_L[k][n]
            Hess_Q_inv[i][:][:] = np.linalg.inv(Hess_Q[i][:][:])
        for i in range(n_alpha):
            for j in range(len(w_region)):
                rho_var_temp[i][j] = integrate.trapezoid(Hess_Q_inv[i][j][:], w_region)
            rho_var[i] = integrate.trapezoid(rho_var_temp[i][:], w_region) 
            rho_var[i] *= -1/norm
            integrand[i] = rho_var[i] * P_alphaDHM_normed[i]
        rho_out_var = integrate.trapezoid(integrand, alpharegion)
        return rho_out_var

    def fitCorrelator(
        self, 
        x: np.ndarray, 
        correlator: np.ndarray, 
        finiteT_kernel: bool, 
        Nt: int, 
        omega: np.ndarray, 
        extractedQuantity: str = "RhoOverOmega", 
        verbose: bool = True
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit a correlator using the Maximum Entropy Method."""
        kernel = self.initKernel(extractedQuantity, finiteT_kernel, Nt, x, omega)
        if verbose:
            print("*"*40)
            print("Starting minimization using svd")
        rho_min = self.step1(correlator, kernel)
        if verbose:
            print("*"*40)
            print("Starting calculation of probability distribution")
        rho_out, Prob_dist, Prob_dist_normed, alpha_reg, Hess_L = self.step2(rho_min, correlator, kernel)
        if np.any(np.isnan(rho_min)):
            print("*"*40)
            print("Nan value in rho_out detected. Aborting error evaluation.")
            error = np.zeros(len(omega))
        if isinstance(Prob_dist_normed, int):
            print("*"*40)
            print("Probability distribution empty. Aborting error evaluation.")
            error = np.zeros(len(omega))
        else:
            if verbose:
                print("*"*40)
                print("Starting error evaluation")
            error_region = self.w[:50]
            error = self.step3(rho_min, Hess_L, error_region, Prob_dist_normed, alpha_reg)
        return rho_out, error, Prob_dist

class ParameterHandler:
    def __init__(
            self, 
            paramsDefaultDict: dict
            ):
        self.allowed_params = paramsDefaultDict.keys()
        self.params = paramsDefaultDict
        
    def load_from_json(
            self, 
            config_path: str
            ) -> None:
        if config_path:
            with open(config_path, 'r') as f:
                data = json.load(f)
            for name in self.allowed_params:
                if name in data:
                    self.params[name] = data[name]

    def override_with_args(
            self, 
            args: argparse.Namespace
            ) -> None:
        for name in self.allowed_params:
            val = getattr(args, name, None)
            if val is not None:
                self.params[name] = val

    def check_parameters(
            self
            ) -> None:
        for name in self.allowed_params:
            if name == "outputFile" and self.params[name] is None:
                continue
            if name not in self.params or self.params[name] is None:
                raise ValueError(f"Parameter '{name}' is not set.")

    def load_params(
            self, 
            config_path: str, 
            args: argparse.Namespace
            ) -> None:
        self.load_from_json(config_path)
        self.override_with_args(args)
        self.check_parameters()

    def get_params(
            self
            ) -> dict:
        return self.params
    
    def get_extractedQuantity(
            self
            ) -> str:
        return self.params["extractedQuantity"]
    
    def get_correlator_file(
            self
            ) -> str:
        return os.path.abspath(self.params["correlatorFile"])

    def get_verbose(
            self
            ) -> bool:
        return self.params["verbose"]
    
    def get_correlator_cols(
            self
            ) -> List[int]:
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
    def __init__(
            self, 
            parameterHandler: ParameterHandler
            ):
        self.parameterHandler = parameterHandler
        self.alpha = np.logspace(
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
        self.default_model = get_default_model(self.omega, 
                                               self.parameterHandler.get_params()["default_model"], 
                                               self.parameterHandler.get_params()["default_model_file"])
        self.finiteT_kernel = self.parameterHandler.get_params()["FiniteT_kernel"]
        self.verbose = self.parameterHandler.get_verbose()
        self.multiFit = self.parameterHandler.get_params()["multiFit"]
        self.extractedQuantity = self.parameterHandler.get_extractedQuantity()
        self.Nt = self.parameterHandler.get_params()["Nt"] or len(self.x)
        self.outputDir = os.path.abspath(self.parameterHandler.get_params()["outputDir"])
        self.outputFile = self.parameterHandler.get_params()["outputFile"] or f"{self.extractedQuantity}_{os.path.basename(self.parameterHandler.get_correlator_file())}"
        cov = np.diag(self.error ** 2)
        cov_inv = np.linalg.inv(cov)
        self.fitter = mem(
             self.omega,
             self.alpha,
             self.default_model,
             cov_inv,
             self.Nt)

    def extractColumns(
            self, 
            file: str, 
            x_col: int, 
            mean_col: int, 
            error_col: int, 
            correlator_cols: List[int]
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data = np.loadtxt(file)
        x = data[:, x_col]
        mean = data[:, mean_col]
        error = data[:, error_col]
        correlator = data[:, correlator_cols]
        return x, mean, error, correlator    

    def run_single_fit(
            self, 
            fittedQuantity, 
            messageString, 
            results: List[np.ndarray],
            errors: List[np.ndarray], 
            probs: List[np.ndarray]
            ) -> None:
        start_time = time.time()
        print("=" * 40)
        print(messageString)
        print("=" * 40)
        rho, error, prob = self.fitter.fitCorrelator(
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
        probs.append(prob)

    
    def run_fits(
            self
            ) -> Tuple[np.ndarray, np.ndarray]:
        results = []
        errors = []
        probs = []
        if self.correlators.ndim == 1:
            self.correlators = np.array([self.correlators])
        else:
            self.correlators = self.correlators.T
        n_correlators = self.correlators.shape[0]
        self.run_single_fit(self.mean, "Fitting mean correlator", results, errors, probs)

        ############################### this needs to be uncommented later

        #for i, corr in enumerate(self.correlators):
        #    self.run_single_fit(corr, f"Fitting correlator sample {i + 1}/{n_correlators}", results, errors, probs)
        return np.array(results), np.array(errors), np.array(probs)

    def calculate_mean_error(
            self, 
            mean: np.ndarray, 
            samples: np.ndarray, 
            errormethod: str = "jackknife"
            ) -> np.ndarray:
        N = len(samples)
        fac = N - 1 if errormethod == "jackknife" else 1
        if errormethod not in ["jackknife", "bootstrap"]:
            raise ValueError("Invalid choice of error estimation method")
        return np.sqrt(fac / N * np.sum((samples - mean) ** 2, axis=0))

    def save_results(
            self, 
            mean: np.ndarray, 
            error: np.ndarray, 
            samples: np.ndarray, 
            prob: np.ndarray
            ) -> None:
        header = "Omega " + self.extractedQuantity + "_mean"
        if samples is not None and error is not None:
            header += f" {self.extractedQuantity}_error"
            for i in range(len(samples)):
                header += f" {self.extractedQuantity}_sample_{i}"
            header += " Probability_distribution"
            writeData = np.column_stack((self.omega, mean, error, samples.T, prob))
        else:
            writeData = np.column_stack((self.omega, mean))
        np.savetxt(os.path.join(self.outputDir, self.outputFile), writeData, header=header)
        prob_header = "alpha Probability_distribution"
        if prob is not None and len(prob) == len(self.alpha):
            probData = np.column_stack((self.alpha, prob))
            np.savetxt(os.path.join(self.outputDir, self.outputFile + "_prob"), probData, header=prob_header)
        if self.parameterHandler.get_params()["saveParams"]:
            self.save_params(self.parameterHandler.get_params(), os.path.join(self.outputDir, self.outputFile + ".params"))

    def save_params(
            self, 
            params: dict, 
            outputFile: str
            ) -> None:
        with open(outputFile + '.json', 'w') as f:
            json.dump(params, f, indent=4)


def initializeArgumentParser(
        paramsDefaultDict: dict
        ) -> argparse.ArgumentParser:
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

def main(
        paramsDefaultDict
        ):
    parser=initializeArgumentParser(paramsDefaultDict)
    args = parser.parse_args()
    parameterHandler = ParameterHandler(paramsDefaultDict)
    parameterHandler.load_params(args.config,args)

    if parameterHandler.get_verbose():
        print("*"*40)
        print("Running fits with the following parameters:")
        pprint.pprint(parameterHandler.get_params())

    fitRunner = FitRunner(parameterHandler)
    results, error, prob = fitRunner.run_fits()
    if len(np.squeeze(prob)) <= 1:
        prob = None
    mean = results[0]
    if len(results)>1:
        samples = results[1:]
        error = fitRunner.calculate_mean_error(samples,mean,parameterHandler.get_params()["errormethod"])
    else:
        samples = None
        error = None
    print("Saving results to:", os.path.join(fitRunner.outputDir, fitRunner.outputFile))
    fitRunner.save_results(mean,error,samples, np.squeeze(prob))



    
paramsDefaultDict = {
    "Method": "MEM",
    #MEM specific; default model: choose from quadratic, constant, file, exact
    #should not give 0 as a minimum for alpha, just a small value
    "alpha_min": 0,
    "alpha_max": 10,
    "alpha_points": 64,
    "default_model": "constant",
    "default_model_file": "",
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
