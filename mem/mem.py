import scipy
from scipy import integrate
from scipy.linalg import solve
from scipy.optimize import root
from scipy.optimize import minimize
from scipy.optimize import check_grad
from scipy.optimize import least_squares
from dataclasses import dataclass, field

import matplotlib.pyplot as plt


# changed the signs in Q from -L to +L and changed the signs in S to -S


#from latqcdtools.base.check import ignoreDivideByZero, ignoreInvalidValue, ignoreUnderflow, ignoreOverflow
#from latqcdtools.base.speedify import parallel_function_eval
#ignoreDivideByZero()
#ignoreInvalidValue()
#ignoreUnderflow()
#ignoreOverflow()


#import imageio
#import imageio.v2 as imageio

import numpy as np
import json
import argparse
import time
import pprint
import os
import itertools
from typing import List, Tuple, Callable


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
    # Ensure both tensors are of the same data type (float64)
    KL = KL.astype(dtype=np.float64)  # Cast KL to float64
    rhoi = rhoi.astype(dtype=np.float64)  # Cast rhoi to float64
    delomega = delomega.astype(dtype=np.float64)  # Cast delomega to float64

    rhoi = np.reshape(rhoi, [-1, 1])

    # Perform matrix multiplication
    dis = KL @ rhoi
    dis = np.squeeze(dis, axis=-1)  # Remove the singleton dimension
    dis = dis * delomega  # Multiply by delomega
    return dis

def get_default_model(
        w: np.ndarray, 
        defmod: str, 
        file: str = "",
        ExtractedQuantity: str = "RhoOverOmega"
        ) -> np.ndarray:
    def_model = np.ones(len(w))
    if defmod == "constant":
        if file != "":
            data = np.loadtxt(file)
            omega_file = data[:, 0]
            exact = data[:, 1]
            m_0 = np.trapz(exact, x=omega_file) / (omega_file[-1] - omega_file[0])
            #if ExtractedQuantity == "RhoOverOmega":
            #    def_model = np.ones(len(w)) * m_0 / w
            #    def_model[w == 0] = 1e3  # Avoid division by zero
            #    return def_model
            #else:
            return np.ones(len(w)) * m_0
        else:
            return np.ones(len(w)) * 1e-3
    if defmod == "quadratic":
        if file != "":
            data = np.loadtxt(file)
            omega_file = data[:, 0]
            exact = data[:, 1]
            # Normalize to match integral
            m_0 = np.trapz(exact, x=omega_file) / np.trapz(omega_file**2, x=omega_file)
            def_model = m_0 * w**2
            #def_model[w == 0] = m_0
            return def_model
        else:
            def_model = w**2
            def_model = np.maximum(def_model, 1e-10)
            return def_model
    if defmod == "asakawa":
        m_0 = 0.0257
        return m_0 * w**2
    if defmod == "exact" or defmod == "file":
        data = np.loadtxt(file)
        def_model = w*data[:, 1] # because recsults from unsupervised are rho/w
        return def_model
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
            kernel: np.ndarray,
            ) -> np.ndarray:
        """Minimization of Q = alpha S - L for all alpha as per MEM paper step 1"""
        U, xi, Vt = np.linalg.svd(kernel.T, full_matrices=False)
        cutoff = 1e-10 * xi[0]  # relative to largest singular value
        xi = xi[xi > cutoff]
        s = xi.size
        Vt = Vt[:s,:]
        U = U[:,:s]
        print("Singular space dimension:", s, "down from:", min(self.N_t, len(self.w)))

        VXi = Vt.T @ np.diag(xi)

        M = VXi.T @ self.cov_mat_inv @ VXi
        rho_min = np.zeros((len(self.alpha), len(self.w)))

        u_g = np.zeros(M.shape[0], dtype=float)

        #def getRhoMin(al):
        #    return self.minimizer(corr, VXi, M, U, al, u_g, kernel)
        

        #rho_min_array = parallel_function_eval(getRhoMin, list(range(len(self.alpha))))


        for i in range(len(self.alpha)):
            rho_min_array = self.minmizer_root(corr, VXi, M, U, self.alpha[i], u_g, kernel)
            rho_min[i][:] = rho_min_array
            rho_safe = np.maximum(rho_min_array, 1e-300)
            def_model_safe = np.maximum(self.def_model, 1e-300)
            log_ratio = np.log(rho_safe / def_model_safe)

            # Check for NaN/Inf before proceeding
            if np.any(~np.isfinite(log_ratio)):
                print(f"  ⚠ Warning: NaN in log(rho/m), using zeros for u_g")
                u_g = np.zeros(M.shape[0])
            else:
                u_g = np.linalg.lstsq(U, log_ratio, rcond=None)[0]
            G_rho = Di(kernel, rho_min[i][:], self.delomega) 

        plt.figure(1, figsize=(12,4))
        plt.subplot(1,2,1)
        for i in range(len(self.alpha)):
            plt.plot(self.w, rho_min[i][:])
        plt.plot(self.w, self.def_model, color = "black", linestyle = "--")
        plt.subplot(1,2,2)
        plt.scatter(self.tau, corr, color = "tomato", marker = "x")
        plt.scatter(self.tau, G_rho, color = "cornflowerblue", marker = "x")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig("mem_alpha_scan.png")
        # Auto-select optimal alpha
        alpha_opt, rho_opt, idx_opt = self.select_optimal_alpha_from_scan(
            rho_min, corr, kernel, method='lcurve')
        
        print(f"\nSelected optimal α = {alpha_opt:.4e}")
        G_final = Di(kernel, rho_opt, self.delomega)
        L_final = 0.5 * (corr - G_final) @ self.cov_mat_inv @ (corr - G_final)
        print(f"  χ²/N = {2*L_final/len(corr):.3f}\n")
        
        return rho_min
    
    def minmizer_root(
        self,
        corr: np.ndarray,
        VXi: np.ndarray,
        M: np.ndarray,
        U: np.ndarray,
        al: float,
        u_guess: np.ndarray,
        kernel: np.ndarray
    ) -> np.ndarray:
        """Minimize Q = alpha*S + L using root finding"""        
        N_s = M.shape[0]
        
        def residual(b):
            """Gradient of Q (should equal zero at optimum)"""
            rho = self.def_model * np.exp(U @ b)
            G_rho = Di(kernel, rho, self.delomega) 
            g = VXi.T @ self.cov_mat_inv @ (G_rho - corr)
            f = -al * b - g
            return f

        def jac(b):
            """Jacobian of gradient (Hessian of Q)"""
            rho = self.def_model * np.exp(U @ b)
            diag_rho_U = np.diag(rho) @ U 
            A = (kernel @ diag_rho_U) * self.delomega
            J_nonlinear = VXi.T @ self.cov_mat_inv @ A
            J = -al * np.eye(N_s) - J_nonlinear
            return J

        # Primary solver: hybr
        res = root(
            residual,
            u_guess,
            jac=jac,
            method='hybr',
            options={
                'maxfev': 5000,
                'xtol': 1e-8,
                'factor': 0.1
            }
        )

        # Fallback if hybr fails
        if not res.success:
            res = root(
                residual,
                u_guess,
                jac=jac,
                method='lm',
                options={
                    'maxiter': 7500,
                    'xtol': 1e-8,
                    'ftol': 1e-8
                }
            )
        
        # Extract solution
        u = res.x
        # Check if solution actually changed from initial guess
        u_change = np.linalg.norm(u - u_guess)
        if u_change < 1e-6:
            print(f"    ⚠ WARNING: Solution didn't change from initial guess!")
            print(f"    This usually means optimizer failed or alpha is too small")
        
        rho = self.def_model * np.exp(U @ u)
        G_pred = Di(kernel, rho, self.delomega)

        # Check if rho is reasonable
        rho_change = np.linalg.norm(rho - self.def_model)
        if rho_change < 1e-6:
            print(f"    ⚠ WARNING: rho is identical to default model!")

        
        # Compute quality metrics
        final_res = residual(u)
        res_norm = np.linalg.norm(final_res)

        # Check convergence quality
        if not res.success:
            print(f"    ✗ Optimizer failed: {res.message}")
            if res_norm > 1e-2:
                print(f"    ✗ SEVERE: ||∇Q||={res_norm:.2e} - returning default model!")
                # Return default model if optimizer completely failed
                return self.def_model
        
        if res_norm > 1e-3:
            print(f"    ⚠ Poor convergence: ||∇Q||={res_norm:.2e}")
        
        S = np.sum(rho - self.def_model - 
                rho * np.nan_to_num(np.log(rho/self.def_model), 
                                    neginf=-1e300)) * self.delomega
        L = 0.5 * (corr - G_pred) @ self.cov_mat_inv @ (corr - G_pred)
        Q = al * S - L
        
        # Concise status output
        status = "✓" if res.success else "✗"
        print(f"  α={al:.2e} {status} ||∇Q||={res_norm:.2e} S={S:.3e} L={L:.3e} Q={Q:.3e}")
        
        if res_norm > 1e-3:
            print(f"    ⚠ Large gradient! Solution may be poor.")
        
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
            S[i] = (np.sum(rho[i][:] - self.def_model) - np.nansum(rho[i][:]*np.log(rho[i][:]/self.def_model))) * self.delomega
            G = Di(kernel, rho[i][:], self.delomega) 
            diff = corr - G
            L[i] = 0.5 * diff @ self.cov_mat_inv @ diff 
            prefactor[i] = np.prod(np.sqrt(self.alpha[i]/(self.alpha[i] + eigval)))
            exp[i] = prefactor[i] * np.exp(self.alpha[i] * S[i] - L[i]) 
        print("S-L:", self.alpha * S-L)
        print("prefactor, e(a*S-L)", prefactor, np.exp(self.alpha * S - L))
        P_alphaDHM = P_alphaHM * exp

        # --- diagnostics ---
        plt.figure(2, figsize=(15,5))
        plt.subplot(1,3,1)
        plt.plot(np.log(self.alpha), np.log(evs), alpha = 0.3, linestyle = "--")
        plt.plot(np.log(self.alpha), np.log(self.alpha))
        plt.xlabel("log(alpha)")
        plt.ylabel("log(mean eigenvalue)")
        plt.subplot(1,3,2)
        plt.plot(self.alpha, P_alphaDHM, color = "tomato")
        plt.plot(self.alpha, prefactor, color = "cornflowerblue", alpha = 0.5)
        plt.plot(self.alpha, np.exp(self.alpha*S + L), color = "green", alpha = 0.5)
        plt.xscale("log")
        plt.xlabel("alpha")
        plt.ylabel("P[alpha|D,H,M]")
        plt.subplot(1,3,3)
        plt.plot(self.alpha, self.alpha*S, color = "tomato")
        plt.plot(self.alpha, L, color = "cornflowerblue")
        plt.plot(self.alpha, self.alpha*S + L, color = "green")
        plt.xlabel("alpha")
        plt.ylabel("Q, L, alpha*S")
        plt.tight_layout()
        plt.savefig("mem_prob_dist.png")
        # --- end diagnostics ---

        alpha_ind = []
        for i in range(len(self.alpha)):
            if P_alphaDHM[i] >= 1e-8 * P_alphaDHM.max():
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
        return rho_out, P_alphaDHM/normalizing_fac, P_alphaDHM_red/normalizing_fac, alpha_int, Hess_mat

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

    def select_optimal_alpha_from_scan(self, rho_min, corr, kernel, method='lcurve'):        
        S_vals = np.zeros(len(self.alpha))
        L_vals = np.zeros(len(self.alpha))
        
        print("\n" + "="*60)
        print("AUTO-ALPHA SELECTION")
        print("="*60)
        
        for i in range(len(self.alpha)):
            rho = rho_min[i]
            S_vals[i] = np.sum(rho - self.def_model - 
                            rho * np.nan_to_num(np.log(rho/self.def_model), 
                                                neginf=-1e300)) * self.delomega
            G_pred = Di(kernel, rho, self.delomega)
            L_vals[i] = 0.5 * (corr - G_pred) @ self.cov_mat_inv @ (corr - G_pred)
        
        Q_vals = self.alpha * S_vals - L_vals

        print(f"\nS values: min={S_vals.min():.3e}, max={S_vals.max():.3e}")
        print(f"L values: min={L_vals.min():.3e}, max={L_vals.max():.3e}")
        print(f"Q values: min={Q_vals.min():.3e}, max={Q_vals.max():.3e}")
        print(f"\nFirst 5 Q values: {Q_vals[:5]}")
        print(f"Last 5 Q values: {Q_vals[-5:]}")
        
        # Check for identical solutions
        unique_rhos = []
        for i, rho in enumerate(rho_min):
            is_unique = True
            for unique_rho in unique_rhos:
                if np.allclose(rho, unique_rho, rtol=1e-6):
                    print(f"  α[{i}]={self.alpha[i]:.2e} gives identical rho to previous alpha")
                    is_unique = False
                    break
            if is_unique:
                unique_rhos.append(rho)
    
        print(f"\nNumber of unique solutions: {len(unique_rhos)} / {len(self.alpha)}")
        
        alpha_lcurve, idx_lcurve = self._lcurve_alpha(S_vals, L_vals)
        alpha_chi2, idx_chi2 = self._chi2_alpha(L_vals, len(corr))
        
        chi2_lcurve = 2 * L_vals[idx_lcurve] / len(corr)
        chi2_chi2 = 2 * L_vals[idx_chi2] / len(corr)
        
        print(f"\nL-curve method: α = {alpha_lcurve:.4e}, χ²/N = {chi2_lcurve:.3f}")
        print(f"Chi² method:    α = {alpha_chi2:.4e}, χ²/N = {chi2_chi2:.3f}")
        
        if method == 'lcurve' or method == 'both':
            idx_opt = idx_lcurve
            alpha_opt = alpha_lcurve
            print(f"Using L-curve: α = {alpha_opt:.4e}")
        else:
            idx_opt = idx_chi2
            alpha_opt = alpha_chi2
            print(f"Using Chi²: α = {alpha_opt:.4e}")
        
        rho_opt = rho_min[idx_opt]
        self._plot_alpha_diagnostics(S_vals, L_vals, Q_vals, corr, kernel, 
                                    rho_min, idx_opt, alpha_opt)
        print("="*60 + "\n")
        
        return alpha_opt, rho_opt, idx_opt

    def _lcurve_alpha(self, S_vals, L_vals):
        """L-curve maximum curvature"""
        valid = (L_vals > 0) & (np.abs(S_vals) > 1e-10)
        
        if np.sum(valid) < 5:
            idx = np.argmin(L_vals)
            return self.alpha[idx], idx
        
        alpha_v = self.alpha[valid]
        S_v = np.abs(S_vals[valid])
        L_v = L_vals[valid]
        
        log_S, log_L = np.log10(S_v), np.log10(L_v)
        order = np.argsort(log_S)
        log_S, log_L = log_S[order], log_L[order]
        alpha_sorted = alpha_v[order]
        
        ds, dl = np.gradient(log_S), np.gradient(log_L)
        dds, ddl = np.gradient(ds), np.gradient(dl)
        
        denom = np.maximum((ds**2 + dl**2)**(1.5), 1e-10)
        curv = np.abs(ds * ddl - dl * dds) / denom
        
        if len(curv) > 4:
            max_idx = np.argmax(curv[2:-2]) + 2
        else:
            max_idx = np.argmax(curv)
        
        alpha_opt = alpha_sorted[max_idx]
        idx_opt = np.argmin(np.abs(self.alpha - alpha_opt))
        return alpha_opt, idx_opt

    def _chi2_alpha(self, L_vals, N_data, target=1.0):
        """Chi-squared target method"""
        chi2_per_N = 2 * L_vals / N_data
        idx_opt = np.argmin(np.abs(chi2_per_N - target))
        return self.alpha[idx_opt], idx_opt

    def _plot_alpha_diagnostics(self, S_vals, L_vals, Q_vals, corr, kernel, 
                                rho_min, idx_opt, alpha_opt):
        """Diagnostic plots"""
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(15, 10))
        
        # L-curve
        plt.subplot(2, 3, 1)
        valid = (L_vals > 0) & (np.abs(S_vals) > 0)
        plt.loglog(np.abs(S_vals[valid]), L_vals[valid], 'o-', alpha=0.6, markersize=4)
        plt.loglog(np.abs(S_vals[idx_opt]), L_vals[idx_opt], 'r*', 
                markersize=20, label=f'α={alpha_opt:.2e}', zorder=10)
        plt.xlabel('|S|'); plt.ylabel('L'); plt.title('L-curve')
        plt.legend(); plt.grid(True, alpha=0.3)
        
        # Q vs alpha
        plt.subplot(2, 3, 2)
        plt.semilogx(self.alpha, Q_vals, 'o-', markersize=4)
        plt.axvline(alpha_opt, color='r', linestyle='--', linewidth=2)
        plt.xlabel('α'); plt.ylabel('Q = αS - L'); plt.title('Quality Functional')
        plt.grid(True, alpha=0.3)
        
        # S and L
        ax3 = plt.subplot(2, 3, 3)
        ax3_twin = ax3.twinx()
        ax3.semilogx(self.alpha, np.abs(S_vals), 'b-o', alpha=0.6, markersize=4, label='|S|')
        ax3_twin.semilogx(self.alpha, L_vals, 'r-s', alpha=0.6, markersize=4, label='L')
        ax3.axvline(alpha_opt, color='k', linestyle='--', linewidth=2)
        ax3.set_xlabel('α'); ax3.set_ylabel('|S|', color='b')
        ax3_twin.set_ylabel('L', color='r'); ax3.set_title('Entropy & Likelihood')
        ax3.grid(True, alpha=0.3)
        
        # Chi-squared
        plt.subplot(2, 3, 4)
        chi2_N = 2 * L_vals / len(corr)
        plt.semilogx(self.alpha, chi2_N, 'o-', markersize=4)
        plt.axhline(1.0, color='g', linestyle='--', linewidth=2, label='χ²/N = 1')
        plt.axvline(alpha_opt, color='r', linestyle='--', linewidth=2)
        plt.xlabel('α'); plt.ylabel('χ²/N')
        plt.title(f'Reduced χ² (opt: {chi2_N[idx_opt]:.2f})')
        plt.legend(); plt.grid(True, alpha=0.3)
        
        # Spectral functions
        plt.subplot(2, 3, 5)
        n_show = min(8, len(self.alpha))
        for idx in np.linspace(0, len(self.alpha)-1, n_show, dtype=int):
            if idx == idx_opt:
                plt.plot(self.w, rho_min[idx], 'r-', linewidth=3, 
                        label=f'α={self.alpha[idx]:.1e}', zorder=10)
            else:
                plt.plot(self.w, rho_min[idx], '-', linewidth=1, alpha=0.5)
        plt.plot(self.w, self.def_model, 'k--', linewidth=2, label='Default')
        plt.xlabel('ω'); plt.ylabel('ρ(ω)'); plt.title('Spectral Functions')
        plt.legend(); plt.grid(True, alpha=0.3); plt.ylim(bottom=-0.1)
        
        # Fit
        plt.subplot(2, 3, 6)
        G_opt = Di(kernel, rho_min[idx_opt], self.delomega)
        try:
            error = np.sqrt(np.diag(np.linalg.inv(self.cov_mat_inv)))
        except:
            error = np.ones(len(corr)) * 0.001
        tau = np.arange(len(corr))
        plt.errorbar(tau, corr, yerr=error, fmt='o', color='tomato', 
                    label='Data', capsize=3, markersize=6)
        plt.plot(tau, G_opt, 'b-', linewidth=2, label='Fit')
        plt.xlabel('τ'); plt.ylabel('G(τ)')
        plt.title(f'Fit (χ²/N={chi2_N[idx_opt]:.2f})')
        plt.legend(); plt.grid(True, alpha=0.3); plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig('alpha_selection_diagnostics.png', dpi=150)
        print("Plots → alpha_selection_diagnostics.png")

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
                print("Ignored omega specific error evaluation")
            #error_region = self.w[:50]
            #error = self.step3(rho_min, Hess_L, error_region, Prob_dist_normed, alpha_reg)
            error = np.zeros(len(omega))
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
        self.finiteT_kernel = self.parameterHandler.get_params()["FiniteT_kernel"]
        if self.finiteT_kernel: temp = "finite_T"
        else: temp = "zero_T"
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
        if temp == "finite_T":
            self.omega = np.linspace(
                self.parameterHandler.get_params()["omega_min"],
                self.parameterHandler.get_params()["omega_max"]/self.parameterHandler.get_params()["Nt"],
                self.parameterHandler.get_params()["omega_points"]
            )
        else:
            self.omega = np.linspace(
                self.parameterHandler.get_params()["omega_min"],
                self.parameterHandler.get_params()["omega_max"],
                self.parameterHandler.get_params()["omega_points"]
            )
        self.default_model = get_default_model(self.omega, 
                                               self.parameterHandler.get_params()["default_model"], 
                                               self.parameterHandler.get_params()["default_model_file"],
                                               self.parameterHandler.get_extractedQuantity())
        self.verbose = self.parameterHandler.get_verbose()
        self.multiFit = self.parameterHandler.get_params()["multiFit"]
        self.extractedQuantity = self.parameterHandler.get_extractedQuantity()
        self.Nt = self.parameterHandler.get_params()["Nt"] or len(self.x)
        self.outputDir = os.path.abspath(self.parameterHandler.get_params()["outputDir"])
        self.outputFile = self.parameterHandler.get_params()["outputFile"] or f"{self.extractedQuantity}_{temp}_prior_{self.parameterHandler.get_params()['default_model']}_{os.path.basename(self.parameterHandler.get_correlator_file())}"
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

        for i, corr in enumerate(self.correlators):
            self.run_single_fit(corr, f"Fitting correlator sample {i + 1}/{n_correlators}", results, errors, probs)
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
            writeData = np.column_stack((self.omega, mean, error, samples.T))
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
