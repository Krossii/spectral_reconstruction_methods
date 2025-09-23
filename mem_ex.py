# Bryan's Maximum Entropy Method (MEM) implementation
# Reference: R.K. Bryan, Eur. Biophys. J. 18, 165–174 (1990)
import numpy as np
from scipy.optimize import root
from scipy.linalg import svd
from scipy.optimize import least_squares
np.random.seed(3000)

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

def mem_bryan(K, d, m, alpha, err, max_iter=100):
    """
    Bryan's MEM implementation.
    K: Kernel matrix (N_data x N_model)
    d: Data vector (N_data)
    m: Default model (N_model)
    alpha: Regularization parameter
    err: Error vector (N_data)
    Returns: rho (N_model)
    """

    # SVD decomposition
    U, s, V = svd(K.T, full_matrices=False)
    # Truncate small singular values
    tol = 1e-10
    s_mask = s > tol
    U = U[:,s_mask]
    s = s[s_mask]
    V = V[:, s_mask]

    # Reduced space
    N_red = len(s)
    # Initial guess
    u0 = np.zeros(N_red)
    #u0 = np.random.rand(N_red)*0.1

    def func(u):
        # rho in model space
        rho = m * np.exp(U @ u)
        # G = K @ rho
        G = K @ rho
        # Gradient in reduced space
        grad = -alpha * u - V.T @ (G - d)
        return grad

    def jac(u):
        rho = m * np.exp(U @ u)
        J = -alpha * np.eye(N_red)
        J -= V.T @ K @ np.diag(rho) @ U
        return J
    
    """sol = root(func, u0, jac=jac, method='hybr', options={'maxfev': max_iter, 'xtol':1e-10})

    if not sol.success:
        print("Not converged:")
        print(alpha)
        print(sol.nfev)
        print(sol.message)"""

    sol = Newton_rhapson_minimization(U,s,V,u0,alpha,err,d,m)

    u_opt = sol
    rho_opt = m * np.exp(U @ u_opt)
    return rho_opt

def Newton_rhapson_minimization(U,s,V,u, alpha, err, d, m):
    max_mu_iter = 10000
    M = np.diag(s) @ V.T @ np.diag(1/err**2) @ V @ np.diag(s)
    mu = 0
    rho = m * np.exp(U @ u)
    F = K @ rho
    diff = F - d
    S_n = np.sum(rho - m - rho * np.log(rho / m))
    L_n = 0.5 * np.sum((diff / err)**2)
    Q_n = alpha *S_n - L_n
    for i in range(max_mu_iter):
        rho = m * np.exp(U @ u)
        T = U.T @ np.diag(rho) @ U
        J = alpha * np.eye(len(s)) + M @ T
        F = K @ rho
        diff = F - d
        g = np.diag(s) @ V.T @ np.diag(1/err**2) @ diff
        del_u = np.linalg.inv(J)@ (-alpha * u - g)
        if del_u.T @ T @ del_u <= 0.2* np.sum(m):
            Q_o = Q_n
            u += del_u
            rho = m * np.exp(U @ u)
            S_n = np.sum(rho - m - rho * np.log(rho / m))
            L_n = 0.5 * np.sum((diff / err)**2)
            Q_n = alpha *S_n -L_n
            del_Q = Q_o - Q_n
            mu = 0
            if abs(del_Q/Q_o) < 10e-5:
                print("Converged after iteration", i)
                print("del_Q/Q =", abs(del_Q/Q_o))
                break
        if mu == 0: mu = 10e-4* alpha 
        else: mu *= 10
        if i == max_mu_iter -1:
            print("Exceeded maximum iterations for alpha", alpha)
    return u

def compute_P_alpha(alpha, S, L, hessian_eigenvalues):
    """
    Compute Bryan's probability distribution P[alpha|D,H,M] for averaging.
    alpha: array of alpha values
    S: array of entropy values
    L: array of likelihood values
    hessian_eigenvalues: 2D array, shape (len(alpha), N_red)
    Returns: P_alpha (normalized)
    """
    Q = alpha * S - L
    # Compute the determinant factor for each alpha
    det_factor = np.prod(alpha[:, None] / (alpha[:, None] + hessian_eigenvalues), axis=1) ** 0.5
    # Unnormalized probability
    P_alpha = 1/alpha * np.exp(Q - np.max(Q)) * det_factor
    # Normalize
    alpharegion = alpha
    """region_P_alpha = []
    for i in range(len(P_alpha)):
        if P_alpha[i] >= 0.01 * np.max(P_alpha):
            region_P_alpha.append(i)
    alpharegion = alpha[np.array(region_P_alpha)]
    if alpharegion.size == 0:
        alpharegion = alpha"""

    P_alpha /= np.trapezoid(P_alpha, alpharegion)
    return P_alpha, alpharegion

def read_corr_data(filename):
    data = np.loadtxt(filename)
    x = data[:, 0]
    d = data[:, 1]
    err = data[:, 2]
    return x, d, err

def read_rho_data(filename):
    data = np.loadtxt(filename)
    omega = data[:, 0]
    rho = data[:, 1]
    return omega, rho

# Example usage:
if __name__ == "__main__":
    realistic = False
    if realistic:
        x, d, err = read_corr_data("/mnt/c/Users/chris/Desktop/mock-data-main/BW/mock_corr_BW_Nt16_noise3.dat")
        omega, rho_true = read_rho_data("/mnt/c/Users/chris/Desktop/mock-data-main/BW/exact_spectral_function_BW.dat")
        N_model = len(omega)
        N_data = len(x)

        K=KL_kernel_Omega(KL_kernel_Position_FiniteT,x,omega,args=(1/N_data,))
    else:
        N_data = 16
        N_model = 500
        x = np.linspace(0, 15, N_data)
        omega = np.linspace(0, 1, N_model)
        # Kernel: exponential decay
        K = np.exp(-x[:, None] * omega[None, :])
        # True spectral function: Gaussian
        rho_true = np.exp(-0.5 * (omega - 0.5)**2 / 0.05**2)
        # Data: K @ rho_true + noise
        d = K @ rho_true
        err = 0.05 * d * np.ones_like(d)
        d = d + np.random.normal(0, err)
    m = np.ones(N_model)
    #m *= omega**2
    #m /= np.trapezoid(m, omega)
    # Regularization parameter
    alpha = np.logspace(4,5,100)
    rho_mem = np.zeros((len(alpha), N_model))
    # Run MEM
    for i in range(len(alpha)):
        rho_mem[i][:] = mem_bryan(K, d, m, alpha[i], err)

    S = np.zeros(len(alpha))
    L = np.zeros(len(alpha))
    for i in range(len(alpha)):
        rho = rho_mem[i]
        S[i] = np.sum(rho - m - rho * np.log(rho / m))
        G = K @ rho
        diff = G - d
        print(S[i])
        L[i] = 0.5 * np.sum((diff / err)**2)

    Q = alpha * S - L
    """hessian_eigenvalues = np.zeros((len(alpha), len(rho_mem[0])))
    P_alpha = np.zeros(len(alpha))
    for i in range(len(alpha)):
        Hessian = np.sqrt(np.diag(rho_mem[i])) @ K.T @ np.diag(1/err**2) @ K @ np.sqrt(np.diag(rho_mem[i]))
        hessian_eigenvalues[i, :len(np.linalg.eigvalsh(Hessian))] = np.linalg.eigvalsh(Hessian)
    
    P_alpha, alpharegion = compute_P_alpha(alpha, S, L, hessian_eigenvalues)"""
    P_alpha = np.exp(Q - np.max(Q))  # For numerical stability
    P_alpha /= np.trapezoid(P_alpha, alpha)  # Normalize

    # Weighted average over alpha
    alpharegion = alpha

    rho_final = np.trapezoid(rho_mem.T * P_alpha, alpharegion, axis=1)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,5))
    plt.subplot(1,4,1)
    plt.plot(alpha, P_alpha)
    plt.xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('P[alpha|D,H,M]')
    plt.title('Alpha Probability Distribution')
    plt.subplot(1,4,2)
    plt.plot(alpha, S, label='Entropy S')
    plt.plot(alpha, L, label='Likelihood L')
    plt.plot(alpha, Q, label='Q = alpha*S - L')
    plt.xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('Value')
    plt.legend()
    plt.title('S, L, Q vs alpha')
    plt.subplot(1,4,3)
    plt.plot(omega, rho_true, label="True")
    plt.plot(omega, rho_final, label="MEM avg")
    for i in range(0, len(alpha), len(alpha)//5):
        plt.plot(omega, rho_mem[i], alpha=0.3)
    plt.legend()
    plt.xlabel('omega')
    plt.ylabel('rho(omega)')
    plt.title('Spectral Function Reconstruction')
    plt.subplot(1,4,4)
    plt.errorbar(x, d, yerr=err, fmt='x', label='Data', alpha=0.5, capsize=2, elinewidth=1)
    plt.scatter(x, K @ rho_final, label='Predicted Data', marker = 'x', color='orange')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('G(x)')
    plt.yscale('log')
    plt.title('Data Comparison')
    plt.tight_layout()
    plt.savefig("mem_bryan_example_avg.png")