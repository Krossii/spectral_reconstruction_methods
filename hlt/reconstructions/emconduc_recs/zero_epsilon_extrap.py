import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import glob

EPSILON_MIN = 0.4
EPSILON_MAX = 0.6

def ansatz(x,a,b,c):
    return a + b*x + c*x**3

def read_files(Ns,Nt,Nb,beta,direction):

    epsilon_vec = np.array([])
    conduct_vec = np.array([])
    conduct_err = np.array([])

    w = 50

    all_files = glob.glob("hlt_specf_eps????.data_wilson_emconduc_%i_%i_b%.3f_B%i_%s.txt"%(Ns,Nt,beta,Nb,direction))

    for input_file in all_files:
        w_vec, rho, rho_err = np.loadtxt(input_file,usecols=(0,1,2),unpack=True)

        epsilon = float(input_file.split("eps")[1][:4])

        if epsilon >= EPSILON_MIN and epsilon <= EPSILON_MAX:
            epsilon_vec = np.append(epsilon_vec,epsilon)

            conduct_vec = np.append(conduct_vec, rho[0])
            conduct_err = np.append(conduct_err, rho_err[0])

    popt,_ = curve_fit(ansatz,epsilon_vec,conduct_vec,sigma=conduct_err)

    cont_epsilon = np.linspace(0,max(epsilon_vec),100)
    #print(popt)
    plt.errorbar(epsilon_vec,conduct_vec,yerr=conduct_err,fmt="s",capsize=4)
    plt.errorbar(cont_epsilon,ansatz(cont_epsilon,*popt),fmt="-",color="black")

    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$\sigma$")
    plt.legend(loc="upper right")

Ns = 48
Nt = 16
Nb = 12
beta = 6.872
direction = "z"

read_files(Ns,Nt,Nb,beta,direction)
read_files(Ns,Nt,0,beta,direction)

#plt.xlim(0,0.25)
#plt.savefig("plots/zero_eps_extrap_Nt%i_%s_noise%i.jpg"%(Nt,SpectralType,NoiseLevel))
plt.show()
