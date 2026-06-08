import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def save_spectral_function(Nt,Nb,w_vec,rho_min,rho_max,epsilon_vec):
    output_file = open("nevanlinna_Nt%i_Nb%i_z_epsilon0.00.dat"%(Nt,Nb),"w")

    output_file.write("# EPSILON_RANGE:")

    for epsilon in epsilon_vec:
        output_file.write(" %.2f"%epsilon)
    output_file.write("\n#\n")

    output_file.write("# omega\trho_min\trho_max\n")

    for n in range(len(w_vec)):
        output_file.write("%.6f\t%.6e\t%.6e\n"%(w_vec[n],rho_min[n],rho_max[n]))

def ansatz(x,a,b,c):
    return a + b*x + c*x**2

def read_files(Ns,Nt,Nb,beta,direction):

    epsilon_vec = np.array([])
    rho_max_dict = dict()
    rho_min_dict = dict()
    #rho_err_dict = dict()

    w_vec = np.array([])

    for epsilon in np.arange(0.2,0.31,0.01):
        input_file = "nevanlinna_Nt%i_Nb%i_%s_epsilon%.2f.dat"%(Nt,Nb,direction,epsilon)
        if os.path.isfile(input_file):
            data_matrix = np.loadtxt(input_file)

            rho_min_dict[float(epsilon)] = dict()
            rho_max_dict[float(epsilon)] = dict()

            #epsilon_vec = np.append(epsilon_vec,epsilon)

            w_vec = data_matrix[:,0]

            for n in range(len(w_vec)):
                rho_min_dict[epsilon][n] = data_matrix[:,1][n]
                rho_max_dict[epsilon][n] = data_matrix[:,2][n]

    epsilon_vec = np.array([eps for eps in rho_min_dict.keys()])

    N_omega_points = len(rho_min_dict[epsilon_vec[0]])

    rho_min_0 = np.array([])
    rho_max_0 = np.array([])

    for n in range(N_omega_points):
        rho_min_vec = np.array([rho_min_dict[epsilon][n] for epsilon in epsilon_vec])
        rho_max_vec = np.array([rho_max_dict[epsilon][n] for epsilon in epsilon_vec])

        popt_min,_ = curve_fit(ansatz,epsilon_vec,rho_min_vec)
        popt_max,_ = curve_fit(ansatz,epsilon_vec,rho_max_vec)

        rho_min_0 = np.append(rho_min_0,ansatz(0,*popt_min))
        rho_max_0 = np.append(rho_max_0,ansatz(0,*popt_max))

    save_spectral_function(Nt,Nb,w_vec,rho_min_0,rho_max_0,epsilon_vec)

    plt.fill_between(Nt*w_vec,rho_min_0,rho_max_0,alpha=0.6,edgecolor="gray")


Ns = 48
Nt = 16
Nb = 12
beta = 6.872
direction = "z"

read_files(Ns,Nt,Nb,beta,direction)

plt.show()
