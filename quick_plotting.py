import numpy as np
import matplotlib.pyplot as plt

losshistory = False

home_path = "/mnt/c/Users/chris/OneDrive/Desktop"
method = "mem"
defmod = "quadratic"

function = "BW"  # 2PGAUSS
temp = "zero_T" # finite_T
extr_Q = "Rho" # Rho

B_field = 4

Nt = 48

mock_data = True
noise = [3] # [2,3,4] 

N_samples = 10 # number of jackknife samples used in the reconstructions

def KL_kernel_Position_Vacuum(
        Position, 
        Omega
        ):
    Position = Position[:, np.newaxis]  # Reshape Position as column to allow broadcasting
    ker = np.exp(-Omega * np.abs(Position)) + np.exp(-Omega*(len(Position)-Position))
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

def read_file_l1(filename):
    data = np.loadtxt(filename)
    return data[:,1]

def read_file_l2(filename):
    data = np.loadtxt(filename)
    return data[:,2]

def divbyOmega(function, w):
    if w[0] == 0:
        w[0]= 1e-10
    return function/w

if losshistory:
    train_loss_history_data = read_file("/home/Christian/Desktop/spec_rec_methods/supervised_ml/outputs/RhoOverOmega_mock_BW_Nt16_noise4_rec_s1e-1_l21e-2_b256_conv.txt.trainloss.dat")
    val_loss_history_data = read_file("/home/Christian/Desktop/spec_rec_methods/supervised_ml/outputs/RhoOverOmega_mock_BW_Nt16_noise4_rec_s1e-1_l21e-2_b256_conv.txt.valloss.dat")

if mock_data:
    # --- load the input data and format the input correlator --- 
    if extr_Q == "RhoOverOmega":
        true_spf = read_file_l2(f"{home_path}/mock-data-main/{temp}/uncorrelated_data/{function}/exact_spectral_function_{function}.dat")
    else:
        true_spf = read_file_l1(f"{home_path}/mock-data-main/{temp}/uncorrelated_data/{function}/exact_spectral_function_{function}.dat")
    w = np.loadtxt(f"{home_path}/mock-data-main/{temp}/uncorrelated_data/{function}/exact_spectral_function_{function}.dat")
    if temp == "finite_T": w = w[:,0]/Nt
    else: w = w[:,0]
    G_input, G_input_err = np.zeros((len(noise), Nt)), np.zeros((len(noise), Nt))
    for i in range(len(noise)):
        G_input_data = np.loadtxt(f"{home_path}/mock-data-main/{temp}/uncorrelated_data/{function}/mock_corr_{function}_Nt{Nt}_noise{noise[i]}.dat")
        tau = G_input_data[:,0]
        G_input[i][:] = G_input_data[:,1]
        G_input_err[i][:] = G_input_data[:,2]

else:
    w = np.linspace(0,20,1000)
    true_spf = np.zeros(len(w))
    G_input, G_input_err = np.zeros(Nt),np.zeros(Nt)
    G_input_data = np.loadtxt(f"{home_path}/spectral_reconstruction_methods/dat/data_wilson_emconduc_48_{Nt}_b6.872_B{B_field}_z.txt")
    tau = G_input_data[:,0]
    G_input = G_input_data[:,1]
    G_input_err = G_input_data[:,2]



def load_MEM():
    # --- initialize kernel ---

    if temp == "finite_T":
        if extr_Q == "RhoOverOmega":
            K=KL_kernel_Omega(KL_kernel_Position_FiniteT, tau, w, args=(1/Nt,))
        else:
            K=KL_kernel_Position_FiniteT(tau, w, 1/Nt)
    if temp == "zero_T":
        if extr_Q == "RhoOverOmega":
            K=KL_kernel_Omega(KL_kernel_Position_Vacuum, tau, w)
        else:
            K=KL_kernel_Position_Vacuum(tau, w)

    # --- load the predicted spectral function ---

    if mock_data:
        predicted_spf = np.zeros((len(noise), len(w)))
        predicted_spf_bins = np.zeros((len(noise), N_samples-1, len(w)))
        spf_var = np.zeros((len(noise), len(w)))
        for i in range(len(noise)):
            spf_data = np.loadtxt(f"{home_path}/spectral_reconstruction_methods/mem/outputs/{extr_Q}_{temp}_prior_{defmod}_mock_corr_{function}_Nt{Nt}_noise{noise[i]}.dat")
            if temp == "finite_T":
                predicted_spf[i][:] = spf_data[:,1]/Nt
                spf_var[i][:] = spf_data[:,2]/Nt
                for j in range(N_samples-1):
                    predicted_spf_bins[i][j][:] = spf_data[:,j+3]/Nt
            else:
                predicted_spf[i][:] = spf_data[:,1]
                spf_var[i][:] = spf_data[:,2]
                for j in range(N_samples-1):
                    predicted_spf_bins[i][j][:] = spf_data[:,j+3]

    else:
        spf_data = np.loadtxt(f"{home_path}/spectral_reconstruction_methods/mem/outputs/{extr_Q}_{temp}_data_wilson_emconduc_48_{Nt}_b6.872_B{B_field}_z.txt")
        predicted_spf = spf_data[:,1]
        spf_var = np.sqrt(spf_data[:,2])
        

    # --- calculate the output corr ---

    if mock_data:
        G_output = np.zeros((len(noise), Nt))
        G_output_bins = np.zeros((len(noise), N_samples-1, Nt))
        G_output_err = np.zeros((len(noise), Nt))
        for i in range(len(noise)):
            if temp == "finite_T":
                G_output[i][:] = Di(K, predicted_spf[i][:]*Nt, w[1]-w[0])
                for j in range(N_samples-1):
                    G_output_bins[i][j][:] = Di(K, predicted_spf_bins[i][j][:]*Nt, w[1] - w[0])
            else:
                G_output[i][:] = Di(K, predicted_spf[i][:], w[1]-w[0])
                for j in range(N_samples-1):
                    G_output_bins[i][j][:] = Di(K, predicted_spf_bins[i][j][:], w[1] - w[0])
            G_output_err[i][:] = np.sqrt((N_samples-1) / N_samples * np.sum((G_output_bins[i] - G_output[i]) ** 2, axis=0))
    else:
        G_output = Di(K, predicted_spf, w[1]-w[0])
        G_output_err = Di(K, spf_var, w[1]-w[0])

    # --- calculate the default model for MEM ---

    if mock_data:
        default_model = np.ones(len(w))
        data = np.loadtxt(f"{home_path}/mock-data-main/{temp}/uncorrelated_data/BW/exact_spectral_function_BW.dat")
        omega_file = data[:, 0]
        exact = data[:, 1]
        if defmod == "constant":
            m_0 = np.trapz(exact, x=omega_file) / (omega_file[-1] - omega_file[0])
            if extr_Q == "RhoOverOmega":
                # Avoid division by zero at ω=0
                default_model = np.ones(len(w)) * m_0
            else:
                # Simple constant
                default_model = np.ones(len(w)) * max(m_0, 1e-10)
        if defmod == "quadratic":
            # Normalize to match integral
            m_0 = np.trapz(exact, x=omega_file) / np.trapz(omega_file**2, x=omega_file)
            default_model = m_0 * w**2
            if extr_Q != "RhoOverOmega":
                # Protect against zero at ω=0
                default_model[w == 0] = m_0 * 1e-10
            else:
                default_model[w == 0] = m_0
        if defmod == "file":
            data = np.loadtxt("/mnt/c/Users/chris/OneDrive/Desktop/unsupervised_results/UnsupAI_mock_corr_BW_Nt36_noise3.dat.txt")
            default_model = w*data[:, 1] # because recsults from unsupervised are rho/w
    else:
        default_model = read_file_l1(f"{home_path}/finite_T_finite_B/unsupervised_ml/rhoOomegaB{B_field}_z_omegamax20_pts1k.txt")
    return predicted_spf, spf_var, G_output, G_output_err, default_model

def load_BG():
    # --- initialize the kernel ---

    K=KL_kernel_Position_Vacuum(tau, w)

    # --- some parameter stuff ---

    #if extr_Q == "Rho":
    #    rec = "exp_sym"
    #if extr_Q == "RhoOverOmega":
    #    rec = "exp_sym_w"
    rec = "exp_sym_w"
    lamb = 0.001

    # --- load the predicted spectral function --- 
    
    predicted_spf = np.zeros((len(noise), len(true_spf)))
    spf_var = np.zeros((len(noise), len(true_spf)))
    for i in range(len(noise)):
        spf_data = np.loadtxt(f"{home_path}/zero_T_uncorrelated/{function}/BG/nt{Nt}/noise{noise[i]}/{rec}/rho_ll{lamb}.txt")
        predicted_spf[i][:] = spf_data[:,1]
        spf_var[i][:] = spf_data[:,2]

    # --- calculate the output corr ---

    G_output = np.zeros((len(noise), Nt))
    G_output_err = np.zeros((len(noise), Nt))
    for i in range(len(noise)):
        G_output[i][:] = Di(K, predicted_spf[i][:]/(2*np.pi), w[1]-w[0]) 
        G_output_err[i][:] = Di(K, spf_var[i][:]/(2*np.pi), w[1]-w[0]) 
    
    return predicted_spf, spf_var, G_output, G_output_err

def load_gaussian():
    # --- initialize the kernel ---

    K=KL_kernel_Position_Vacuum(tau, w)

    # --- load the predicted spectral function --- (I only take half the values here - is this even correct?)
    
    predicted_spf_g = np.zeros((len(noise), len(true_spf)*2))
    spf_var_g = np.zeros((len(noise), len(true_spf)*2))
    predicted_spf = np.zeros((len(noise), len(true_spf)))
    predicted_spf_bins_g = np.zeros((len(noise), N_samples-1, len(w)*2))
    predicted_spf_bins = np.zeros((len(noise), N_samples-1, len(w)))
    spf_var = np.zeros((len(noise), len(true_spf)))
    for i in range(len(noise)):
        spf_data = np.loadtxt(f"{home_path}/zero_T_uncorrelated/{function}/Gaussian/gauss_specf.mock_corr_{function}_Nt{Nt}_noise{noise[i]}.dat")
        predicted_spf_g[i][:] = spf_data[:,1] *2*np.pi
        spf_var_g[i][:] = spf_data[:,2]*2*np.pi
        for j in range(N_samples-1):
            predicted_spf_bins_g[i][j][:] = spf_data[:,j+3]*2*np.pi
        for j in range(len(true_spf)):
            predicted_spf[i][j] = predicted_spf_g[i][j*2] *w[j]
            spf_var[i][j] = spf_var_g[i][j*2] *w[j]
            for k in range(N_samples-1):
                predicted_spf_bins[i][k][j] = predicted_spf_bins_g[i][k][j*2] * w[j]

    # --- calculate the output corr ---

    G_output = np.zeros((len(noise), Nt))
    G_output_err = np.zeros((len(noise), Nt))
    G_output_bins = np.zeros((len(noise), N_samples-1, Nt))
    for i in range(len(noise)):
        G_output[i][:] = Di(K, predicted_spf[i][:]/(2*np.pi), w[1]-w[0])
        for j in range(N_samples-1):
            G_output_bins[i][j][:] = Di(K, predicted_spf_bins[i][j][:]/(2*np.pi), w[1] - w[0])
    G_output_err = np.sqrt((N_samples-1) / N_samples * np.sum((G_output_bins - G_output) ** 2, axis=1))
    
    return predicted_spf, spf_var, G_output, G_output_err

def load_unsupervised():
    # --- initialize the kernel ---

    K=KL_kernel_Omega(KL_kernel_Position_FiniteT,tau, w, args = (1/Nt,))

    # --- load the predicted spectral function --- 
    
    predicted_spf = np.zeros(len(w))
    spf_var = np.zeros(len(w))
    spf_data = np.loadtxt(f"{home_path}/finite_T_finite_B/{method}/{extr_Q}_data_wilson_emconduc_B2_z_w10_500_seed42.txt")
    predicted_spf = spf_data[:,1]
    spf_var = spf_data[:,2]

    # --- calculate the output corr ---

    G_output = np.zeros(Nt)
    G_output_err = np.zeros(Nt)
    G_output = Di(K, predicted_spf, w[1]-w[0])
    G_output_err = Di(K, spf_var, w[1]-w[0])

    return predicted_spf, spf_var, G_output, G_output_err

def load_supervised():
    # --- initialize the kernel ---

    K=KL_kernel_Omega(KL_kernel_Position_Vacuum,tau, w)

    # --- load the predicted spectral function --- 
    
    predicted_spf = np.zeros(len(w))
    spf_var = np.zeros(len(w))
    spf_data = np.loadtxt(f"supervised_ml/outputs/{extr_Q}_mock_corr_BW_Nt{Nt}_noise{noise[0]}.dat")
    predicted_spf = spf_data[:,1]
    spf_var = spf_data[:,2]

    # --- calculate the output corr ---

    G_output = np.zeros(Nt)
    G_output_err = np.zeros(Nt)
    G_output = Di(K, predicted_spf, w[1]-w[0])
    G_output_err = Di(K, spf_var, w[1]-w[0])

    return predicted_spf, spf_var, G_output, G_output_err

def plotting_spf(rho_input, rho_learned):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(w, rho_input, label='True ρ', color='cornflowerblue')
    plt.plot(w, rho_learned, label='Learned ρ', color='tomato')
    plt.legend()
    plt.title("Spectral Function divided by omega")

def plotting_spf_corr(rho_input, rho_learned, G_exact, G_err, G_learned):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(w, rho_input, label='True ρ', color='cornflowerblue')
    plt.plot(w, rho_learned, label='Learned ρ', color='tomato')
    plt.legend()
    plt.title("Spectral Function")

    plt.subplot(1, 2, 2)
    plt.errorbar(tau,  G_exact, G_err, label='True G', color='cornflowerblue', capsize = 3, markeredgewidth = 1, elinewidth=1, fmt = 'x')
    plt.scatter(tau, G_learned, label='G from Learned ρ', marker = 'x', color='tomato')
    plt.yscale("log")
    plt.legend()
    plt.title("Correlator")

    plt.tight_layout()

def plotting_spf_corr_loss(train_losses, val_losses, rho_input, rho_learned, G_exact, G_learned):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(w, rho_input, label='True ρ', color='cornflowerblue')
    plt.plot(w, rho_learned, label='Learned ρ', color='tomato')
    plt.legend()
    plt.title("Spectral Function divided by omega")

    plt.subplot(1, 3, 2)
    plt.errorbar(tau,  G_exact, abs(G_exact - tf.squeeze(G_in)), label='True G', color='cornflowerblue', capsize = 3, markeredgewidth = 1, elinewidth=1, fmt = 'x')
    plt.scatter(tau, G_learned, label='G from Learned ρ', marker = 'x', color='tomato')
    plt.yscale("log")
    plt.legend()
    plt.title("Correlator")

    plt.subplot(1, 3, 3)
    plt.plot(train_losses, label='Train loss', color='tomato')
    plt.plot(val_losses, label='Validation loss', color='cornflowerblue')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.legend()
    plt.yscale("log")

    plt.tight_layout()

def plotting_spf_loss(rho_input, rho_learned,train_losses, val_losses):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(w, rho_input/w, label='True ρ', color='cornflowerblue')
    plt.plot(w, rho_learned, label='Learned ρ', color='tomato')
    plt.legend()
    plt.title("Spectral Function divided by omega")

    plt.subplot(1,2,2)
    plt.plot(train_losses, label="Train loss", color="tomato")
    plt.plot(val_losses, label="Validation loss", color ="cornflowerblue")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.legend()
    plt.yscale("log")

    plt.tight_layout()

def plotting_MEM(w, rho_input, rho_learned, rho_err, G_exact, G_err, G_learned, G_learned_err, d_model):
    colors = ['tomato', 'mediumseagreen', 'turquoise', 'deepskyblue', 'mediumslateblue', 'violet', 'plum', 'pink']
    if temp == "finite_T":
        w = w*Nt
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    if mock_data:
        plt.plot(w, np.squeeze(rho_input), label='True ρ', color='cornflowerblue')
        for i in range(len(noise)):
            plt.plot(w, np.squeeze(rho_learned[i][:]), label=f"Noise{noise[i]}", color=colors[i])
            plt.fill_between(w, np.squeeze(rho_learned[i][:]) - np.squeeze(rho_err[i][:]), np.squeeze(rho_learned[i][:]) + np.squeeze(rho_err[i][:]), color = colors[i], alpha = 0.5)
    else:
        plt.plot(w, rho_learned, label=f"Learned ρ", color="tomato")
        plt.fill_between(w, rho_learned-rho_err, rho_learned+rho_err, color = "tomato", alpha =0.5)
    plt.plot(w, d_model, label = 'Prior', color = 'black', linestyle = 'dashed', alpha = 0.6)
    plt.legend()
    if extr_Q == "RhoOverOmega":
        plt.ylabel(r"$\rho (\omega)/ \omega$")
    else:
        plt.ylabel(r"$\rho (\omega)$")
    if temp == "finite_T":
        plt.xlabel(r"$\omega/T$")
    else:
        plt.xlabel(r"$\omega$")
    #plt.ylim(-0.1,50)
    plt.title("Spectral Function")

    plt.subplot(1, 2, 2)
    for i in range(len(noise)):
        plt.errorbar(tau,  np.squeeze(G_exact[i][:]), yerr=np.squeeze(G_err[i][:]), label='True G', color='cornflowerblue', capsize = 3, markeredgewidth = 1, elinewidth=1, fmt = 'x')
    if mock_data:
        for i in range(len(noise)):
            plt.errorbar(tau, G_learned[i][:], yerr=G_learned_err[i][:], label=f"Noise{noise[i]}", fmt = 'x', color=colors[i], capsize = 3, markeredgewidth = 1, elinewidth=1)
    else:
        plt.errorbar(tau,G_learned, yerr=G_learned_err, label=f"Learned G", fmt = 'x', color="tomato", capsize = 3, markeredgewidth = 1, elinewidth=1)
    plt.yscale("log")
    plt.ylabel(r"$G(\tau)$")
    plt.xlabel(r"$\tau$")
    plt.legend()
    plt.title("Correlator")

    plt.tight_layout()

def plotting_BG_Gauss(rho_input, rho_learned, rho_err, G_exact, G_err, G_learned, G_learned_err):
    colors = ['tomato', 'mediumseagreen', 'turquoise', 'deepskyblue', 'mediumslateblue', 'violet', 'plum', 'pink']
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(w, rho_input, label='True ρ', color='cornflowerblue')
    for i in range(len(noise)):
        plt.plot(w, rho_learned[i][:], label=f"Noise{i+2}", color=colors[i])
        plt.fill_between(w, rho_learned[i][:] - rho_err[i][:], rho_learned[i][:] + rho_err[i][:], color = colors[i], alpha = 0.5)
    plt.legend()
    plt.ylim(0,1.5)
    plt.ylabel(r"$\rho (\omega)$")
    plt.xlabel(r"$\omega$")
    plt.title("Spectral Function")

    plt.subplot(1, 2, 2)
    plt.errorbar(tau,  G_exact, yerr=G_err, label='True G', color='cornflowerblue', capsize = 3, markeredgewidth = 1, elinewidth=1, fmt = 'x')
    for i in range(len(noise)):
        plt.errorbar(tau, G_learned[i][:], yerr=G_learned_err[i][:], label=f"Noise{i+2}", fmt = 'x', color=colors[i], capsize = 3, markeredgewidth = 1, elinewidth=1)
    plt.yscale("log")
    plt.ylabel(r"$G(\tau)$")
    plt.xlabel(r"$\tau$")
    plt.legend()
    plt.title("Correlator")

    plt.tight_layout()

def plotting_ml(rho_learned, rho_err, G_exact, G_err, G_learned, G_learned_err):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(w, np.squeeze(rho_learned), label="Reconstructed ρ", color="tomato")
    plt.fill_between(w, np.squeeze(rho_learned) - np.squeeze(rho_err), np.squeeze(rho_learned) + np.squeeze(rho_err), color = "tomato", alpha = 0.5)
    plt.legend()
    plt.ylabel(r"$\rho (\omega)/ \omega$")
    plt.xlabel(r"$\omega$")
    plt.title("Spectral Function")

    plt.subplot(1, 2, 2)
    plt.errorbar(tau,  np.squeeze(G_exact), yerr=np.squeeze(G_err), label='True G', color='cornflowerblue', capsize = 3, markeredgewidth = 1, elinewidth=1, fmt = 'x')
    plt.errorbar(tau, G_learned, yerr=G_learned_err, label="Reconstructed G", fmt = 'x', color="tomato", capsize = 3, markeredgewidth = 1, elinewidth=1)
    plt.yscale("log")
    plt.ylabel(r"$G(\tau)$")
    plt.xlabel(r"$\tau$")
    plt.legend()
    plt.title("Correlator")

    plt.tight_layout()

def comparing_mock(
        rho_input, 
        rho_gauss, 
        rho_bg, 
        rho_mem, 
        rho_err_g, 
        rho_err_bg, 
        rho_err_mem, 
        G_exact,
        G_err,
        G_l_gauss,
        G_l_bg,
        G_l_mem,
        G_err_gauss,
        G_err_bg,
        G_err_mem,
        ):

    rho_input = divbyOmega(rho_input, w)
    rho_gauss = divbyOmega(rho_gauss, w)
    rho_err_g = divbyOmega(rho_err_g, w)
    rho_bg = divbyOmega(rho_bg, w)
    rho_err_bg = divbyOmega(rho_err_bg, w)
    rho_mem = divbyOmega(rho_mem, w)
    rho_err_mem = divbyOmega(rho_err_mem, w)

    omega=w[1:]
    rho_input = rho_input[1:]
    rho_gauss = rho_gauss[1:]
    rho_err_g = rho_err_g[1:]
    rho_bg = rho_bg[1:]
    rho_err_bg = rho_err_bg[1:]
    rho_mem = rho_mem[1:]
    rho_err_mem = rho_err_mem[1:]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(omega, rho_input, label = "Input ρ", color = "k")
    #plt.plot(w, rho_gauss, label="Gaussian", color="tomato")
    #plt.plot(w, rho_bg, label="BG", color="mediumseagreen")
    #plt.plot(w, rho_mem, label="MEM", color="violet")
    plt.fill_between(omega, rho_gauss - rho_err_g, rho_gauss + rho_err_g, color = "tomato", alpha = 0.5, label="Gaussian")
    plt.fill_between(omega, rho_bg - rho_err_bg, rho_bg + rho_err_bg, color = "mediumseagreen", alpha = 0.5, label = "BG")
    plt.fill_between(omega, rho_mem - rho_err_mem, rho_mem + rho_err_mem, color = "violet", alpha = 0.5, label = "MEM")
    plt.legend()
    plt.ylim(0,1.7)
    plt.ylabel(r"$\rho (\omega) / \omega$")
    plt.xlabel(r"$\omega$")
    plt.title("Spectral Function")

    plt.subplot(1, 2, 2)
    plt.errorbar(tau,  G_exact, yerr=G_err, label='True G', color='k', capsize = 3, markeredgewidth = 1, elinewidth=1, fmt = 'x')
    plt.errorbar(tau, G_l_gauss, yerr=G_err_gauss, label="Gaussian", fmt = 'x', color="tomato", capsize = 3, markeredgewidth = 1, elinewidth=1, alpha=0.7)
    plt.errorbar(tau, G_l_bg, yerr=G_err_bg, label="BG", fmt = 'x', color="mediumseagreen", capsize = 3, markeredgewidth = 1, elinewidth=1, alpha=0.7)
    plt.errorbar(tau, G_l_mem, yerr=G_err_mem, label="MEM", fmt = 'x', color="violet", capsize = 3, markeredgewidth = 1, elinewidth=1, alpha=0.7)
    plt.yscale("log")
    plt.ylabel(r"$G(\tau)$")
    plt.xlabel(r"$\tau$")
    plt.legend()
    plt.title("Correlator")

    plt.tight_layout()

predicted_spf, spf_var, G_output, G_output_err, default_model = load_MEM()
#predicted_spf_bg, spf_var_bg, G_output_bg, G_output_err_bg = load_BG()
#predicted_spf_gauss, spf_var_gauss, G_output_gauss, G_output_err_gauss = load_gaussian()
#predicted_spf, spf_var, G_output, G_output_err = load_supervised()

if losshistory:
    plotting_spf_loss(true_spf, predicted_spf, train_loss_history_data, val_loss_history_data)
else:
    """if mock_data:
        comparing_mock(
            true_spf, predicted_spf_gauss[0][:], predicted_spf_bg[0][:], predicted_spf_mem[0][:], spf_var_gauss[0][:], spf_var_bg[0][:], spf_var_mem[0][:], 
            G_input[0][:], G_input_err[0][:], G_output_gauss[0][:], G_output_bg[0][:], G_output_mem[0][:], G_output_err_gauss[0][:], G_output_err_bg[0][:], G_output_err_mem[0][:])
    else:
        plotting_MEM(true_spf, predicted_spf_mem, spf_var_mem, G_input, G_input_err, G_output_mem, G_output_err_mem, default_model)"""
    #plotting_BG_Gauss(true_spf, predicted_spf, spf_var, G_input[0][:], G_input_err[0][:], G_output, G_output_err)
    plotting_MEM(w, true_spf, predicted_spf, spf_var, G_input, G_input_err, G_output, G_output_err, default_model)

if mock_data:
    if method == "mem":
        if len(noise) > 1:
            plt.savefig(f"plots/{method}/{method}_{extr_Q}_prior_{defmod}_{function}_{temp}_Nt{Nt}_noise_comparison.png")
        else:
            plt.savefig(f"plots/{method}/{method}_{extr_Q}_prior_{defmod}_{function}_{temp}_Nt{Nt}_noise{noise[0]}.png")
    else:
        plt.savefig(f"plots/{method}/{method}_{extr_Q}_{function}_{temp}_Nt{Nt}_noise{noise[0]}.png")
else:
    plt.savefig(f"plots/{method}/{method}_{extr_Q}_{function}_{temp}_Nt{Nt}_B{B_field}_lat.png")