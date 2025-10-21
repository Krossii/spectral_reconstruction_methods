import numpy as np
import matplotlib.pyplot as plt

losshistory = False

home_path = "/mnt/c/Users/chris/Desktop"
method = "unsupervised_ml"

function = "BW"  # 2PGAUSS
temp = "finite_T" # finite_T
extr_Q = "RhoOverOmega" # RhoOverOmega

Nt = 16

mock_data = False
noise = [2,3,4] # currently this only allows a list of noise levels --- handle this dynamically later

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

def read_file_l1(filename):
    data = np.loadtxt(filename)
    return data[:,1]

if losshistory:
    train_loss_history_data = read_file("/home/Christian/Desktop/spec_rec_methods/supervised_ml/outputs/RhoOverOmega_mock_BW_Nt16_noise4_rec_s1e-1_l21e-2_b256_conv.txt.trainloss.dat")
    val_loss_history_data = read_file("/home/Christian/Desktop/spec_rec_methods/supervised_ml/outputs/RhoOverOmega_mock_BW_Nt16_noise4_rec_s1e-1_l21e-2_b256_conv.txt.valloss.dat")

if mock_data:
    # --- load the input data and format the input correlator --- 
    true_spf = read_file_l1(f"{home_path}/mock-data-main/{temp}/uncorrelated_data/{function}/exact_spectral_function_{function}.dat")
    w = np.loadtxt(f"{home_path}/mock-data-main/{temp}/uncorrelated_data/{function}/exact_spectral_function_{function}.dat")
    w = w[:,0]
    G_input, G_input_err = np.zeros((len(noise), Nt)), np.zeros((len(noise), Nt))
    for i in range(len(noise)):
        G_input_data = np.loadtxt(f"{home_path}/mock-data-main/{temp}/uncorrelated_data/{function}/mock_corr_{function}_Nt{Nt}_noise{noise[i]}.dat")
        tau = G_input_data[:,0]
        G_input[i][:] = G_input_data[:,1]
        G_input_err[i][:] = G_input_data[:,2]

else:
    w = np.linspace(0,10,500)
    G_input, G_input_err = np.zeros(Nt),np.zeros(Nt)
    G_input_data = np.loadtxt(f"{home_path}/spectral_reconstruction_methods/dat/data_wilson_emconduc_48_{Nt}_b6.872_B2_z.txt")
    tau = G_input_data[:,0]
    G_input = G_input_data[:,1]
    G_input_err = G_input_data[:,2]

def load_MEM():
    # --- initialize kernel ---

    K=KL_kernel_Position_Vacuum(tau, w)

    # --- load the predicted spectral function ---

    predicted_spf = np.zeros((len(noise), len(true_spf)))
    spf_var = np.zeros((len(noise), len(true_spf)))
    for i in range(len(noise)):
        spf_data = np.loadtxt(f"{home_path}/spectral_reconstruction_methods/mem/outputs/{extr_Q}_{temp}_mock_corr_{function}_Nt{Nt}_noise{noise[i]}.dat")
        predicted_spf[i][:] = spf_data[:,1]
        spf_var[i][:] = np.sqrt(spf_data[:,2])

    # --- calculate the output corr ---

    G_output = np.zeros((len(noise), Nt))
    G_output_err = np.zeros((len(noise), Nt))
    for i in range(len(noise)):
        G_output[i][:] = Di(K, predicted_spf[i][:]/(2*np.pi), w[1]-w[0])
        G_output_err[i][:] = Di(K, spf_var[i][:]/(2*np.pi), w[1]-w[0])

    # --- calculate the default model for MEM ---

    m_0 = np.trapezoid(true_spf, x=w)/np.trapezoid(np.ones(len(true_spf)), x=w)
    def_model = np.ones(len(true_spf))
    default_model = def_model*m_0
    return predicted_spf, spf_var, G_output, G_output_err, default_model

def load_BG():
    # --- initialize the kernel ---

    K=KL_kernel_Position_Vacuum(tau, w)

    # --- some parameter stuff ---

    if extr_Q == "Rho":
        rec = "exp_sym"
    if extr_Q == "RhoOverOmega":
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
    spf_var = np.zeros((len(noise), len(true_spf)))
    for i in range(len(noise)):
        spf_data = np.loadtxt(f"{home_path}/zero_T_uncorrelated/{function}/Gaussian/gauss_specf.mock_corr_{function}_Nt{Nt}_noise{noise[i]}.dat")
        predicted_spf_g[i][:] = spf_data[:,1] *2*np.pi
        spf_var_g[i][:] = spf_data[:,2]*2*np.pi
        for j in range(len(true_spf)):
            predicted_spf[i][j] = predicted_spf_g[i][j*2] *w[j]
            spf_var[i][j] = spf_var_g[i][j*2] *w[j]

    # --- calculate the output corr ---

    G_output = np.zeros((len(noise), Nt))
    G_output_err = np.zeros((len(noise), Nt))
    for i in range(len(noise)):
        G_output[i][:] = Di(K, predicted_spf[i][:]/(2*np.pi), w[1]-w[0])
        G_output_err[i][:] = Di(K, spf_var[i][:]/(2*np.pi), w[1]-w[0])
    
    return predicted_spf, spf_var, G_output, G_output_err

def load_unsupervised():
    # --- initialize the kernel ---

    K=KL_kernel_Omega(KL_kernel_Position_FiniteT,tau, w, args = (1/Nt,))

    # --- load the predicted spectral function --- (I only take half the values here - is this even correct?)
    
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

def plotting_MEM(rho_input, rho_learned, rho_err, G_exact, G_err, G_learned, G_learned_err, d_model):
    colors = ['tomato', 'mediumseagreen', 'turquoise', 'deepskyblue', 'mediumslateblue', 'violet', 'plum', 'pink']
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(w, rho_input, label='True ρ', color='cornflowerblue')
    for i in range(len(noise)):
        plt.plot(w, rho_learned[i][:], label=f"Noise{i+2}", color=colors[i])
        plt.fill_between(w, rho_learned[i][:] - rho_err[i][:], rho_learned[i][:] + rho_err[i][:], color = colors[i], alpha = 0.5)
    plt.plot(w, d_model, label = 'Prior', color = 'black', linestyle = 'dashed', alpha = 0.6)
    plt.legend()
    plt.ylim(0,2.5)
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

def plotting_unsupervised(rho_learned, rho_err, G_exact, G_err, G_learned, G_learned_err):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(w, rho_learned, label="Reconstructed ρ", color="tomato")
    plt.fill_between(w, rho_learned - rho_err, rho_learned + rho_err, color = "tomato", alpha = 0.5)
    plt.legend()
    plt.ylabel(r"$\rho (\omega)/ \omega$")
    plt.xlabel(r"$\omega$")
    plt.title("Spectral Function")

    plt.subplot(1, 2, 2)
    plt.errorbar(tau,  G_exact, yerr=G_err, label='True G', color='cornflowerblue', capsize = 3, markeredgewidth = 1, elinewidth=1, fmt = 'x')
    plt.errorbar(tau, G_learned, yerr=G_learned_err, label="Reconstructed G", fmt = 'x', color="tomato", capsize = 3, markeredgewidth = 1, elinewidth=1)
    plt.yscale("log")
    plt.ylabel(r"$G(\tau)$")
    plt.xlabel(r"$\tau$")
    plt.legend()
    plt.title("Correlator")

    plt.tight_layout()
    

#predicted_spf, spf_var, G_output, G_output_err, default_model = load_MEM()
#predicted_spf, spf_var, G_output, G_output_err = load_BG()
#predicted_spf, spf_var, G_output, G_output_err = load_gaussian()
predicted_spf, spf_var, G_output, G_output_err = load_unsupervised()

if losshistory:
    plotting_spf_loss(true_spf, predicted_spf, train_loss_history_data, val_loss_history_data)
else:
    #plotting_MEM(true_spf, predicted_spf, spf_var, G_input[0][:], G_input_err[0][:], G_output, G_output_err, default_model)
    #plotting_BG_Gauss(true_spf, predicted_spf, spf_var, G_input[0][:], G_input_err[0][:], G_output, G_output_err)
    plotting_unsupervised(predicted_spf, spf_var, G_input, G_input_err, G_output, G_output_err)

if mock_data:
    plt.savefig(f"{method}_{extr_Q}_{function}_{temp}_Nt{Nt}_noises_comparison.png")
else:
    plt.savefig(f"{method}_{extr_Q}_{function}_{temp}_Nt{Nt}_B2_lat.png")