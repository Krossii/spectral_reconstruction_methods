import numpy as np
import matplotlib.pyplot as plt

losshistory = False

home_path = "/home/Christian/Desktop"

function = "BW"  # 2PGAUSS
temp = "zero_T" # finite_T
method = "mem" # other
extr_Q = "Rho" # RhoOverOmega

Nt = 48
noise = 2

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

# --- load the input data --- 
true_spf = read_file_l1(f"{home_path}/mock-data-main/{temp}/uncorrelated_data/{function}/exact_spectral_function_{function}.dat")
G_input_data = np.loadtxt(f"{home_path}/mock-data-main/{temp}/uncorrelated_data/{function}/mock_corr_{function}_Nt{Nt}_noise{noise}.dat")

# --- load the predicted spectral function ---

predicted_spf = read_file_l1(f"{home_path}/spec_rec_methods/{method}/outputs/{extr_Q}_mock_corr_{function}_Nt{Nt}_noise{noise}.dat")
w = np.loadtxt(f"{home_path}/mock-data-main/{temp}/uncorrelated_data/{function}/exact_spectral_function_{function}.dat")
w = w[:,0]

# --- format the input corr and calculate the output corr ---

tau = G_input_data[:,0]
G_input = G_input_data[:,1]
G_input_err = G_input_data[:,2]
K=KL_kernel_Position_Vacuum(tau, w)
G_output = Di(K, predicted_spf/(2*np.pi), w[1]-w[0])

# --- load the probability distribution for MEM ---

prob_data = np.loadtxt(f"{home_path}/spec_rec_methods/{method}/outputs/{extr_Q}_mock_corr_{function}_Nt{Nt}_noise{noise}.dat_prob")
alpha = prob_data[:,0]
prob = prob_data[:,1]

# --- calculate the default model for MEM ---

m_0 = np.trapezoid(true_spf, x=w)/np.trapezoid(np.ones(len(true_spf)), x=w)
def_model = np.ones(len(true_spf))
default_model = def_model*m_0


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
    plt.show()  

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

def plotting_MEM(rho_input, rho_learned, G_exact, G_err, G_learned, alpha, prob, d_model):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(w, rho_input, label='True ρ', color='cornflowerblue')
    plt.plot(w, rho_learned, label='Learned ρ', color='tomato')
    plt.plot(w, d_model, label = 'Prior', color = 'black', linestyle = 'dashed', alpha = 0.6)
    plt.legend()
    plt.ylabel("Rho")
    plt.xlabel("omega")
    plt.title("Spectral Function")

    plt.subplot(1, 3, 2)
    plt.errorbar(tau,  G_exact, G_err, label='True G', color='cornflowerblue', capsize = 3, markeredgewidth = 1, elinewidth=1, fmt = 'x')
    plt.scatter(tau, G_learned, label='G from Learned ρ', marker = 'x', color='tomato')
    plt.yscale("log")
    plt.ylabel("G(tau)")
    plt.xlabel("tau")
    plt.legend()
    plt.title("Correlator")

    plt.subplot(1,3,3)
    plt.plot(alpha, prob, label='P[al DHm]', color='tomato')
    plt.title("Probability distribution")
    plt.xlabel("alpha")
    plt.ylabel("P[al DHm]")
    plt.xscale("log")
    plt.legend()

    plt.tight_layout()

if losshistory:
    plotting_spf_loss(true_spf, predicted_spf, train_loss_history_data, val_loss_history_data)
else:
    plotting_MEM(true_spf, predicted_spf, G_input, G_input_err, G_output, alpha, prob, default_model)
plt.show()