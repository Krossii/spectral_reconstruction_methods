import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



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

class loading:
    def __init__(
            self,
            w: np.ndarray,
            tau: np.ndarray,
            Nt: int,
            finite_T: bool,
            home_path: str,
            mock_data: bool,
            **kwargs,
            ) -> None:
        """
        Initialize the loading class.
        Parameters:
        w (np.ndarray): The frequency grid.
        tau (np.ndarray): The Euclidean time grid.
        Nt (int): The number of time slices.
        finite_T (bool): Whether the data is at finite temperature or not.
        home_path (str): The home path for loading the data.
        mock_data (bool): Whether the data is mock data or not.
        kwargs: Additional keyword arguments for loading the data. Should include either 'function', 'noise', 'N_samples' or 'B_field', 'direction' depending on the type of data being loaded.
        """
        self.w = w
        self.tau = tau
        self.Nt = Nt
        self.finite_T = finite_T
        self.home_path = home_path
        self.mock_data = mock_data
        self.kwargs = {k: v for k, v in kwargs.items() if k in ['function', 'noise', 'N_samples', 'B_field', 'direction']}
        if self.finite_T:
            self.temp = "finite_T"
        else:
            self.temp = "zero_T"

    def initKernel(
            self,
            extractedQuantity: str,
            finiteT_kernel: bool,
            Nt: int,
            tau: np.ndarray,
            omega: np.ndarray
            ) -> np.ndarray:
        if finiteT_kernel:
            if extractedQuantity == "RhoOverOmega":
                K=KL_kernel_Omega(KL_kernel_Position_FiniteT, tau, omega, args=(1/Nt,))
            else:
                K=KL_kernel_Position_FiniteT(tau, omega, 1/Nt)
        else:
            if extractedQuantity == "RhoOverOmega":
                K=KL_kernel_Omega(KL_kernel_Position_Vacuum, tau, omega)
            else:
                K=KL_kernel_Position_Vacuum(tau, omega)
        return K

    def load_MEM(
            self,
            extr_Q: str,
            defmod: str,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # --- initialize kernel ---

        K = self.initKernel(extr_Q, self.finite_T, self.Nt, self.tau, self.w)

        # --- load the predicted spectral function ---

        if self.mock_data:
            predicted_spf = np.zeros((len(self.kwargs['noise']), len(self.w)))
            predicted_spf_bins = np.zeros((len(self.kwargs['noise']), self.kwargs['N_samples']-1, len(self.w)))
            spf_var = np.zeros((len(self.kwargs['noise']), len(self.w)))
            for i in range(len(self.kwargs['noise'])):
                spf_data = np.loadtxt(f"{self.home_path}/spectral_reconstruction_methods/mem/outputs/{extr_Q}_{self.temp}_prior_{defmod}_mock_corr_{self.kwargs['function']}_Nt{self.Nt}_noise{self.kwargs['noise'][i]}.dat")
                print("Loaded from:",  f"{self.home_path}/spectral_reconstruction_methods/mem/outputs/{extr_Q}_{self.temp}_prior_{defmod}_mock_corr_{self.kwargs['function']}_Nt{self.Nt}_noise{self.kwargs['noise'][i]}.dat")
                if self.temp == "finite_T":
                    predicted_spf[i][:] = spf_data[:,1]/self.Nt
                    spf_var[i][:] = spf_data[:,2]/self.Nt
                    for j in range(self.kwargs['N_samples']-1):
                        predicted_spf_bins[i][j][:] = spf_data[:,j+3]/self.Nt
                else:
                    predicted_spf[i][:] = spf_data[:,1]
                    spf_var[i][:] = spf_data[:,2]
                    for j in range(self.kwargs['N_samples']-1):
                        predicted_spf_bins[i][j][:] = spf_data[:,j+3]

        else:
            spf_data = np.loadtxt(f"{self.home_path}/spectral_reconstruction_methods/mem/outputs/{extr_Q}_{self.temp}_prior_{defmod}_data_wilson_emconduc_48_{self.Nt}_b6.872_B{self.kwargs['B_field']}_{self.kwargs['direction']}.txt")
            print("Loaded from:",  f"{self.home_path}/spectral_reconstruction_methods/mem/outputs/{extr_Q}_{self.temp}_prior_{defmod}_data_wilson_emconduc_48_{self.Nt}_b6.872_B{self.kwargs['B_field']}_{self.kwargs['direction']}.txt")
            predicted_spf = spf_data[:,1]
            spf_var = np.sqrt(spf_data[:,2])
            

        # --- calculate the output corr ---

        if self.mock_data:
            G_output = np.zeros((len(self.kwargs['noise']), self.Nt))
            G_output_bins = np.zeros((len(self.kwargs['noise']), self.kwargs['N_samples']-1, self.Nt))
            G_output_err = np.zeros((len(self.kwargs['noise']), self.Nt))
            for i in range(len(self.kwargs['noise'])):
                if temp == "finite_T":
                    G_output[i][:] = Di(K, predicted_spf[i][:]*self.Nt, self.w[1]-self.w[0])
                    for j in range(self.kwargs['N_samples']-1):
                        G_output_bins[i][j][:] = Di(K, predicted_spf_bins[i][j][:]*self.Nt, self.w[1] - self.w[0])
                else:
                    G_output[i][:] = Di(K, predicted_spf[i][:], self.w[1]-self.w[0])
                    for j in range(self.kwargs['N_samples']-1):
                        G_output_bins[i][j][:] = Di(K, predicted_spf_bins[i][j][:], self.w[1] - self.w[0])
                G_output_err[i][:] = np.sqrt((self.kwargs['N_samples']-1) / self.kwargs['N_samples'] * np.sum((G_output_bins[i] - G_output[i]) ** 2, axis=0))
        else:
            G_output = Di(K, predicted_spf, self.w[1]-self.w[0])
            G_output_err = Di(K, spf_var, self.w[1]-self.w[0])

        # --- calculate the default model for MEM ---

        if self.mock_data:
            default_model = np.ones(len(self.w))
            data = np.loadtxt(f"{self.home_path}/mock-data-main/{self.temp}/uncorrelated_data/BW/exact_spectral_function_BW.dat")
            omega_file = data[:, 0]
            exact = data[:, 1]
            if defmod == "constant":
                m_0 = np.trapz(exact, x=omega_file) / (omega_file[-1] - omega_file[0])
                if extr_Q == "RhoOverOmega":
                    # Avoid division by zero at w=0
                    default_model = np.ones(len(self.w)) * m_0
                else:
                    # Simple constant
                    default_model = np.ones(len(self.w)) * max(m_0, 1e-10)
            if defmod == "quadratic":
                # Normalize to match integral
                m_0 = np.trapz(exact, x=omega_file) / np.trapz(omega_file**2, x=omega_file)
                default_model = m_0 * self.w**2
                if extr_Q != "RhoOverOmega":
                    # Protect against zero at w=0
                    default_model[self.w == 0] = m_0 * 1e-10
                else:
                    default_model[self.w == 0] = m_0
            if defmod == "file":
                data = np.loadtxt("/mnt/c/Users/chris/OneDrive/Desktop/unsupervised_results/UnsupAI_mock_corr_BW_Nt36_noise4_750.dat.txt")
                default_model = self.w*data[:, 1] # because recsults from unsupervised are rho/w
        else:
            default_model = np.ones(len(self.w))
            if defmod == "constant":
                default_model = np.ones(len(self.w)) * 1e-2
            if defmod == "quadratic":
                default_model = self.w**2
                default_model = np.maximum(default_model, 1e-10)
            #default_model = read_file_l1(f"{home_path}/finite_T_finite_B/unsupervised_ml/rhoOomegaB{B_field}_z_omegamax20_pts1k.txt")
        return predicted_spf, spf_var, G_output, G_output_err, default_model

    def load_BG(
            self,
            extr_Q: str,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # --- initialize the kernel ---

        K = self.initKernel(extr_Q, self.temp, self.Nt, self.tau, self.w)

        rec = "exp_sym_w"
        lamb = 0.9

        # --- load the predicted spectral function --- 
        
        predicted_spf = np.zeros((len(self.kwargs['noise']), len(self.w)))
        spf_var = np.zeros((len(self.kwargs['noise']), len(self.w)))
        for i in range(len(self.kwargs['noise'])):
            spf_data = np.loadtxt(f"{self.home_path}/proceedings/egarnacho/plots_zero_T/{self.kwargs['function']}/BG/nt{self.Nt}/noise{self.kwargs['noise'][i]}/{rec}/rho_ll{lamb}.txt")
            predicted_spf[i][:] = spf_data[:,1]
            spf_var[i][:] = spf_data[:,2]

        # --- calculate the output corr ---

        G_output = np.zeros((len(self.kwargs['noise']), self.Nt))
        G_output_err = np.zeros((len(self.kwargs['noise']), self.Nt))
        for i in range(len(self.kwargs['noise'])):
            G_output[i][:] = Di(K, predicted_spf[i][:], self.w[1]-self.w[0]) 
            G_output_err[i][:] = Di(K, spf_var[i][:], self.w[1]-self.w[0]) 
        
        return predicted_spf, spf_var, G_output, G_output_err

    def load_gaussian(
            self,
            extr_Q: str,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # --- initialize the kernel ---

        K = self.initKernel(extr_Q, self.temp, self.Nt, self.tau, self.w)

        # --- load the predicted spectral function --- (I only take half the values here - is this even correct?)
        
        predicted_spf = np.zeros((len(self.kwargs['noise']), len(self.w)*2))
        spf_var = np.zeros((len(self.kwargs['noise']), len(self.w)*2))
        for i in range(len(self.kwargs['noise'])):
            spf_data = np.loadtxt(f"{self.home_path}/proceedings/gaussian/gauss_specf.mock_corr_{self.kwargs['function']}_Nt{self.Nt}_noise{self.kwargs['noise'][i]}.dat")
            predicted_spf[i][:] = spf_data[:,1]
            spf_var[i][:] = spf_data[:,2]

        G_output = np.zeros((len(self.kwargs['noise']), self.Nt))
        G_output_err = np.zeros((len(self.kwargs['noise']), self.Nt))
        for i in range(len(self.kwargs['noise'])):
            G_data = np.loadtxt(f"{self.home_path}/proceedings/gaussian/gauss_corr.mock_corr_{self.kwargs['function']}_Nt{self.Nt}_noise{self.kwargs['noise'][i]}.dat")
            for j in range(self.Nt):
                if j <= self.Nt//2:
                    G_output[i][j] = G_data[j,1]
                    G_output_err[i][j] = G_data[j,2]
                else:
                    G_output[i][j] = G_data[self.Nt-j,1]
                    G_output_err[i][j] = G_data[self.Nt-j,2]

        return predicted_spf, spf_var, G_output, G_output_err

    def load_unsupervised(
            self,
            extr_Q: str,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # --- initialize the kernel ---

        K = self.initKernel(extr_Q, self.temp, self.Nt, self.tau, self.w)

        # --- load the predicted spectral function --- 
        
        predicted_spf = np.zeros(len(self.w))
        spf_var = np.zeros(len(self.w))
        spf_data = np.loadtxt(f"{self.home_path}/proceedings/Nt36A234andrho/dataforchristian/UnsupAI_mock_corr_{self.kwargs['function']}_Nt{self.Nt}_noise{self.kwargs['noise'][0]}.dat.txt")
        predicted_spf = spf_data[:,1]
        spf_var = spf_data[:,2]

        # --- calculate the output corr ---

        G_output = np.zeros(self.Nt)
        G_output_err = np.zeros(self.Nt)
        G_output = Di(K, predicted_spf, self.w[1]-self.w[0])
        G_output_err = Di(K, spf_var, self.w[1]-self.w[0])

        return predicted_spf, spf_var, G_output, G_output_err

    def load_supervised(
            self,
            extr_Q: str,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # --- initialize the kernel ---

        K = self.initKernel(extr_Q, self.temp, self.Nt, self.tau, self.w)

        # --- load the predicted spectral function --- 
        
        predicted_spf = np.zeros(len(self.w))
        spf_var = np.zeros(len(self.w))
        spf_data = np.loadtxt(f"supervised_ml/outputs/{extr_Q}_mock_corr_BW_Nt{self.Nt}_noise{self.kwargs['noise'][0]}.dat")
        predicted_spf = spf_data[:,1]
        spf_var = spf_data[:,2]

        # --- calculate the output corr ---

        G_output = np.zeros(self.Nt)
        G_output_err = np.zeros(self.Nt)
        G_output = Di(K, predicted_spf, self.w[1]-self.w[0])
        G_output_err = Di(K, spf_var, self.w[1]-self.w[0])

        return predicted_spf, spf_var, G_output, G_output_err
    
    def load_call(
            self,
            method: str,
            extr_Q: str,
            *args,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if method == "MEM":
            return self.load_MEM(extr_Q, *args)
        elif method == "BG":
            return self.load_BG(extr_Q)
        elif method == "Gaussian":
            return self.load_gaussian(extr_Q)
        elif method == "Unsupervised":
            return self.load_unsupervised(extr_Q)
        elif method == "Supervised":
            return self.load_supervised(extr_Q)
        else:
            raise ValueError(f"Method {method} not recognized. Please choose from 'MEM', 'BG', 'Gaussian', 'Unsupervised' or 'Supervised'.")

class plotting:
    def __init__(
            self,
            w: np.ndarray,
            tau: np.ndarray,
            extrQ: str,
            temp: bool,
            mock_data: bool,
            **kwargs,
            ) -> None:
        self.tau = tau
        self.w = w
        if extrQ == "RhoOverOmega": self.ylabel = r"$\rho (\omega)/ \omega$"
        else: self.ylabel = r"$\rho (\omega)$"
        if temp: self.wlabel = r"$\omega/T$"
        else: self.wlabel = r"$\omega$"
        self.mock_data = mock_data
        self.kwargs = {k: v for k, v in kwargs.items() if k in ['function', 'noise', 'N_samples', 'B_field', 'direction']}
        
    def plotting_spf(
            self,
            rho_input, 
            rho_learned
            ) -> None:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.w, rho_input, label='True ρ', color='cornflowerblue')
        plt.plot(self.w, rho_learned, label='Learned ρ', color='tomato')
        plt.legend()
        plt.title("Spectral Function divided by omega")

    def plotting_spf_corr(
            self, 
            rho_input, 
            rho_learned, 
            G_exact, 
            G_err, 
            G_learned
            ) -> None:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.w, rho_input, label='True ρ', color='cornflowerblue')
        plt.plot(self.w, rho_learned, label='Learned ρ', color='tomato')
        plt.legend()
        plt.title("Spectral Function")

        plt.subplot(1, 2, 2)
        plt.errorbar(self.tau,  G_exact, G_err, label='True G', color='cornflowerblue', capsize = 3, markeredgewidth = 1, elinewidth=1, fmt = 'x')
        plt.scatter(self.tau, G_learned, label='G from Learned ρ', marker = 'x', color='tomato')
        plt.yscale("log")
        plt.legend()
        plt.title("Correlator")

        plt.tight_layout()

    def plotting_spf_corr_loss(
            self, 
            train_losses, 
            val_losses, 
            rho_input, 
            rho_learned, 
            G_exact, 
            G_learned
            ) -> None:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(self.w, rho_input, label='True ρ', color='cornflowerblue')
        plt.plot(self.w, rho_learned, label='Learned ρ', color='tomato')
        plt.legend()
        plt.title("Spectral Function divided by omega")

        plt.subplot(1, 3, 2)
        plt.errorbar(self.tau,  G_exact, abs(G_exact - tf.squeeze(G_in)), label='True G', color='cornflowerblue', capsize = 3, markeredgewidth = 1, elinewidth=1, fmt = 'x')
        plt.scatter(self.tau, G_learned, label='G from Learned ρ', marker = 'x', color='tomato')
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

    def plotting_spf_loss(
            self, 
            rho_input, 
            rho_learned, 
            train_losses, 
            val_losses
            ) -> None:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.w, rho_input/self.w, label='True ρ', color='cornflowerblue')
        plt.plot(self.w, rho_learned, label='Learned ρ', color='tomato')
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

    def plotting_MEM(
            self,
            rho_input, 
            rho_learned, 
            rho_err, 
            G_exact, 
            G_err, 
            G_learned, 
            G_learned_err, 
            d_model
            ) -> None:
        colors = ['tomato', 'mediumseagreen', 'turquoise', 'deepskyblue', 'mediumslateblue', 'violet', 'plum', 'pink']
        #if temp == "finite_T":
        #    w = w*Nt
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        if self.mock_data:
            noise = self.kwargs['noise']
            #plt.plot(self.w, np.squeeze(rho_input), label='True ρ', color='cornflowerblue')
            for i in range(len(noise)):
                plt.plot(self.w, np.squeeze(rho_learned[i][:]), label=f"Noise{noise[i]}", color=colors[i])
                plt.fill_between(self.w, np.squeeze(rho_learned[i][:]) - np.squeeze(rho_err[i][:]), np.squeeze(rho_learned[i][:]) + np.squeeze(rho_err[i][:]), color = colors[i], alpha = 0.5)
        else:
            plt.plot(self.w, rho_learned, label=f"Learned ρ", color="tomato")
            plt.fill_between(self.w, rho_learned-rho_err, rho_learned+rho_err, color = "tomato", alpha =0.5)
        plt.plot(self.w, d_model, label = 'Prior', color = 'black', linestyle = 'dashed', alpha = 0.6)
        plt.legend()
        plt.ylabel(self.ylabel)
        plt.xlabel(self.wlabel)
        #plt.ylim(-0.1,50)
        plt.title("Spectral Function")

        plt.subplot(1, 2, 2)
        if self.mock_data:
            noise = self.kwargs['noise']
            for i in range(len(noise)):
                plt.errorbar(self.tau,  np.squeeze(G_exact[i][:]), yerr=np.squeeze(G_err[i][:]), label='True G', color='cornflowerblue', capsize = 3, markeredgewidth = 1, elinewidth=1, fmt = 'x')
        else:
            plt.errorbar(self.tau, G_exact, yerr=G_err, label='True G', color='cornflowerblue', capsize = 3, markeredgewidth = 1, elinewidth=1, fmt = 'x')
        if self.mock_data:
            noise = self.kwargs['noise']
            for i in range(len(noise)):
                plt.errorbar(self.tau, G_learned[i][:], yerr=G_learned_err[i][:], label=f"Noise{noise[i]}", fmt = 'x', color=colors[i], capsize = 3, markeredgewidth = 1, elinewidth=1)
        else:
            plt.errorbar(self.tau, G_learned, yerr=G_learned_err, label=f"Learned G", fmt = 'x', color="tomato", capsize = 3, markeredgewidth = 1, elinewidth=1)
        plt.yscale("log")
        plt.ylabel(r"$G(\tau)$")
        plt.xlabel(r"$\tau$")
        plt.legend()
        plt.title("Correlator")

        plt.tight_layout()

    def plotting_BG_Gauss(
            self,
            rho_input, 
            rho_learned, 
            rho_err,
            G_exact,
            G_err, 
            G_learned, 
            G_learned_err
            ) -> None:
        colors = ['tomato', 'mediumseagreen', 'turquoise', 'deepskyblue', 'mediumslateblue', 'violet', 'plum', 'pink']
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.w, rho_input, label='True ρ', color='cornflowerblue')
        if self.mock_data:
            noise = self.kwargs['noise']
            for i in range(len(noise)):
                plt.plot(self.w, rho_learned[i][:], label=f"Noise{i+2}", color=colors[i])
                plt.fill_between(self.w, rho_learned[i][:] - rho_err[i][:], rho_learned[i][:] + rho_err[i][:], color = colors[i], alpha = 0.5)
        plt.legend()
        plt.ylim(0,1.5)
        plt.ylabel(self.ylabel)
        plt.xlabel(self.wlabel)
        plt.title("Spectral Function")

        plt.subplot(1, 2, 2)
        plt.errorbar(self.tau,  G_exact, yerr=G_err, label='True G', color='cornflowerblue', capsize = 3, markeredgewidth = 1, elinewidth=1, fmt = 'x')
        if self.mock_data:
            noise = self.kwargs['noise']
            for i in range(len(noise)):
                plt.errorbar(self.tau, G_learned[i][:], yerr=G_learned_err[i][:], label=f"Noise{i+2}", fmt = 'x', color=colors[i], capsize = 3, markeredgewidth = 1, elinewidth=1)
        plt.yscale("log")
        plt.ylabel(r"$G(\tau)$")
        plt.xlabel(r"$\tau$")
        plt.legend()
        plt.title("Correlator")

        plt.tight_layout()

    def plotting_ml(
            self,
            rho_learned, 
            rho_err, 
            G_exact, 
            G_err, 
            G_learned, 
            G_learned_err
            ) -> None:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.w, np.squeeze(rho_learned), label="Reconstructed ρ", color="tomato")
        plt.fill_between(self.w, np.squeeze(rho_learned) - np.squeeze(rho_err), np.squeeze(rho_learned) + np.squeeze(rho_err), color = "tomato", alpha = 0.5)
        plt.legend()
        plt.ylabel(self.ylabel)
        plt.xlabel(self.wlabel)
        plt.title("Spectral Function")

        plt.subplot(1, 2, 2)
        plt.errorbar(self.tau,  np.squeeze(G_exact), yerr=np.squeeze(G_err), label='True G', color='cornflowerblue', capsize = 3, markeredgewidth = 1, elinewidth=1, fmt = 'x')
        plt.errorbar(self.tau, G_learned, yerr=G_learned_err, label="Reconstructed G", fmt = 'x', color="tomato", capsize = 3, markeredgewidth = 1, elinewidth=1)
        plt.yscale("log")
        plt.ylabel(r"$G(\tau)$")
        plt.xlabel(r"$\tau$")
        plt.legend()
        plt.title("Correlator")

        plt.tight_layout()

    def comparing_mock(
            self,
            rho_input, 
            rho_unsup, 
            rho_bg, 
            rho_mem, 
            rho_gauss,
            rho_err_unsup, 
            rho_err_bg, 
            rho_err_mem,
            rho_err_gauss,
            G_exact,
            G_err,
            G_l_unsup,
            G_l_bg,
            G_l_mem,
            G_l_gauss,
            G_err_unsup,
            G_err_bg,
            G_err_mem,
            G_err_gauss,
            ) -> None:

        omega_gauss = np.linspace(0,2,1000)
        omega=self.w[1:]
        rho_input = rho_input[1:]
        rho_unsup = rho_unsup[1:]*omega
        rho_err_unsup = rho_err_unsup[1:]*omega
        rho_bg = rho_bg[1:]
        rho_err_bg = rho_err_bg[1:]
        rho_mem = rho_mem[1:]
        rho_err_mem = rho_err_mem[1:]

        rho_gauss *= omega_gauss
        rho_err_gauss *= omega_gauss

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(omega, rho_input, label = "Input ρ", color = "k")
        plt.fill_between(omega, rho_unsup - rho_err_unsup, rho_unsup + rho_err_unsup, color = "tomato", alpha = 0.5, label="Unsupervised")
        plt.fill_between(omega, rho_bg - rho_err_bg, rho_bg + rho_err_bg, color = "mediumseagreen", alpha = 0.5, label = "BG")
        plt.fill_between(omega, rho_mem - rho_err_mem, rho_mem + rho_err_mem, color = "violet", alpha = 0.5, label = "MEM")
        plt.fill_between(omega_gauss, rho_gauss - rho_err_gauss, rho_gauss + rho_err_gauss, color = "cornflowerblue", alpha = 0.5, label = "Gaussian")

        plt.legend()
        #plt.ylim(0,20)
        plt.ylim(0,4)
        plt.ylabel(self.ylabel)
        #plt.ylabel(r"$\rho (\omega) / \omega$")
        plt.xlabel(self.wlabel)
        if self.mock_data:
            noise = self.kwargs['noise']
            plt.title(f"Spectral Function for noise level A={noise[0]}")

        offset = 0.007
        plt.subplot(1, 2, 2)
        plt.errorbar(self.tau,  G_exact, yerr=G_err, label='True G', color='k', capsize = 3, markeredgewidth = 1, elinewidth=1, fmt = 'o', markersize=4)
        plt.errorbar(self.tau, G_l_unsup+offset, yerr=G_err_unsup, label="Unsupervised", color="tomato", fmt = 'x', capsize = 3, markeredgewidth = 1, elinewidth=1, markersize=4, mfc='none')
        plt.errorbar(self.tau, G_l_bg, yerr=G_err_bg, label="BG", color="mediumseagreen", fmt = '^', capsize = 3, markeredgewidth = 1, elinewidth=1, markersize=4, mfc='none')
        plt.errorbar(self.tau, G_l_mem-offset, yerr=G_err_mem, label="MEM", color="violet", fmt = 'v', capsize = 3, markeredgewidth = 1, elinewidth=1, markersize=4, mfc='none')
        plt.errorbar(self.tau, G_l_gauss+offset, yerr=G_err_gauss, label="Gaussian", color="cornflowerblue", fmt = 'd', capsize = 3, markeredgewidth = 1, elinewidth=1, markersize=4, mfc='none')
        plt.yscale("log")
        plt.ylabel(r"$G(\tau)$")
        plt.xlabel(r"$\tau$/a")
        plt.legend()
        if self.mock_data:
            noise = self.kwargs['noise']
            plt.title(f"Correlator for noise level A={noise[0]}")

        plt.tight_layout()

    def mem_zoomed(
            self,
            w, 
            rho_learned, 
            rho_err, 
            G_exact, 
            G_err, 
            G_learned, 
            G_learned_err, 
            d_model
            ) -> None:
        w_zoom = w[w <= 1.879]
        rho_learned_zoom = rho_learned[:len(w_zoom)]
        rho_err_zoom = rho_err[:len(w_zoom)]
        d_model_zoom = d_model[:len(w_zoom)]
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(w_zoom, rho_learned_zoom, label='Learned ρ', color='tomato')
        plt.fill_between(w_zoom, rho_learned_zoom - rho_err_zoom, rho_learned_zoom + rho_err_zoom, color = "tomato", alpha = 0.5)
        plt.plot(w_zoom, d_model_zoom, label = 'Prior', color = 'black', linestyle = 'dashed', alpha = 0.6)
        plt.legend()
        plt.ylabel(self.ylabel)
        plt.xlabel(self.wlabel)
        plt.title("Zoomed Spectral Function")
        plt.subplot(1, 2, 2)
        plt.errorbar(self.tau,  G_exact, yerr=G_err, label='True G', color='cornflowerblue', capsize = 3, markeredgewidth = 1, elinewidth=1, fmt = 'x')
        plt.errorbar(self.tau, G_learned, yerr=G_learned_err, label=f"Learned G", fmt = 'x', color="tomato", capsize = 3, markeredgewidth = 1, elinewidth=1)
        plt.yscale("log")
        plt.ylabel(r"$G(\tau)$")
        plt.xlabel(r"$\tau$")
        plt.legend()
        plt.title("Correlator")
        plt.tight_layout()

    def integrate_spatial_correlator(
            self,
            spectral_function, 
            omega_max
            ) -> float:
        omega_range = np.linspace(0,omega_max)
        spatial_correlator = np.trapz(spectral_function, x=omega_range)
        return spatial_correlator

    def compare_spatial_correlators(
            self,
            spectral_function
            ) -> np.ndarray:
        w_maxes = np.array([10,20,30])
        spatial_correlators = []
        for i in range(len(w_maxes)):
            spatial_correlators.append(self.integrate_spatial_correlator(spectral_function, w_maxes[i]))
        print(spatial_correlators)
        return np.array(spatial_correlators)

    def plot_spatial_correlators(
            self,
            spectral_function
            ) -> None:
        spatial_correlators = self.compare_spatial_correlators(spectral_function)
        plt.figure(1)
        for i in range(3):
            plt.scatter(spatial_correlators[i], label=f'Omega_max = {i*10}')
        plt.savefig("Spatial_correlator_comparison.jpg")

class latticedata:
    def __init__(
            self,
            Nb: int,
            direction: str,
            epsilon: float,
            Ns: int,
            Nt: int,
            ) -> None:
        self.Nb = Nb
        self.epsilon = epsilon
        self.direction = direction
        self.Ns = Ns
        self.Nt = Nt
    
    def initialize_my_color_setup(
            self,
            fraction_of_pagesize, 
            size_1 = 1,
            size_2 = 1
            ) -> None:
        page_size_in_inches = [size_1*5.3, size_2*5.3] # default 5.3
        #fraction_of_pagesize = 0.6 # default 0.7
        plt.rcParams["figure.figsize"] = (fraction_of_pagesize * page_size_in_inches[0], fraction_of_pagesize * page_size_in_inches[1])
        plt.rcParams["font.size"] = 8
        plt.rcParams["errorbar.capsize"] = 3 # default 5
        plt.rcParams["lines.markersize"] = 3 # default 4
        plt.rcParams["scatter.marker"] = "s" # default "s"
        plt.rcParams["lines.linewidth"] = 1

    def set_color_palette_using_keys(
            self,
            pal,
            vector
            ) -> None:
        color_dict = dict()
        my_palette = mpl.colormaps[pal]
        index_vec = np.linspace(0,1,len(vector))
        return {vector[n]:my_palette(index_vec[n]) for n in range(len(vector))}

    def compare_elec_conduct_different_methods(
            self,
            ) -> None:

        plot_name = plt.figure("compare electric conductivities")

        methods = {
            "Gaussian":"gaussian",
            "Multipoint":"multipoint",
            "Unsupervised learning":"unsupervised", 
            "MEM constant 1e-2":"mem_constant_1e-2", 
            "MEM constant 3e-2":"mem_constant_3e-2",
            "MEM quadratic":"mem_quadratic", 
            "MEM linear":"mem_linear",
            "MEM neural":"mem_neural"}

        temp = self.set_color_palette_using_keys("Set1",np.arange(9))
        my_palette = dict()
        my_palette["Gaussian"] = temp[0]
        my_palette["Multipoint"] = temp[1]
        my_palette["Unsupervised learning"] = temp[2]
        my_palette["MEM quadratic"] = temp[7]
        my_palette["MEM constant 1e-2"] = temp[8]
        my_palette["MEM linear"] = temp[6]
        my_palette["MEM constant 3e-2"] = temp[5]
        my_palette["MEM neural"] = temp[4]

        ax = plt.subplot(111)

        for m in methods.keys():
            input_file = "../dat/finite_T_reconstructions/%s/econduct_B_%s.dat"%(methods[m],self.direction)
            if os.path.isfile(input_file):
                B_vec, conduct_vec, conduct_err = np.loadtxt(input_file,usecols=(0,1,2),unpack=True)

                fac = 1./conduct_vec[0]

                B_vec = 6*np.pi*B_vec*self.Nt**2/self.Ns**2
                conduct_vec = fac * conduct_vec
                conduct_err = fac * conduct_err

                plt.errorbar(B_vec[1:],conduct_vec[1:],yerr=conduct_err[1:],fmt="s",capsize=4,color=my_palette[m],label="%s"%m)

        #plt.text(0.5, 0.5, "PRELIMINARY", color="green", alpha=0.15,transform=ax.transAxes, fontsize=26,horizontalalignment="center")

        #plt.title(r"$48^3\times16$",x=0.15,y=0.58,horizontalalignment="center")
        plt.legend(loc="upper left")
        plt.xlabel("$eB/T^2$")
        plt.ylabel("$\sigma_%s(B)/\sigma_%s(B=0)$"%(self.direction,self.direction))
        plt.xlim(0,27)
        plt.ylim(0,5)
        plt.legend(loc="upper left")
        plt.subplots_adjust(left=0.17,right=0.99,bottom=0.13,top=0.98)

        plt.savefig("conduct_%s_variousmethods.jpg"%self.direction,dpi=500)

    def compare_spectral_function_different_methods(
            self,
            ) -> None:

        plot_name = plt.figure("compare spectral function")

        ax = plt.subplot()

        temp = self.set_color_palette_using_keys("Set1",np.arange(9))
        my_palette = dict()
        my_palette["gauss"] = temp[0]
        my_palette["multi"] = temp[1]
        my_palette["unsup"] = temp[2]
        my_palette["mem_const_1e-2"] = temp[8]
        my_palette["mem_const_3e-2"] = temp[5]
        my_palette["mem_quad"] = temp[7]
        my_palette["mem_lin"] = temp[6]
        my_palette["mem_neural"] = temp[4]

        # Gaussian results
        input_file = "../gaussian/emconduc_recs/gauss_specf.data_wilson_emconduc_48_16_b6.872_B%i_%s.txt"%(self.Nb,self.direction)
        if os.path.isfile(input_file):
            w_vec, rho_gaussian, rho_gaussian_err = np.loadtxt(input_file,usecols=(0,1,2),unpack=True)
            plt.fill_between(self.Nt*w_vec,rho_gaussian-rho_gaussian_err,rho_gaussian+rho_gaussian_err,alpha=0.5,color=my_palette["gauss"],edgecolor="black",label="Gaussian")

        # HLT results
        """
        input_file = "data/hlt/hlt_specf_eps%.2f.data_wilson_emconduc_48_16_b6.872_B%i_%s.txt"%(epsilon,Nb,direction)
        if os.path.isfile(input_file):
            w_vec, rho_vec, rho_err = np.loadtxt(input_file,usecols=(0,1,2),unpack=True)
            plt.fill_between(Nt*w_vec,rho_vec-rho_err,rho_vec+rho_err,alpha=0.5,color=my_palette["hlt"],edgecolor="black",label=r"HLT (smearing = %.2f)"%epsilon)
        """

        # MEM results
        input_file_1e2_const = "../mem/outputs/emconduc_recs/constant_1e-2_prior/RhoOverOmega_finite_T_prior_constant_data_wilson_emconduc_48_16_b6.872_B%i_%s.txt"%(self.Nb,self.direction)
        input_file_3e2_const = "../mem/outputs/emconduc_recs/constant_3e-2_prior/RhoOverOmega_finite_T_prior_constant_data_wilson_emconduc_48_16_b6.872_B%i_%s.txt"%(self.Nb,self.direction)
        input_file_quad = "../mem/outputs/emconduc_recs/quadratic_prior/RhoOverOmega_finite_T_prior_quadratic_data_wilson_emconduc_48_16_b6.872_B%i_%s.txt"%(self.Nb,self.direction)
        input_file_lin = "../mem/outputs/emconduc_recs/linear_prior/RhoOverOmega_finite_T_prior_linear_data_wilson_emconduc_48_16_b6.872_B%i_%s.txt"%(self.Nb,self.direction)
        input_file_neural = "../mem/outputs/emconduc_recs/neural_prior/RhoOverOmega_finite_T_prior_file_data_wilson_emconduc_48_16_b6.872_B%i_%s.txt"%(self.Nb,self.direction)
        
        # MEM const 1e-2 results        
        if os.path.isfile(input_file_1e2_const):
            w_vec, rho_vec, rho_err = np.loadtxt(input_file_1e2_const,usecols=(0,1,2),unpack=True)
            w_mem = w_vec[w_vec <= 1.879]
            rho_mem_1e2 = rho_vec[:len(w_mem)]
            rho_err_mem_1e2 = rho_err[:len(w_mem)]
            plt.fill_between(self.Nt*w_mem,rho_mem_1e2-rho_err_mem_1e2,rho_mem_1e2+rho_err_mem_1e2,alpha=0.5,color=my_palette["mem_const_1e-2"],edgecolor="black",label="MEM 1e-2 const prior")
        
        # MEM const 3e-2 results
        if os.path.isfile(input_file_3e2_const):
            w_vec, rho_vec, rho_err = np.loadtxt(input_file_3e2_const,usecols=(0,1,2),unpack=True)
            w_mem = w_vec[w_vec <= 1.879]
            rho_mem_3e2 = rho_vec[:len(w_mem)]
            rho_err_mem_3e2 = rho_err[:len(w_mem)]
            plt.fill_between(self.Nt*w_mem,rho_mem_3e2-rho_err_mem_3e2,rho_mem_3e2+rho_err_mem_3e2,alpha=0.5,color=my_palette["mem_const_3e-2"],edgecolor="black",label="MEM 3e-2 const prior")
        
        # MEM quadratic results
        if os.path.isfile(input_file_quad):
            w_vec, rho_vec, rho_err = np.loadtxt(input_file_quad,usecols=(0,1,2),unpack=True)
            w_mem_quad = w_vec[w_vec <= 1.879]
            rho_mem_quad = rho_vec[:len(w_mem_quad)]
            rho_err_mem_quad = rho_err[:len(w_mem_quad)]
            plt.fill_between(self.Nt*w_mem_quad,rho_mem_quad-rho_err_mem_quad,rho_mem_quad+rho_err_mem_quad,alpha=0.5,color=my_palette["mem_quad"],edgecolor="black",label="MEM quadratic prior")

        # MEM linear results
        if os.path.isfile(input_file_lin):
            w_vec, rho_vec, rho_err = np.loadtxt(input_file_lin,usecols=(0,1,2),unpack=True)
            w_mem_lin = w_vec[w_vec <= 1.879]
            rho_mem_lin = rho_vec[:len(w_mem_lin)]
            rho_err_mem_lin = rho_err[:len(w_mem_lin)]
            plt.fill_between(self.Nt*w_mem_lin,rho_mem_lin-rho_err_mem_lin,rho_mem_lin+rho_err_mem_lin,alpha=0.5,color=my_palette["mem_lin"],edgecolor="black",label="MEM linear prior")

        # MEM neural results
        if os.path.isfile(input_file_neural):
            w_vec, rho_vec, rho_err = np.loadtxt(input_file_neural,usecols=(0,1,2),unpack=True)
            w_mem_neural = w_vec[w_vec <= 1.879]
            rho_mem_neural = rho_vec[:len(w_mem_neural)]
            rho_err_mem_neural = rho_err[:len(w_mem_neural)]
            plt.fill_between(self.Nt*w_mem_neural,rho_mem_neural-rho_err_mem_neural,rho_mem_neural+rho_err_mem_neural,alpha=0.5,color=my_palette["mem_neural"],edgecolor="black",label="MEM neural prior")

        # Multipoint results
        input_file = "../dat/finite_T_reconstructions/multipoint/econduct_B_%s.dat"%self.direction
        if os.path.isfile(input_file):
            B_vec, conduct_vec, conduct_err = np.loadtxt(input_file,usecols=(0,1,2),unpack=True)

            conduct_dict = {Nb:conduct for Nb,conduct in [*zip(B_vec,conduct_vec)]}
            conduct_err_dict = {Nb:err for Nb,err in [*zip(B_vec,conduct_err)]}

            plt.errorbar([0],[conduct_dict[self.Nb]],yerr=[conduct_err_dict[self.Nb]],fmt="s",capsize=4,color=my_palette["multi"],label="Multipoint")

        # Unsupervised ML results
        if self.direction == "x":
            input_file = "../neuralFit/outputs/emconduc_recs/rhoOomegaB%i_x_delomega02783_pts500.txt"%self.Nb
        else:
            input_file = "../neuralFit/outputs/emconduc_recs/rho_over_omega_B%i_z.txt"%self.Nb
        if os.path.isfile(input_file):
            w_vec, rho_unsupervised, rho_unsupervised_err = np.loadtxt(input_file,usecols=(0,1,2),unpack=True)
            plt.fill_between(self.Nt*w_vec,rho_unsupervised-rho_unsupervised_err,rho_unsupervised+rho_unsupervised_err,alpha=0.7,color=my_palette["unsup"],edgecolor="black",label=r"Unsupervised learning")

        # Correlator
        input_file = "../dat/data_wilson_emconduc_48_16_b6.872_B%i_%s.txt"%(self.Nb,self.direction)
        if os.path.isfile(input_file):
            tau, corr = np.loadtxt(input_file, usecols = (0,1), unpack=True)
            corr_sum = sum(corr)

        #plt.text(0.5, 0.5, "PRELIMINARY", color="green", alpha=0.15,transform=ax.transAxes, fontsize=26,horizontalalignment="center")
        #plt.title(r"$48^3\times16$",x=0.15,y=0.62,horizontalalignment="center")
        plt.legend(loc="upper left")
        plt.xlabel(r"$\omega/T$")
        plt.ylabel(r"$\rho(\omega)/\omega$")
        plt.xlim(-2,35)
        plt.ylim(0,0.16)
        plt.legend(loc="upper left")
        plt.subplots_adjust(left=0.17,right=0.99,bottom=0.13,top=0.98)
        plt.savefig("spec_func_variousmethods_%s.jpg"%self.direction,dpi=500)
        plt.close(plot_name)

        def integrate_spatial_correlator(
                spectral_function, 
                omega_max
                ):
            for i in range(len(spectral_function)):
                if self.Nt*w_vec[i] == omega_max:
                    cutoff_index = i
                    break
            else:
                cutoff_index = len(spectral_function)

            omega_range = np.linspace(0,omega_max, cutoff_index)
            spatial_correlator = np.trapz(spectral_function[:cutoff_index], x=omega_range)
            return spatial_correlator

        def compare_spatial_correlators(
                spectral_function
                ):
            w_maxes = np.array([10,20,30])
            spatial_correlators = []
            for i in range(len(w_maxes)):
                spatial_correlators.append(integrate_spatial_correlator(spectral_function, w_maxes[i]))
            print(spatial_correlators)
            return np.array(spatial_correlators)

        def plot_spatial_correlators(
                corr_sum, 
                spf_gaussian, 
                spf_unsupervised, 
                spf_mem_1e2, 
                spf_mem_3e2, 
                spf_mem_quad, 
                spf_mem_lin,
                spf_mem_neural
                ):
            print(len(spf_gaussian), len(spf_unsupervised), len(spf_mem_1e2), len(spf_mem_3e2), len(spf_mem_quad), len(spf_mem_lin), len(spf_mem_neural))
            spatial_correlators_gaussian = compare_spatial_correlators(spf_gaussian)
            spatial_correlators_unsupervised = compare_spatial_correlators(spf_unsupervised)
            spatial_correlators_mem_1e2 = compare_spatial_correlators(spf_mem_1e2)
            spatial_correlators_mem_3e2 = compare_spatial_correlators(spf_mem_3e2)
            spatial_correlators_mem_quad = compare_spatial_correlators(spf_mem_quad)
            spatial_correlators_mem_lin = compare_spatial_correlators(spf_mem_lin)
            spatial_correlators_mem_neural = compare_spatial_correlators(spf_mem_neural)
            plt.figure(figsize=(6,5))
            print(corr_sum)
            plt.axhline(corr_sum, label="Sum of correlator")
            plt.scatter([10,20,30], spatial_correlators_gaussian, label="Gaussian", marker = 'x', color=my_palette["gauss"])
            plt.scatter([10,20,30], spatial_correlators_unsupervised, label="Unsupervised", marker = 'o', color=my_palette["unsup"])
            plt.scatter([10,20,30], spatial_correlators_mem_1e2, label="MEM 1e-2 constant", marker = 's', color=my_palette["mem_const_1e-2"])
            plt.scatter([10,20,30], spatial_correlators_mem_3e2, label="MEM 3e-2 constant", marker = 's', color=my_palette["mem_const_3e-2"])
            plt.scatter([10,20,30], spatial_correlators_mem_quad, label="MEM quadratic", marker = 'd', color=my_palette["mem_quad"])
            plt.scatter([10,20,30], spatial_correlators_mem_lin, label="MEM linear", marker = '>', color=my_palette["mem_lin"])
            plt.scatter([10,20,30], spatial_correlators_mem_neural, label="MEM neural", marker = '<', color=my_palette["mem_neural"])
            plt.xlabel(r"$\omega_{max}/T$")
            plt.ylabel(r"$\int_0^{\omega_{max}} \frac{\rho(\omega)}{\omega} d\omega$")
            plt.title("Spatial correlator as a function of spectral function cutoff")
            plt.legend()
            plt.savefig("Spatial_correlator_comparison_B%i_direction_%s.jpg"%(self.Nb,self.direction), dpi=500)

        #plot_spatial_correlators(corr_sum, rho_gaussian, rho_unsupervised, rho_mem_1e2, rho_mem_3e2, rho_mem_quad, rho_mem_lin, rho_mem_neural)
    

def main():
    lattice_data = True

    if lattice_data:
        latdat = latticedata(6, "x", 0.4, 48, 16)
        latdat.initialize_my_color_setup(0.6)

        latdat.compare_elec_conduct_different_methods()

        latdat.compare_spectral_function_different_methods()

        plt.show()

    else:
        #losshistory = False
        comparison = True # to be implemented: whether to compare the methods in one plot or just one method at a time

        home_path = "/mnt/c/Users/chris/OneDrive/Desktop" # home path to the data, should contain the "mock-data-main" folder with the mock data and the "spectral_reconstruction_methods" folder
        method = "Unsupervised" # method to load and plot, can be "MEM", "BG", "Gaussian", "Unsupervised" or "Supervised"
        defmod = "quadratic" # default model for MEM, only relevant if method == "MEM", can be "constant", "quadratic" or "file"

        finite_T = False # Finite T or zero T kernel
        temp = "finite_T" if finite_T else "zero_T"
        extr_Q = "Rho" # Rho or RhoOverOmega, might differ for different reconstruction methods

        Nt = 36 # number of points in the temporal direction

        mock_data = True # whether to use mock data or real lattice data
        noise = [4] # [2,3,4] # noise levels to compare in the plots
        N_samples = 10 # number of jackknife samples used in the reconstructions

        B_field = 12 # only relevant for finite T, finite B dataset
        direction = "z" # x or z
        function = "BW"  # functions for the mock data, currently only BW implemented
        """if losshistory:
        train_loss_history_data = read_file("/home/Christian/Desktop/spec_rec_methods/supervised_ml/outputs/RhoOverOmega_mock_BW_Nt16_noise4_rec_s1e-1_l21e-2_b256_conv.txt.trainloss.dat")
        val_loss_history_data = read_file("/home/Christian/Desktop/spec_rec_methods/supervised_ml/outputs/RhoOverOmega_mock_BW_Nt16_noise4_rec_s1e-1_l21e-2_b256_conv.txt.valloss.dat")"""

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
            w = np.linspace(0,30,500)
            true_spf = np.zeros(len(w))
            G_input, G_input_err = np.zeros(Nt),np.zeros(Nt)
            G_input_data = np.loadtxt(f"{home_path}/spectral_reconstruction_methods/dat/data_wilson_emconduc_48_{Nt}_b6.872_B{B_field}_{direction}.txt")
            tau = G_input_data[:,0]
            G_input = G_input_data[:,1]
            G_input_err = G_input_data[:,2]

        ld = loading(w, tau, Nt, finite_T, home_path, mock_data, noise = noise, function = function, N_samples = N_samples)
        predicted_spf_mem, spf_var_mem, G_output_mem, G_output_err_mem, default_model = ld.load_call("MEM", extr_Q, defmod)
        predicted_spf_bg, spf_var_bg, G_output_bg, G_output_err_bg = ld.load_call("BG", extr_Q)
        predicted_spf_gauss, spf_var_gauss, G_output_gauss, G_output_err_gauss = ld.load_call("Gaussian", extr_Q)
        predicted_spf_unsup, spf_var_unsup, G_output_unsup, G_output_err_unsup = ld.load_call("Unsupervised", extr_Q)

        plot = plotting(w, tau, extr_Q, finite_T, mock_data, noise = noise, function = function, N_samples = N_samples, B_field = B_field, direction = direction)

        """if losshistory:
            plotting_spf_loss(true_spf, predicted_spf, train_loss_history_data, val_loss_history_data)
        else:"""
        if comparison:
            plot.comparing_mock(true_spf, predicted_spf_unsup, predicted_spf_bg[0][:], predicted_spf_mem[0][:], predicted_spf_gauss[0][:], spf_var_unsup, spf_var_bg[0][:], spf_var_mem[0][:], spf_var_gauss[0][:], 
            G_input[0][:], G_input_err[0][:], G_output_unsup, G_output_bg[0][:], G_output_mem[0][:], G_output_gauss[0][:], G_output_err_unsup, G_output_err_bg[0][:], G_output_err_mem[0][:], G_output_err_gauss[0][:])
        else:
            #plotting_BG_Gauss(true_spf, predicted_spf, spf_var, G_input[0][:], G_input_err[0][:], G_output, G_output_err)
            #plotting_MEM(w, true_spf, predicted_spf_mem, spf_var_mem, G_input, G_input_err, G_output_mem, G_output_err_mem, default_model)
            plot.mem_zoomed(w, predicted_spf_mem, spf_var_mem, G_input, G_input_err, G_output_mem, G_output_err_mem, default_model)

        if mock_data:
            if method == "mem":
                if len(noise) > 1:
                    plt.savefig(f"plots/{method}/{method}_{extr_Q}_prior_{defmod}_{function}_{temp}_Nt{Nt}_noise_comparison.png")
                else:
                    plt.savefig(f"plots/{method}/{method}_{extr_Q}_prior_{defmod}_{function}_{temp}_Nt{Nt}_noise{noise[0]}.png")
            else:
                if comparison:
                    plt.savefig(f"plots/Rho_comparison_{function}_{temp}_Nt{Nt}_noise{noise[0]}_v2.png")
                else:
                    plt.savefig(f"plots/{method}/{method}_{extr_Q}_{function}_{temp}_Nt{Nt}_noise{noise[0]}.png")
        else:
            plt.savefig(f"plots/{method}/{method}_{extr_Q}_{function}_{temp}_Nt{Nt}_B{B_field}_{direction}_zoomed.png")

if __name__ == "__main__":
    main()