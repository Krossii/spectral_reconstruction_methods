import numpy as np
import matplotlib.pyplot as plt
import my_plot_style as myplot
import os

def compare_elec_conduct_different_methods(direction):

    plot_name = plt.figure("compare electric conductivities")

    Ns = 48
    Nt = 16

    methods = {"Gaussian":"gaussian","Multipoint":"multipoint","Unsupervised learning":"simran_data", "MEM":"mem", "MEM quadratic":"mem_quadratic"}

    temp = myplot.set_color_palette_using_keys("Set1",np.arange(9))
    my_palette = dict()
    my_palette["Gaussian"] = temp[0]
    my_palette["Multipoint"] = temp[1]
    my_palette["Unsupervised learning"] = temp[2]
    my_palette["MEM quadratic"] = temp[7]
    my_palette["MEM"] = temp[8]

    ax = plt.subplot(111)

    for m in methods.keys():
        input_file = "data/%s/econduct_B_%s.dat"%(methods[m],direction)
        if os.path.isfile(input_file):
            B_vec, conduct_vec, conduct_err = np.loadtxt(input_file,usecols=(0,1,2),unpack=True)

            fac = 1./conduct_vec[0]

            B_vec = 6*np.pi*B_vec*Nt**2/Ns**2
            conduct_vec = fac * conduct_vec
            conduct_err = fac * conduct_err

            plt.errorbar(B_vec[1:],conduct_vec[1:],yerr=conduct_err[1:],fmt="s",capsize=4,color=my_palette[m],label="%s"%m)

    #plt.text(0.5, 0.5, "PRELIMINARY", color="green", alpha=0.15,transform=ax.transAxes, fontsize=26,horizontalalignment="center")

    plt.title(r"$48^3\times16$",x=0.15,y=0.58,horizontalalignment="center")
    plt.legend(loc="upper left")
    plt.xlabel("$eB/T^2$")
    plt.ylabel("$\sigma_%s(B)/\sigma_%s(B=0)$"%(direction,direction))
    plt.xlim(0,27)
    plt.ylim(0,5)
    plt.legend(loc="upper left")
    plt.subplots_adjust(left=0.17,right=0.99,bottom=0.13,top=0.98)

    plt.savefig("conduct_%s_variousmethods.jpg"%direction,dpi=500)

def compare_spectral_function_different_methods(Nb,epsilon,direction):

    plot_name = plt.figure("compare spectral function")

    Ns = 48
    Nt = 16

    ax = plt.subplot()

    temp = myplot.set_color_palette_using_keys("Set1",np.arange(9))
    my_palette = dict()
    my_palette["gauss"] = temp[0]
    my_palette["multi"] = temp[1]
    my_palette["unsup"] = temp[2]
    my_palette["mem_const"] = temp[8]
    my_palette["mem_quad"] = temp[7]

    # Gaussian results
    input_file = "data/gaussian/gauss_specf.data_wilson_emconduc_48_16_b6.872_B%i_%s.txt"%(Nb,direction)
    if os.path.isfile(input_file):
        w_vec, rho_gaussian, rho_gaussian_err = np.loadtxt(input_file,usecols=(0,1,2),unpack=True)
        plt.fill_between(Nt*w_vec,rho_gaussian-rho_gaussian_err,rho_gaussian+rho_gaussian_err,alpha=0.5,color=my_palette["gauss"],edgecolor="black",label="Gaussian")

    # HLT results
    """
    input_file = "data/hlt/hlt_specf_eps%.2f.data_wilson_emconduc_48_16_b6.872_B%i_%s.txt"%(epsilon,Nb,direction)
    if os.path.isfile(input_file):
        w_vec, rho_vec, rho_err = np.loadtxt(input_file,usecols=(0,1,2),unpack=True)
        plt.fill_between(Nt*w_vec,rho_vec-rho_err,rho_vec+rho_err,alpha=0.5,color=my_palette["hlt"],edgecolor="black",label=r"HLT (smearing = %.2f)"%epsilon)
    """

    # MEM results
    input_file_const = "../../spectral_reconstruction_methods/mem/outputs/emconduc_recs/constant_prior/RhoOverOmega_finite_T_prior_constant_data_wilson_emconduc_48_16_b6.872_B%i_%s.txt"%(Nb,direction)
    input_file_quad = "../../spectral_reconstruction_methods/mem/outputs/emconduc_recs/quadratic_prior/RhoOverOmega_finite_T_prior_quadratic_data_wilson_emconduc_48_16_b6.872_B%i_%s.txt"%(Nb,direction)
    
    
    if os.path.isfile(input_file_const):
        w_vec, rho_vec, rho_err = np.loadtxt(input_file_const,usecols=(0,1,2),unpack=True)
        w_mem = w_vec[w_vec <= 1.879]
        rho_mem = rho_vec[:len(w_mem)]
        rho_err_mem = rho_err[:len(w_mem)]
        plt.fill_between(Nt*w_mem,rho_mem-rho_err_mem,rho_mem+rho_err_mem,alpha=0.5,color=my_palette["mem_const"],edgecolor="black",label="MEM constant prior")
    
    if os.path.isfile(input_file_quad):
        w_vec, rho_vec, rho_err = np.loadtxt(input_file_quad,usecols=(0,1,2),unpack=True)
        w_mem_quad = w_vec[w_vec <= 1.879]
        rho_mem_quad = rho_vec[:len(w_mem_quad)]
        rho_err_mem_quad = rho_err[:len(w_mem_quad)]
        plt.fill_between(Nt*w_mem_quad,rho_mem_quad-rho_err_mem_quad,rho_mem_quad+rho_err_mem_quad,alpha=0.5,color=my_palette["mem_quad"],edgecolor="black",label="MEM quadratic prior")

    # Multipoint results
    input_file = "data/multipoint/econduct_B_%s.dat"%direction
    if os.path.isfile(input_file):
        B_vec, conduct_vec, conduct_err = np.loadtxt(input_file,usecols=(0,1,2),unpack=True)

        conduct_dict = {Nb:conduct for Nb,conduct in [*zip(B_vec,conduct_vec)]}
        conduct_err_dict = {Nb:err for Nb,err in [*zip(B_vec,conduct_err)]}

        plt.errorbar([0],[conduct_dict[Nb]],yerr=[conduct_err_dict[Nb]],fmt="s",capsize=4,color=my_palette["multi"],label="Multipoint")

    # Unsupervised ML results
    if direction == "x":
        input_file = "data/simran_data/dataBx/rhoOomegaB%i_x_delomega02783_pts500.txt"%Nb
    else:
        input_file = "data/simran_data/omega_max20/rho_over_omega_B%i_z.txt"%Nb
    if os.path.isfile(input_file):
        w_vec, rho_unsupervised, rho_unsupervised_err = np.loadtxt(input_file,usecols=(0,1,2),unpack=True)
        plt.fill_between(Nt*w_vec,rho_unsupervised-rho_unsupervised_err,rho_unsupervised+rho_unsupervised_err,alpha=0.7,color=my_palette["unsup"],edgecolor="black",label=r"Unsupervised learning")

    # Correlator
    input_file = "../../spectral_reconstruction_methods/dat/data_wilson_emconduc_48_16_b6.872_B%i_%s.txt"%(Nb,direction)
    if os.path.isfile(input_file):
        tau, corr = np.loadtxt(input_file, usecols = (0,1), unpack=True)
        corr_sum = sum(corr)

    #plt.text(0.5, 0.5, "PRELIMINARY", color="green", alpha=0.15,transform=ax.transAxes, fontsize=26,horizontalalignment="center")
    plt.title(r"$48^3\times16$",x=0.15,y=0.62,horizontalalignment="center")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\omega/T$")
    plt.ylabel(r"$\rho(\omega)/\omega$")
    plt.xlim(-2,35)
    plt.ylim(0,0.16)
    plt.legend(loc="upper left")
    plt.subplots_adjust(left=0.17,right=0.99,bottom=0.13,top=0.98)
    plt.savefig("spec_func_variousmethods_%s.jpg"%direction,dpi=500)
    plt.close(plot_name)

    def integrate_spatial_correlator(spectral_function, omega_max):
        for i in range(len(spectral_function)):
            if Nt*w_vec[i] == omega_max:
                cutoff_index = i
                break
        else:
            cutoff_index = len(spectral_function)

        omega_range = np.linspace(0,omega_max, cutoff_index)
        spatial_correlator = np.trapz(spectral_function[:cutoff_index], x=omega_range)
        return spatial_correlator

    def compare_spatial_correlators(spectral_function):
        w_maxes = np.array([10,20,30])
        spatial_correlators = []
        for i in range(len(w_maxes)):
            spatial_correlators.append(integrate_spatial_correlator(spectral_function, w_maxes[i]))
        print(spatial_correlators)
        return np.array(spatial_correlators)

    def plot_spatial_correlators(corr_sum, spf_gaussian, spf_unsupervised, spf_mem, spf_mem_quad):
        print(len(spf_gaussian), len(spf_unsupervised), len(spf_mem), len(spf_mem_quad))
        spatial_correlators_gaussian = compare_spatial_correlators(spf_gaussian)
        spatial_correlators_unsupervised = compare_spatial_correlators(spf_unsupervised)
        spatial_correlators_mem = compare_spatial_correlators(spf_mem)
        spatial_correlators_mem_quad = compare_spatial_correlators(spf_mem_quad)
        plt.figure(figsize=(6,5))
        print(corr_sum)
        plt.axhline(corr_sum, label="Sum of correlator")
        plt.scatter([10,20,30], spatial_correlators_gaussian, label="Gaussian", marker = 'x', color=my_palette["gauss"])
        plt.scatter([10,20,30], spatial_correlators_unsupervised, label="Unsupervised", marker = 'o', color=my_palette["unsup"])
        plt.scatter([10,20,30], spatial_correlators_mem, label="MEM constant", marker = 's', color=my_palette["mem_const"])
        plt.scatter([10,20,30], spatial_correlators_mem_quad, label="MEM quadratic", marker = 'd', color=my_palette["mem_quad"])
        plt.xlabel(r"$\omega_{max}/T$")
        plt.ylabel(r"$\int_0^{\omega_{max}} \frac{\rho(\omega)}{\omega} d\omega$")
        plt.title("Spatial correlator as a function of spectral function cutoff")
        plt.legend()
        plt.savefig("Spatial_correlator_comparison_B%i_direction_%s.jpg"%(Nb,direction), dpi=500)

    #plot_spatial_correlators(corr_sum, rho_gaussian, rho_unsupervised, rho_mem, rho_mem_quad)
#
#===============================================================================
# Main
#===============================================================================
#
Nb = 6
epsilon = 0.4
direction = "x"

myplot.initialize_my_color_setup(0.6)

compare_elec_conduct_different_methods(direction)

#compare_spectral_function_different_methods(Nb,epsilon,direction)

plt.show()
