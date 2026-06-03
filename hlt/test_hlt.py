from mpmath import *
from hlt import *
import utils
import inspect

def analytic_f_exp_kernel(w_ref,t,Nt,lamb,smearing,w0=0,alpha=0):
    Z = 0.5*(1.+erf(w_ref/(sqrt(2)*smearing)))
    N = lambda k: 0.5*(1.-lamb)*exp(0.5*(alpha-k)*((alpha-k)*smearing**2 + 2*w_ref))/Z
    F = lambda k: 1.+erf(((alpha-k)*smearing**2 + w_ref - w0)/(sqrt(2)*smearing))
    return N(t)*F(t) + N(Nt-t)*F(Nt-t)

def analytic_R_exp_kernel(t,Nt):
    return 1./t + 1./(Nt-t)

def analytic_W_exp_kernel(t,r,Nt,w0=0,alpha=0):
    A = exp(-w0*(t+r+alpha))/(t+r-alpha)
    B = exp(-w0*(Nt+t-r-alpha))/(Nt+t-r-alpha)
    C = exp(-w0*(Nt-t+r-alpha))/(Nt-t+r-alpha)
    D = exp(-w0*(2*Nt-t-r-alpha))/(2*Nt-t-r-alpha)
    return A+B+C+D

def show_test_result(test_status,function_name):

    if test_status == False:
        utils.print_failed("%s: Failed"%function_name)
    else:
        utils.print_passed("%s: Passed"%function_name)

def test_smearing_function_normalization():

    test_status = True

    hlt = HansenLupoTantalo(0.3,0.3)

    w_ref = 7.3392

    target_f = lambda w: hlt.target_smearing_function(w,w_ref)

    IntegralResult = quad(target_f,[0,hlt.omega_max])

    if abs(IntegralResult-1) > 1e-8:
        test_status = False
        print("Smearing normalization gave: %.6e - expected: 1"%IntegralResult)

    this_function_name = inspect.currentframe().f_code.co_name
    show_test_result(test_status,this_function_name)

def test_f_vector_elements():
    test_status = True

    hlt = HansenLupoTantalo(0.3,0.3)

    hlt.Nt = 12
    hlt.data_dict["lattice"] = range(1,hlt.Nt//2+1)
    hlt.Npts = len(hlt.data_dict["lattice"])

    w_ref = 1.356688

    f_vec = hlt.build_f_vector(w_ref)

    for n in range(hlt.Npts):
        tau = hlt.data_dict["lattice"][n]
        numeric = f_vec[n]
        analytic = analytic_f_exp_kernel(w_ref,tau,hlt.Nt,hlt.lamb,hlt.smearing)
        if abs(numeric-analytic) > 1e-8:
            test_status = False
            print("(tau=%i): %.6e %.6e"%(tau,numeric,analytic))

    this_function_name = inspect.currentframe().f_code.co_name
    show_test_result(test_status,this_function_name)

def test_R_vector_elements():

    test_status = True

    hlt = HansenLupoTantalo(0.3,0.3)

    hlt.Nt = 10
    hlt.data_dict["lattice"] = range(1,hlt.Nt//2+1)
    hlt.Npts = len(hlt.data_dict["lattice"])

    R_vec = hlt.build_R_vector()

    for n in range(hlt.Npts):
        tau = hlt.data_dict["lattice"][n]
        numeric = R_vec[n]
        analytic = analytic_R_exp_kernel(tau,hlt.Nt)
        if abs(numeric-analytic) > 1e-8:
            test_status = False

    this_function_name = inspect.currentframe().f_code.co_name
    show_test_result(test_status,this_function_name)

def test_W_matrix_elements():

    test_status = True

    hlt = HansenLupoTantalo(0.3,0.3)

    hlt.omega_max = 50
    hlt.Nt = 6
    hlt.data_dict["lattice"] = range(1,hlt.Nt//2+1)
    hlt.Npts = len(hlt.data_dict["lattice"])

    W_matrix = hlt.build_W_matrix()

    for n in range(hlt.Npts):
        for l in range(n+1):
            tau_1 = hlt.data_dict["lattice"][n]
            tau_2 = hlt.data_dict["lattice"][l]
            numeric = W_matrix[n,l]
            analytic = analytic_W_exp_kernel(tau_1,tau_2,hlt.Nt)
            if abs(numeric-analytic) > 1e-8:
                test_status = False
                print("(tau1=%i,tau2=%i): %.6e %.6e"%(tau_1,tau_2,numeric,analytic))

    this_function_name = inspect.currentframe().f_code.co_name
    show_test_result(test_status,this_function_name)

def test_symmetry_of_W():
    test_status = True

    hlt = HansenLupoTantalo(0.3,0.3)

    hlt.omega_max = 50
    hlt.Nt = 8
    hlt.data_dict["lattice"] = range(1,hlt.Nt//2+1)
    hlt.Npts = len(hlt.data_dict["lattice"])

    W_matrix = hlt.build_W_matrix()

    for n in range(hlt.Npts):
        for l in range(n+1):
            tau_1 = hlt.data_dict["lattice"][n]
            tau_2 = hlt.data_dict["lattice"][l]
            if abs(W_matrix[n,l]-W_matrix[l,n]) > 1e-10:
                test_status = False
                print("(tau1=%i,tau2=%i): %.6e %.6e"%(tau_1,tau_2,W_matrix[n,l],W_matrix[l,n]))

    this_function_name = inspect.currentframe().f_code.co_name
    show_test_result(test_status,this_function_name)

def test_normalization_of_coefficients():

    test_status = True

    hlt = HansenLupoTantalo(0.3,0.3)

    hlt.read_lattice_data("data/mock_corr_BW_Nt16_noise4.dat")

    w_vec = [1.3]
    R_vec = hlt.build_R_vector()
    rho_vec = hlt.solve(w_vec)

    for n in range(len(w_vec)):
        if abs(hlt.dot_product(hlt.q_vec,R_vec) - 1) > 1e-10:
            test_status = False

    this_function_name = inspect.currentframe().f_code.co_name
    show_test_result(test_status,this_function_name)

def test_precision_of_result():
    # Design a test case where the result is known, say pi,
    # and test whether all digits are correct
    return 0
#
#==============================================================================
# Main
#==============================================================================
#
test_smearing_function_normalization()
test_f_vector_elements()
test_R_vector_elements()
test_W_matrix_elements()
test_symmetry_of_W()
test_normalization_of_coefficients()
