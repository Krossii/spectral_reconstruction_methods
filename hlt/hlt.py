#=============================================================================
#
# Implementation of the Hansen-Lupo-Tantalo algorithm
# By: Dean Valois (c)
#
#=============================================================================
#
from mpmath import *
from numpy import loadtxt
import numpy as np

import pprint
import argparse
import json
import os
from typing import List, Tuple, Callable

mp.pretty = True

class HansenLupoTantalo:
    def __init__(self,lamb,smearing):
        self.lamb = lamb
        self.smearing = smearing
        self.omega_max = mpf("inf")

        self.KernelType = "zero_T" # or "finite_T"
        self.rho_over_w = False

        self.kernel = lambda w,tau: self.zero_T_kernel(w,tau)

        self.Nt = 0
        self.Npts = 0
        self.Njacks = 0
        self.cov_matrix_norm = 1

        self.q_vec = 0
        self.d = 8e-6 # d = sqrt(A[q]/A[0]) for stability analysis
        self.weight = lambda w,n: exp(w*n)
        self.input_file = "whatever"
        self.save_dir = "reconstructions"
        self.data_dict = {"lattice":None, "mean":None, "error":None, "jackknifes":None}

        self.rho_dict = {"lattice":None, "mean":None, "error":None, "jackknifes":None}

    def set_hlt_kernel(self):
        if self.KernelType == "zero_T":
            self.kernel = lambda w,tau: self.zero_T_kernel(w,tau)
        elif self.KernelType == "half_zero_T":
            self.kernel = lambda w,tau: self.half_zero_T_kernel(w,tau)
        elif self.KernelType == "finite_T":
            self.kernel = lambda w,tau: self.finite_T_kernel(w,tau)

    def finite_T_kernel(self,w,tau):
        return w*cosh(w*(tau-0.5*self.Nt))/sinh(0.5*w*self.Nt)

    def zero_T_kernel(self,w,tau):
        return w*(exp(-w*tau) + exp(-w*(self.Nt-tau)))

    def half_zero_T_kernel(self,w,tau):
        return w*exp(-w*tau)

    def target_smearing_function(self,w,w_ref):
        Z = 0.5*( 1 + erf(w_ref/(sqrt(2)*self.smearing)) )
        return exp(-0.5*((w-w_ref)/self.smearing)**2)/(Z*sqrt(2*mp.pi)*self.smearing)

    def mean_and_error_from_jackknifes(self):
        N_omegas = len(self.rho_dict["mean"])
        for w in range(N_omegas):
            self.rho_dict["mean"][w] = fsum([self.rho_dict["jackknifes"][w,jack] for jack in range(self.Njacks)])/self.Njacks

        for w in range(N_omegas):
            self.rho_dict["error"][w] = fsum([(self.rho_dict["mean"][w] - self.rho_dict["jackknifes"][w,njacks])**2 for njacks in range(self.Njacks)])
            self.rho_dict["error"][w] = sqrt((self.Njacks-1)* self.rho_dict["error"][w] /self.Njacks)

    def build_cov_matrix_from_jackknifes(self):

        cov_matrix = zeros(self.Npts)
        conn_part = zeros(self.Npts)
        disc_part = matrix([0 for k in range(self.Npts)])

        for jack in range(self.Njacks):
            for k in range(self.Npts):
                disc_part[k] += self.data_dict["jackknifes"][k,jack]/self.Njacks
                for l in range(k+1):
                    conn_part[k,l] += self.data_dict["jackknifes"][k,jack] * self.data_dict["jackknifes"][l,jack]/self.Njacks

        for k in range(self.Npts):
            for l in range(k+1):
                cov_matrix[k,l] = self.cov_matrix_norm * (self.Njacks-1) * (conn_part[k,l] - disc_part[k]*disc_part[l])
                #print( (self.Njacks-1) *(conn_part[k,l] - disc_part[k]*disc_part[l]))
                cov_matrix[l,k] = cov_matrix[k,l]

        return cov_matrix

    def build_f_vector(self,w_ref,n=0,w0=0):
        f_vec = matrix([0 for k in range(self.Npts)])

        for k in range(self.Npts):
            tau = self.data_dict["lattice"][k]
            Integrand = lambda w: self.weight(w,n) * self.kernel(w,tau) * self.target_smearing_function(w,w_ref)
            f_vec[k] = quad(Integrand,[w0,self.omega_max])
        return f_vec

    def build_R_vector(self,w0=0):
        R_vec = matrix([0 for k in range(self.Npts)])

        for k in range(self.Npts):
            tau = self.data_dict["lattice"][k]
            Integrand = lambda w: self.kernel(w,tau)
            R_vec[k] = quad(Integrand,[w0,self.omega_max])
        return R_vec

    def build_W_matrix(self,n=0,w0=0):
        W_matrix = zeros(self.Npts)

        for k in range(self.Npts):
            for l in range(k+1):
                tau_1 = self.data_dict["lattice"][k]
                tau_2 = self.data_dict["lattice"][l]
                Integrand = lambda w: self.weight(w,n) * self.kernel(w,tau_1) * self.kernel(w,tau_2)
                W_matrix[k,l] = quad(Integrand,[w0,self.omega_max])
                W_matrix[l,k] = W_matrix[k,l]

        return W_matrix

    def calculate_d(self,n,w_ref,A_matrix,f_vec,w0=0):
        L2_norm = quad(lambda w: self.weight(w,n) * self.target_smearing_function(w,w_ref)**2,[w0,self.omega_max])
        self.d = self.q_vec.T * A_matrix * self.q_vec - 2*fdot(self.q_vec,f_vec)
        self.d = sqrt(self.d[0]/L2_norm + 1)


    def solve(self,w_vec,n_index=0):

        self.rho_dict["mean"] = matrix([0 for n in range(len(w_vec))])
        self.rho_dict["error"] = matrix([0 for n in range(len(w_vec))])
        self.rho_dict["jackknifes"] = zeros(len(w_vec),self.Njacks)

        R_vec = self.build_R_vector()
        A_matrix = self.build_W_matrix()

        cov_matrix = self.build_cov_matrix_from_jackknifes()
        W_matrix = (1.-self.lamb)*A_matrix + self.lamb * cov_matrix

        WR = lu_solve(W_matrix,R_vec)

        norm_R = fdot(R_vec,WR)

        for n in range(len(w_vec)):
            aux_vec = self.build_f_vector(w_vec[n])
            f_vec = (1.-self.lamb) * aux_vec

            Wf = lu_solve(W_matrix,f_vec)

            norm_f = fdot(R_vec,Wf)

            self.q_vec = Wf + WR*(1.-norm_f)/norm_R

            for jack in range(self.Njacks):
                self.rho_dict["jackknifes"][n,jack] = fdot(self.q_vec,self.data_dict["jackknifes"][:,jack])

            if n == 0:
                self.save_smearing_function(w_vec[n])

            self.calculate_d(n_index,w_vec[n],A_matrix,aux_vec) # NOTE: Not optimal. It is calculating d for multiple w

        self.mean_and_error_from_jackknifes()

    def read_lattice_data(self,input_file,ExcludeTauZero = True):

        self.input_file = input_file

        data_matrix = loadtxt(self.input_file)

        tau_vec = data_matrix[:,0]
        correlator_vec = data_matrix[:,1]
        error_vec = data_matrix[:,2]
        jackknifes_vec = data_matrix[:,3:]

        self.Njacks = len(jackknifes_vec[0])

        tau_start = 1 if ExcludeTauZero == True else 0

        self.Nt = len(data_matrix[:,0])

        self.data_dict["lattice"] = matrix([nstr(tau) for tau in tau_vec[tau_start:self.Nt//2+1]])
        self.data_dict["mean"] = matrix([nstr(G) for G in correlator_vec[tau_start:self.Nt//2+1]])#data_matrix[:,][tau_start:self.Nt//2+1]
        self.data_dict["error"] = matrix([nstr(err) for err in error_vec[tau_start:self.Nt//2+1]])#data_matrix[:,1][tau_start:self.Nt//2+1]

        self.data_dict["jackknifes"] = matrix(self.Nt//2+1,self.Njacks)
        for k in range(self.Nt//2+1):
            for jack in range(self.Njacks):
                self.data_dict["jackknifes"][k,jack] = nstr(jackknifes_vec[k][jack])

        self.cov_matrix_norm = 1./correlator_vec[0]**2

        self.Npts = len(self.data_dict["lattice"])

    def save_output_data(self,w_vec):

        filename = self.input_file.split("/")[-1]
        output_file = open("%s/hlt_specf_%.2f.%s"%(self.save_dir,self.smearing,filename),"w")

        for n in range(len(w_vec)):
            output_file.write("%.6e\t%.6e\t%.6e"%(w_vec[n],self.rho_dict["mean"][n],self.rho_dict["error"][n]))
            for jack in range(self.Njacks):
                output_file.write("\t%.6e"%self.rho_dict["jackknifes"][n,jack])
            output_file.write("\n")

    def save_smearing_function(self,w):

        filename = self.input_file.split("/")[-1]
        output_file = open("reconstructions/hlt_smearf_%.2f.%s"%(self.smearing,filename),"w")

        smearing_func = np.array([])
        w_vec = list([float(i) for i in np.linspace(1e-8,3,100)])

        for w_star in w_vec:
            kernel_vec = [self.kernel(w_star,tau) for tau in self.data_dict["lattice"]]
            smearing_func = fdot(self.q_vec,kernel_vec)
            target = self.target_smearing_function(w_star,w)
            output_file.write("%.6e\t%.6e\t%.6e\n"%(w_star,target,smearing_func))

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

#
#==============================================================================
# Main
#==============================================================================
#

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

    
    mp.dps = parameterHandler.get_params()["precision"] # Precision

    w_vec = list([float(i) for i in np.linspace(
        parameterHandler.get_params()["omega_min"], 
        parameterHandler.get_params()["omega_max"], 
        parameterHandler.get_params()["omega_points"])])

    hlt = HansenLupoTantalo(
        parameterHandler.get_params()["lamb"], 
        parameterHandler.get_params()["smearing"]) # 1st = lambda, 2nd = smearing
    
    hlt.read_lattice_data(parameterHandler.get_params()["correlatorFile"])

    if parameterHandler.get_params()["FiniteT_kernel"]:
        hlt.KernelType = "finite_T"
    else:
        hlt.KernelType = "zero"

    hlt.solve(w_vec)

    hlt.save_output_data(w_vec)
    

paramsDefaultDict = {
    "Method": "HLT",
    #HLT specific; precision, lambda, smearing
    "precision": 20,
    "lamb": 0,
    "smearing": 0.3,
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
