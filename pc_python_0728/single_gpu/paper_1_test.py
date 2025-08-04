# -*- coding: utf-8 -*-
#
# Created on 2024-04-28 (Sunday) at 16:52:44
#
# Author: Epsilon-79th
#
# Usage: Basic operations and debugging for calculations of photonic crystal bandgaps.
#
# tol_cmp:  variable tolerance.
# scal_cmp: variable scalings.
# pnt_cmp:  variable penalty coefficients.
# rela_cmp: variable relaxation.
# pack_cmp: variable packages (numpy,scipy,cupy,cupyx).
# grid_cmp: variable grid sizes.
# eps_cmp:  variable epsilons, dielectric coefficients.


# Packages.
import numpy as np
import cupy as cp

import discretization as mfd
import dielectric as diel

from eigensolver import *

from numpy import pi
from gpu_opts import *

from numerical_experiment import *

# Default package: cupy.
owari = owari_cuda
NP = cp
     
"""
Part I: Control experiment.
"""
 
# Control experiment.
# Factor: tolerance (tol), SCALing coefficient (SCAL), grid size (n),
#         linear algebra package (pack), dielectric coefficient (eps).
#         penalty number (pnt), relaxation test (rela)
    
def tol_cmp(n,d_flag_name,tols,alpha=np.array([pi,pi,pi])):
    
    """
    Usage:
        The effect of different thresholds on convergence.
    
    Input:
        n:           grid size.
        d_flag_name: name of diel type.
        tols:        various tolerances.
        alpha:       lattice vector.
    
    """

    a_fft, b_fft, diel_ind, inv_fft, x0, shift = uniform_initialization(n, d_flag_name, alpha)
    Diels = lambda x: diel.diels_chiral(EPS)(x, diel_ind)
    A_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)

    n_tols = len(tols)
    lambdas_pnt = NP.zeros((n_tols, M_CONV))
    lambdas_re  = NP.zeros((n_tols, M_CONV))
    iters = np.zeros((n_tols,2))
    
    for i in range(n_tols):
        lambdas_pnt[i,:], x, iters[i, :] = lobpcg_sep_nolock(A_func, P_func, x0, M_CONV, tol = tols[i])
            
        print(f"\ntol = {tols[i]:<5.2e} is done computing.\n")
        lambdas_pnt[i, :], lambdas_re[i, :] = recompute_normalize_print(lambdas_pnt[i, :], x, A_func)

    print("\nConvergence results:")
    for i in range(n_tols):
        print(f"tol = {tols[i]:<5.2e}, iterations = {int(iters[i,0])}, runtime = {iters[i,1]:<5.2f}s.")
    
    print_standard_deviation(lambdas_pnt, lambdas_re)
     
    return

def pnt_cmp(n,d_flag_name,pnts,alpha=np.array([pi,pi,pi])):
    # Increase default penalty by ratio n^pnts[i].
    
    a_fft, b_fft, diel_ind, _ , x0, shift = uniform_initialization(n, d_flag_name, alpha)
    Diels = lambda x: diel.diels_chiral(EPS)(x, diel_ind)
    _ , pnt0 = mfd.set_relaxation(n, alpha)
    b_fft = (b_fft[0] / pnt0, b_fft[1] / pnt0)

    n_pnts = len(pnts)
    lambdas_pnt = NP.zeros((n_pnts, M_CONV))
    lambdas_re  = NP.zeros((n_pnts, M_CONV))
    iters = np.zeros((n_pnts,2))
    
    for i in range(n_pnts):
        pnt = 2 * n ** (pnts[i] + 1.0)
        inv_fft = mfd.inverse_3_times_3_B(b_fft, pnt, shift)
        A_func, P_func = pc_mfd_handle(a_fft, (b_fft[0] * pnt, b_fft[1] * pnt), Diels, inv_fft, shift)

        lambdas_pnt[i,:], x, iters[i,:] = lobpcg_sep_nolock(A_func, P_func, x0, M_CONV, shift = shift)
        
        print(f"\npnt=n^{pnts[i]:<.2f} is done computing.\n")
        lambdas_pnt[i,:], lambdas_re[i,:] = recompute_normalize_print(lambdas_pnt[i, :], x, A_func)
        
    print("Convergence results:")
    for i in range(n_pnts):
        print(f"pnt = {pnts[i]:<5.2e}, iterations = {int(iters[i,0])}, runtime = {iters[i,1]:<5.2f}s.")
    
    print_standard_deviation(lambdas_pnt, lambdas_re)
    
    return


def rela_cmp(n,d_flag_name,relas,alpha=np.array([pi,pi,pi])):
    
    """
    Usage:
        The effect of different relaxation ratios on convergence.
    
    Input:
        n:           grid size.
        d_flag_name: name of diel type.
        relas:       various relaxation ratios.
        alpha:       lattice vector.
    
    """
    
    a_fft, b_fft, diel_ind, inv_fft, _ , shift = uniform_initialization(n,d_flag_name,alpha)
    Diels = lambda x: diel.diels_chiral(EPS)(x, diel_ind)
    A_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)
    
    nn = n * n * n
    n_relas = len(relas)
    lambdas_pnt = NP.zeros((n_relas, M_CONV))
    lambdas_re  = NP.zeros((n_relas, M_CONV))
    iters = np.zeros((n_relas, 2))
    
    for i in range(n_relas):
        m = M_CONV + round(M_CONV * relas[i])
        x = cp.random.rand(3 * nn, m) + 1j * cp.random.rand(3 * nn, m)
        lambdas_pnt[i,:], x, iters[i,:] = lobpcg_sep_nolock(A_func, P_func, x, M_CONV)
        lambdas_pnt[i,:], lambdas_re[i,:] = recompute_normalize_print(lambdas_pnt[i,:], x, A_func)
    
    print("Convergence results:")
    for i in range(n_relas):
        print(f"Relaxation ratio = {relas[i]:<5.2f}, iterations = {int(iters[i,0])}, runtime = {iters[i,1]:<5.2f}s.")
    
    print_standard_deviation(lambdas_pnt, lambdas_re)
    
    return

def scal_cmp(n,d_flag_name,scals,alpha=np.array([pi,pi,pi])):
    
    """
    Usage:
        The effect of different saclings on convergence.
    
    Input:
        n:           grid size.
        d_flag_name: name of diel type.
        scals:       various scalings.
        alpha:       lattice vector.
        
    """

    a_fft, b_fft, diel_ind, inv_fft, x0, shift=uniform_initialization(n,d_flag_name,alpha)
    Diels = lambda x: diel.diels_chiral(EPS)(x, diel_ind)

    n_scals=len(scals)
    lambdas_pnt = NP.zeros((n_scals, M_CONV))
    lambdas_re  = NP.zeros((n_scals, M_CONV))
    iters = np.zeros((n_scals, 2))
    
    for i in range(n_scals):
        scal0=(n**scals[i])**2
        A_func, P_func = pc_mfd_handle(a_fft / np.sqrt(scal0), 
                (b_fft[0]/scal0,b_fft[1]/scal0),Diels,(inv_fft[0]*scal0,inv_fft[1]*scal0), shift)
        lambdas_pnt[i,:], x, iters[i,:] = lobpcg_sep_nolock(A_func, P_func, x0, M_CONV, tol = TOL / scal0)

        print(f"\nscal = {scals[i]:<.2e} is done computing.\n")
        lambdas_pnt[i,:], lambdas_re[i,:] = recompute_normalize_print(lambdas_pnt[i,:], x, A_func)
        lambdas_pnt[i,:], lambdas_re[i,:] = lambdas_pnt[i,:] * (scal0**0.5), lambdas_re[i,:] * (scal0**0.5)
    
    print("Convergence results:")
    for i in range(n_scals):
        print(f"SCAL = {scals[i]:<5.2e}, iterations = {int(iters[i,0])}, runtime = {iters[i,1]:<5.2f}s.")
    
    print_standard_deviation(lambdas_pnt, lambdas_re)
    
    return

def eps_cmp(n, epss, alpha=np.array([pi,pi,pi])):
    
    """
    Usage:
        The effect of different diel constants on convergence.
        (Randomly generated dielectric coefficients)        
    
    Input:
        n:           grid size.
        d_flag_name: name of diel type.
        eps:         various diel constants.
        alpha:       lattice vector.
    
    """

    a_fft, b_fft, diel_ind, inv_fft, x0, shift = uniform_initialization(n,"sc_curv",alpha)
    Diels = lambda x: diel.diels_chiral(EPS)(x, diel_ind)
    n_epss = len(epss)
    iters = np.zeros((n_epss,2))
    
    for i in range(n_epss):
        diels = (diels[0], 1/epss[i])
        A_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)
        lambdas_pnt, x, iters[i,:] = lobpcg_sep_nolock(A_func, P_func, x0, M_CONV)        
        print(f"\neps = {epss[i]:<.2e} is done computing.\n")
        _ , _ = recompute_normalize_print(lambdas_pnt, x, A_func)
    
    print("Convergence results:")
    for i in range(n_epss):
        print(f"Diles_EPS = {epss[i]:<5.1f}, iterations = {int(iters[i,0])}, runtime = {iters[i,1]:<5.2f}s.")
    
    return
    
def grid_cmp(Ns,d_flag_name,alpha=np.array([pi,pi,pi])):
    
    """
    Usage:
        The effect of different grid sizes on convergence.
    
    Input:
        n:           grid size.
        d_flag_name: name of diel type.
        relas:       various grid sizes.
        alpha:       lattice vector.
    
    """
    
    Ns.sort()

    n_Ns=len(Ns)
    lambdas_pnt = NP.zeros((n_Ns,M_CONV))
    lambdas_re  = NP.zeros((n_Ns,M_CONV))
    iters = np.zeros((n_Ns,2))
    
    for i in range(n_Ns):
        a_fft, b_fft, diel_ind, inv_fft, x0, shift = uniform_initialization(Ns[i],d_flag_name,alpha)
        Diels = lambda x: diel.diels_chiral(EPS)(x, diel_ind)
        A_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)
        lambdas_pnt[i,:], x, iters[i,:] = lobpcg_sep_nolock(A_func, P_func, x0, M_CONV)

        print(f"\nGrid size= {Ns[i]} is done computing.\n")
        lambdas_pnt[i,:], lambdas_re[i,:] = recompute_normalize_print(lambdas_pnt[i,:], x, A_func, shift) 

    print("Convergence results:")
    for i in range(n_Ns):
        print(f"n = {Ns[i]}, iterations = {int(iters[i,0])}, runtime = {iters[i,1]:<5.2f}s.")
    
    print_standard_deviation(lambdas_pnt, lambdas_re)
        
    return


def main():
    
    """
    Main function.
    """
    
    eigen_1p(100, "sc_curv", alpha = np.array([pi,pi,pi]))
    #tol_cmp(100,"sc_curv",[1e-4, 1e-5, 1e-6])
    #pnt_cmp(100,"sc_curv",[0.7, 1.0, 1.2, 1.5])        
    #scal_cmp(100,"sc_curv",[0,0.2,0.5,0.8,1])
    #grid_cmp([100,120],"bcc_double_gyroid")
    #eps_cmp(100, [13.0, 13.0+2j, 20.0, 50.0, 100.0])
    #rela_cmp(100, 'sc_curv', [0.0, 0.2, 0.3, 0.4])
    
if __name__=="__main__":
    main()

    
