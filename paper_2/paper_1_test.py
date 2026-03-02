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
from numpy import pi

import discretization as mfd
from numerical_experiments import *

# Default package: cupy.
owari = owari_cuda
NP = cp
SOLVER = lobpcg_sep_softlock
     
"""
Part I: Control experiment.
"""
 
# Control experiment.
# Factor: tolerance (tol), SCALing coefficient (SCAL), grid size (n),
#         linear algebra package (pack), dielectric coefficient (eps).
#         penalty number (pnt), relaxation test (rela)
    
def tol_cmp(n,d_flag,tols,alpha=np.array([pi,pi,pi]), solver = SOLVER):
    
    """
    Usage:
        The effect of different thresholds on convergence.
    
    Input:
        n:           grid size.
        d_flag: name of diel type.
        tols:        various tolerances.
        alpha:       lattice vector.
    
    """

    a_fft, b_fft, inv_fft, x0, shift = uniform_initialization(n, d_flag, alpha)
    Diels = mfd.chiral_handle(n, d_flag)
    A_func, H_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)

    n_tols = len(tols)
    lambdas_pnt = NP.zeros((n_tols, NEV))
    lambdas_re  = NP.zeros((n_tols, NEV))
    iters = np.zeros((n_tols,2))
    
    for i in range(n_tols):
        lambdas_pnt[i,:], x, iters[i, :] = solver(H_func, P_func, x0, NEV, tol = tols[i])
            
        print(f"\ntol = {tols[i]:<5.2e} is done computing.\n")
        lambdas_pnt[i, :], lambdas_re[i, :] = recompute_normalize_print(lambdas_pnt[i, :], x, A_func)

    print("\nConvergence results:")
    for i in range(n_tols):
        print(f"tol = {tols[i]:<5.2e}, iterations = {int(iters[i,0])}, runtime = {iters[i,1]:<5.2f}s.")
    
    print_standard_deviation(lambdas_pnt, lambdas_re)
     
    return

def pnt_cmp(n,d_flag,pnts,alpha=np.array([pi,pi,pi]), solver = SOLVER):
    # Increase default penalty by ratio n^pnts[i].
    
    a_fft, b_fft, _ , x0, shift = uniform_initialization(n, d_flag, alpha)
    Diels =  mfd.chiral_handle(n, d_flag)
    _ , pnt0 = mfd.set_relaxation(n, alpha)
    b_fft = (b_fft[0] / pnt0, b_fft[1] / pnt0)

    n_pnts = len(pnts)
    lambdas_pnt = NP.zeros((n_pnts, NEV))
    lambdas_re  = NP.zeros((n_pnts, NEV))
    iters = np.zeros((n_pnts,2))
    
    for i in range(n_pnts):
        pnt = 2 * n ** (pnts[i])
        inv_fft = mfd.inverse_3_times_3_B(b_fft, pnt, shift)
        A_func, H_func, P_func = pc_mfd_handle(a_fft, (b_fft[0] * pnt, b_fft[1] * pnt), Diels, inv_fft, shift)

        lambdas_pnt[i,:], x, iters[i,:] = solver(H_func, P_func, x0, NEV)
        
        print(f"\npnt=n^{pnts[i]:<.2f} is done computing.\n")
        lambdas_pnt[i,:], lambdas_re[i,:] = recompute_normalize_print(lambdas_pnt[i, :], x, A_func)
        
    print("Convergence results:")
    for i in range(n_pnts):
        print(f"pnt = 2N^{pnts[i]:<5.2e}, iterations = {int(iters[i,0])}, runtime = {iters[i,1]:<5.2f}s.")
    
    print_standard_deviation(lambdas_pnt, lambdas_re)
    
    return


def rela_cmp(n,d_flag,relas,alpha=np.array([pi,pi,pi]), solver = SOLVER):
    
    """
    Usage:
        The effect of different relaxation ratios on convergence.
    
    Input:
        n:           grid size.
        d_flag: name of diel type.
        relas:       various relaxation ratios.
        alpha:       lattice vector.
    
    """
    
    a_fft, b_fft, inv_fft, _ , shift = uniform_initialization(n,d_flag,alpha)
    Diels =  mfd.chiral_handle(n, d_flag)
    A_func, H_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)
    
    nn = n * n * n
    n_relas = len(relas)
    lambdas_pnt = NP.zeros((n_relas, NEV))
    lambdas_re  = NP.zeros((n_relas, NEV))
    iters = np.zeros((n_relas, 2))
    
    for i in range(n_relas):
        m = NEV + round(NEV * relas[i])
        x = cp.random.rand(3 * nn, m) + 1j * cp.random.rand(3 * nn, m)
        lambdas_pnt[i,:], x, iters[i,:] = solver(H_func, P_func, x, NEV)
        lambdas_pnt[i,:], lambdas_re[i,:] = recompute_normalize_print(lambdas_pnt[i,:], x, A_func)
    
    print("Convergence results:")
    for i in range(n_relas):
        print(f"Relaxation ratio = {relas[i]:<5.2f}, iterations = {int(iters[i,0])}, runtime = {iters[i,1]:<5.2f}s.")
    
    print_standard_deviation(lambdas_pnt, lambdas_re)
    
    return

def scal_cmp(n,d_flag,scals,alpha=np.array([pi,pi,pi]), solver = SOLVER):
    
    """
    Usage:
        The effect of different saclings on convergence.
    
    Input:
        n:           grid size.
        d_flag: name of diel type.
        scals:       various scalings.
        alpha:       lattice vector.
        
    """

    a_fft, b_fft, inv_fft, x0, shift  =uniform_initialization(n,d_flag,alpha)
    Diels = mfd.chiral_handle(n, d_flag)

    n_scals = len(scals)
    lambdas_pnt = NP.zeros((n_scals, NEV))
    lambdas_re  = NP.zeros((n_scals, NEV))
    iters = np.zeros((n_scals, 2))
    
    for i in range(n_scals):
        scal0=(n**scals[i])**2
        A_func, H_func, P_func = pc_mfd_handle(a_fft / np.sqrt(scal0), 
                (b_fft[0]/scal0,b_fft[1]/scal0),Diels,(inv_fft[0]*scal0,inv_fft[1]*scal0), shift)
        lambdas_pnt[i,:], x, iters[i,:] = solver(H_func, P_func, x0, NEV, tol = TOL / scal0)

        print(f"\nscal = N^({scals[i]:<.2f}) is done computing.\n")
        lambdas_pnt[i,:], lambdas_re[i,:] = recompute_normalize_print(lambdas_pnt[i,:], x, A_func, scal=scal0)
    
    print("Convergence results:")
    for i in range(n_scals):
        print(f"SCAL = {scals[i]:<5.2e}, iterations = {int(iters[i,0])}, runtime = {iters[i,1]:<5.2f}s.")
    
    print_standard_deviation(lambdas_pnt, lambdas_re)
    
    return

def eps_cmp(n, epss, alpha = np.array([pi,pi,pi]), solver = SOLVER):
    
    """
    Usage:
        The effect of different diel constants on convergence.
        (Randomly generated dielectric coefficients)        
    
    Input:
        n:           grid size.
        d_flag: name of diel type.
        eps:         various diel constants.
        alpha:       lattice vector.
    
    """

    a_fft, b_fft, inv_fft, x0, shift = uniform_initialization(n,SC_C,alpha)
    Diels =  mfd.chiral_handle(n, d_flag=SC_C)
    n_epss = len(epss)
    iters = np.zeros((n_epss,2))
    
    for i in range(n_epss):
        diels = (diels[0], 1/epss[i])
        A_func, H_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)
        lambdas_pnt, x, iters[i,:] = solver(H_func, P_func, x0, NEV)        
        print(f"\neps = {epss[i]:<.2e} is done computing.\n")
        _ , _ = recompute_normalize_print(lambdas_pnt, x, A_func)
    
    print("Convergence results:")
    for i in range(n_epss):
        print(f"Diles_EPS = {epss[i]:<5.1f}, iterations = {int(iters[i,0])}, runtime = {iters[i,1]:<5.2f}s.")
    
    return
    
def grid_cmp(Ns,d_flag,alpha=np.array([pi,pi,pi]), solver = SOLVER):
    
    """
    Usage:
        The effect of different grid sizes on convergence.
    
    Input:
        n:           grid size.
        d_flag: name of diel type.
        relas:       various grid sizes.
        alpha:       lattice vector.
    
    """
    
    Ns.sort()

    n_Ns=len(Ns)
    lambdas_pnt = NP.zeros((n_Ns,NEV))
    lambdas_re  = NP.zeros((n_Ns,NEV))
    iters = np.zeros((n_Ns,2))
    
    for i in range(n_Ns):
        a_fft, b_fft, inv_fft, x0, shift = uniform_initialization(Ns[i],d_flag,alpha)
        Diels = mfd.chiral_handle(Ns[i], d_flag=SC_C)
        A_func, H_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)
        lambdas_pnt[i,:], x, iters[i,:] = solver(H_func, P_func, x0, NEV)

        print(f"\nGrid size= {Ns[i]} is done computing.\n")
        lambdas_pnt[i,:], lambdas_re[i,:] = recompute_normalize_print(lambdas_pnt[i,:], x, A_func, shift) 

    print("Convergence results:")
    for i in range(n_Ns):
        print(f"n = {Ns[i]}, iterations = {int(iters[i,0])}, runtime = {iters[i,1]:<5.2f}s.")
    
    print_standard_deviation(lambdas_pnt, lambdas_re)
        
    return

def test_cpxlobpcg(n, d_flag=SC_C, alpha=np.array([pi,pi,pi])):
    """
    LOBPCG from cupyx.scipy.sparse.linalg, fails for no reason.
    """

    a_fft, b_fft, inv_fft, x0, shift = uniform_initialization(n, d_flag, alpha)
    diel_func = mfd.chiral_handle(n, d_flag=SC_C)
    A_func, H_func, P_func = pc_mfd_handle(a_fft, b_fft, diel_func, inv_fft, shift)

    lambdas_pnt, x, iters = lobpcg_sep_cpxlinalg(H_func, P_func, x0, NEV)
    _, _ = recompute_normalize_print(lambdas_pnt, x, A_func)
    print(f"Iterations = {int(iters[0])}, runtime = {iters[1]:<5.2f}s.")

    return

def eigen_1p_single_precision(n,d_flag,alpha, type = "chiral", nev = NEV, cpu_x0=False):
    
    """
    Usage:
        Single precision version of eigen_1p.
    """

    a_fft, b_fft, inv_fft, x0, shift = uniform_initialization(n,d_flag,alpha,nev=nev)
    x0 = x0.astype(cp.complex64)
    a_fft = a_fft.astype(cp.complex64)
    b_fft = (b_fft[0].astype(cp.float32), b_fft[1].astype(cp.complex64))
    inv_fft = (inv_fft[0].astype(cp.float32), inv_fft[1].astype(cp.complex64))

    if cpu_x0:
        x0.tofile("../cpu/x0_from_single_gpu.bin")

    if type is None:
        Diels = lambda x:x
    else:
        Diels = eval("mfd."+type+"_handle")(n, d_flag)
    
    A_func, H_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)
    lambdas_pnt, x, iters = lobpcg_sep_softlock(H_func, P_func, x0, nev, singleprecision=True)
    del x0
    
    print(f"n = {n}, lattice type: {d_flag}, alpha = "
          f"[{alpha[0]/pi:<5.2f}, {alpha[1]/pi:<5.2f}, {alpha[2]/pi:<5.2f}] pi, "
          f"iter = {int(iters[0])}, runtime = {iters[1]:<6.3f}s.")
    
    recompute_normalize_print(lambdas_pnt, x, A_func, shift = shift)

    return
        


def main():

    #eigen_1p(100, SC_C, alpha = np.array([pi,pi,pi])) # This from num exp.
    #tol_cmp(100,SC_C,[1e-4*4*pi*pi, 1e-5*4*pi*pi, 1e-3*4*pi*pi])
    #pnt_cmp(100,SC_C,[0.5,1,2])        
    #scal_cmp(100,SC_C,[0,0.2,0.5,0.8,1])
    #grid_cmp([100,120],BCC_DG)
    #eps_cmp(100, [13.0, 13.0+2j, 20.0, 50.0, 100.0])
    #rela_cmp(100, 'sc_curv', [0.0, 0.2, 0.3, 0.4])

    #test_cpxlobpcg(100)
    eigen_1p_single_precision(120, SC_C, np.array([pi,0,0]))

    return
    
if __name__=="__main__":
    main()

    
