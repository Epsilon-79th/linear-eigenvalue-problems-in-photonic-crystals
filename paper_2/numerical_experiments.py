# -*- coding: utf-8 -*-
#
# Created on 2025-02-25 (Tuesday) at 14:48:00
#
# Author: Epsilon-79th
#
# Usage: Numerical experiments.
#

# Packages.
import time
import os, json

import numpy as np
import cupy as cp

from environment import *
import discretization as mfd
import dielectric as diel
from lobpcg import *
from pcfft import *

from numpy import pi

# Default package: cupy.
owari = owari_cuda
NP = cp

"""
Part 0: Initialization, recomputing and normalization, print.
"""

def uniform_initialization(n,d_flag,alpha,nev=NEV,k=K):
    
    """
    Usage:
        Trivial operations in matrix assembling.
    
    Input:
        n:       grid size.
        d_flag:  name of lattice type.
        alpha:   lattice vector.
    
    Output:
        a_fft:   fft3d blocks of A.
        b_fft:   fft3d blocks of B'B.
        inv_fft: fft3d blocks of inv(AA'+pnt B'B).
        x0:      initial guess.
        shift:   ensuring the insingularity of the system.
        
    """
    
    t_h = time.time()

    relax_opt, pnt = mfd.set_relaxation(alpha, scal=SCAL)

    ct = diel.diel_info(d_flag, option='ct')
    a_fft, b_fft = mfd.fft_blocks(n,k,ct,alpha = alpha,scal = SCAL)
    inv_fft = mfd.inverse_3_times_3_B(b_fft,pnt,relax_opt[0])

    a_fft /= SCAL
    b_fft = (pnt * b_fft[0] / SCAL / SCAL, pnt * b_fft[1] / SCAL / SCAL)
    inv_fft = (inv_fft[0] * SCAL * SCAL,inv_fft[1] * SCAL * SCAL)
    m = round(nev * relax_opt[1]) + nev

    x0 = cp.random.rand(3 * n ** 3,m) + 1j * cp.random.rand(3 * n ** 3,m)

    t_o = owari()
    print(f"Matrix blocks done, {t_o - t_h:<6.3f}s elapsed.")
    
    return a_fft, b_fft, inv_fft, x0, relax_opt[0]

def pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift = 0.0):
    
    """
    (Photonic Crystals / Mimetic Finite Difference / Matrix-free Handle.)
    Usage: 
        Matrix handle of A_func, H_func and P_func.
    """

    A_func = lambda x: AMA(x, a_fft, Diels) # AMA' handle.
    H_func = lambda x: AMA_BB(x, a_fft, b_fft, Diels, shift) # AMA+\gamma B'B handle.
    P_func = lambda x: H_block(x, inv_fft)  # Precondition inv(AA'+\gamma B'B) handle.
    
    return A_func, H_func, P_func

def recompute_normalize_print(lambdas_in, x, A_func, shift=0.0, scal = SCAL):
    """
    Usage:
        Recompute eigenvalues, normalize eigenvectors, print results.
        Output result table.
    Input:
        lambdas_pnt: Eigenvalues from lobpcg.
        x:           Eigenvectors from lobpcg.
        A_func:      Matrix handle of A.
        nev:         Number of desired eigenpairs.
        shift:       Shift of the system (default = 0).
        scal:        Scaling, lattice constant (default = 1).
    Output:
        lambdas_pnt: Recomputed eigenvalues.
        lambdas_re:  Recomputed and normalized eigenvalues.
    """
    
    t_h = time.time()
    adax = A_func(x)
    if shift > 0.0:
        lambdas_pnt = lambdas_in - shift
    else:
        lambdas_pnt = lambdas_in.copy()
    R = adax - x * lambdas_pnt
    lambdas_re = ((x.T.conj() @ adax).diagonal() / (x.T.conj() @ x).diagonal()).real

    # Check NaN (optional, usually unnecessary).
    nan_pnt = cp.where(cp.isnan(lambdas_pnt))[0]
    nan_re = cp.where(cp.isnan(lambdas_re))[0]

    # If pnt nan, re valid, yellow warning.
    # If both nan, red warning. 
    if len(nan_pnt) > 0:
        for ind in nan_pnt:
            if cp.isnan(lambdas_re[ind]):
                print(f"{RED}Warning: NaN occurs in both lambda_pnt and lambda_re, index = {ind}. "
                      f"Please run the program again.")
            else:
                print(f"{YELLOW}Warning: NaN occurs in lambda_pnt, index = {nan_pnt}, "
                      f"but same index in lambda_re is valid.")

    # If re nan, pnt valid, yellow warning and set re = pnt.       
    if len(nan_re) > 0:
        for ind in nan_re:
            if cp.isnan(lambdas_pnt[ind]) is False:
                lambdas_re[ind] = lambdas_pnt[ind]

    # Robust square roots (avoid very small negatives).
    def sqrt_robust(a):
        if (a <= 0) & (a > -1e-8):
            sqrt_a = 0.0
        else:
            sqrt_a = a**0.5
        return sqrt_a
        
    t_o = owari()
    flag_spurious = False
    
    print(f"Runtime for recomputing: {t_o - t_h:<6.3f}s.")
    print("| i  |    omega   |  omega_re  |  abs(omega - omega_re)  | residual  |")
    for i in range(len(lambdas_pnt)):
        l1 = sqrt_robust(lambdas_pnt[i]) * scal/(2*pi)
        l2 = sqrt_robust(lambdas_re[i]) * scal/(2*pi)
        print(f"| {i + 1:<2d} | {l1:<10.6f} | {l2:<10.6f} |        {abs(l1-l2):<10.3e}       | {norm(R[:,i]):<6.3e} |")
        lambdas_pnt[i], lambdas_re[i] = l1, l2
        if l1-l2>1e-3:
            flag_spurious = True
    
    if flag_spurious:
        raise ValueError(f"{RED}Spurious eigenvalues occur.{RESET}")
    
    return lambdas_pnt, lambdas_re

def condition_number(mat_in):
    """
    Compute the condition number of matrix.
    """

    if callable(mat_in):
        prec = lambda x:x
    else:
        d = cp.array(1.0/mat_in.diagonal())
        prec = lambda x: (x.T*d).T
    eig_s, iters = lobpcg_default(mat_in, prec = prec, nev = 2, rlx=4, info = True)
    print(f"Eig_s done, runtime = {iters[1]:<6.3f}s, {eig_s}.")

    eig_l, iters = lobpcg_default(mat_in, nev = 2, rlx=4, info = True, maxmin="max")
    print(f"Eig_l done, runtime = {iters[1]:<6.3f}s, {eig_l}.")
    print(f"Condition number of eps_loc: {eig_l[0]/eig_s[0]:<6.3f}.")

    return

def print_standard_deviation(lambdas_pnt, lambdas_re, nev=NEV):
    """
    Data postprocessing.
    """
    sd_pnt, sd_re = cp.std(lambdas_pnt,axis=0), cp.std(lambdas_re,axis=0)
    print("\nStandard deviation of each eigenvalue:")
    print("| i  |  std_pnt  |  std_re   |")
    for i in range(nev):
        print(f"| {i + 1:<2d} | {sd_pnt[i]:<6.3e} | {sd_re[i]:<6.3e} |")

def convergence_rate(residuals):
    def rated(x):   # Compute dampening rate by linear regression.  
        return np.polyfit(np.arange(len(x)), x, 1)[0]

    m0 = rated(np.log(residuals))
    print(f"\nGlobal average convergence rate: {np.exp(m0):<6.3f}.")

    n_half = len(residuals) // 2
    m1 = rated(np.log(residuals[:n_half]))
    m2 = rated(np.log(residuals[n_half:]))
    print(f"First half average convergence rate: {np.exp(m1):<6.3f}.")
    print(f"Second half average convergence rate: {np.exp(m2):<6.3f}.")

    return

"""
Part 0.5:
    Demo - Eigenproblem at a single lattice vector.
"""

def eigen_1p(n,d_flag,alpha, type = "chiral", nev = NEV, solver = lobpcg_sep_softlock, cpu_x0=False):
    
    """
    Usage:
        Compute eigenvalues at one SINGLE lattice vector.
    
    Input:
        n:           grid size.
        d_flag:      name of diel type.
        alpha:       lattice vector.
        type:        type of dielectric lattice, default is "chiral".
    """

    a_fft, b_fft, inv_fft, x0, shift = uniform_initialization(n,d_flag,alpha,nev=nev)

    if cpu_x0:
        x0.tofile("../cpu/x0_from_single_gpu.bin")

    if type is None:
        Diels = lambda x:x
    else:
        Diels = eval("mfd."+type+"_handle")(n, d_flag)
    
    A_func, H_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)

    lambdas_pnt, x, iters = solver(H_func, P_func, x0, nev)
    del x0
    
    print(f"n = {n}, lattice type: {d_flag}, alpha = "
          f"[{alpha[0]/pi:<5.2f}, {alpha[1]/pi:<5.2f}, {alpha[2]/pi:<5.2f}] pi, "
          f"iter = {int(iters[0])}, runtime = {iters[1]:<6.3f}s.")
    
    print(f"{CYAN}\nEigenvalues (not sqrt normalized) are:\n")
    print(lambdas_pnt)
    print("\n")
    print(f"\n{RESET}")
    recompute_normalize_print(lambdas_pnt[:nev], x[:,:nev], A_func, shift = shift)

    return
        

"""
Part 1: Bandgap.
"""

def bandgap_wnk_check(n, d_flag, type = TYPE0, eps_opt = 0, ind=[]):

    """
    Usage:
        Check wave number vector of a given index (start from 1).
    """
    
    path_bandgap = "../output/" + type + "/bandgap_" + d_flag + str(eps_opt) + ".json"
    with open(path_bandgap,'r') as file:
        gap_lib = json.load(file)
    
    freq = np.array(gap_lib[d_flag + "_" + str(n) + "_frequencies"])
    info = np.array(gap_lib[d_flag + "_" + str(n) + "_iterations"])

    for i in ind:
        alpha=diel.diel_alpha(d_flag, i)
        print(f"Index = {i}, wnk = ({alpha[0]/pi:<6.3f}, {alpha[1]/pi:<6.3f}, {alpha[2]/pi:<6.3f})pi.")
        print(f"Iterations = {int(info[i,0]):4d}, runtime = {info[i,1]:6.3f}s.")
        print("List of frequencies follows as:")
        print(freq[i,:])

    return 

def bandgap_history_check(n, d_flag, type = TYPE0, eps_opt = 0):

    # Extract d_flag from path_bandgap.
    path_bandgap = "../output/" + type + "/bandgap_" + d_flag + str(eps_opt) + ".json"

    # Name of variable.
    var_name_it = d_flag + "_" + str(n) + "_iterations"

    if not os.path.exists(path_bandgap):
        # New lattice bandgap.        
        print(f"The bandgap of type {type},{d_flag} has no previous record.")
    else:
        with open(path_bandgap,'r') as file:
            gap_lib = json.load(file)
        if var_name_it in gap_lib.keys():
            # Previous record exists.
            print(f"{GREEN}Lattice type {type},{d_flag} with grid size n = {n} has a_fft previous record.{RESET}")
            gap_rec_it = gap_lib[var_name_it]
            
            # Get error, uncomputed index.
            err_ind = [i for i,a in enumerate(gap_rec_it) if a==[-1,-1]]
            if len(err_ind) > 0:
                print(f"{RED}Warning: Blow up results detected: {err_ind}.{RESET}")
                
            empty_ind = [i for i,a in enumerate(gap_rec_it) if a==[0,0]]
            if len(empty_ind) > 0:
                print(f"{YELLOW}Following indices remain uncomputed: {empty_ind}.{RESET}")
        
            if len(empty_ind) == 0 and len(err_ind) == 0:
                print(f"{GREEN}All indices of {type},{d_flag} have been computed without errors.{RESET}")
        else:
            print(f"{YELLOW}Lattice type {type},{d_flag} has a record, "
                  f"but grid size {n} hasn't been computed.{RESET}")
            
    return

def bandgap(n, d_flag, solver = lobpcg_sep_softlock, type = TYPE0, eps_opt = 0, indices = None):
    
    """
    Usage:
        Compute the complete (or part of) bandgap w.r.t to a_fft given lattice material with 
        certain grid size. (uniformly tensor division).
        Support ONLY linear eigenproblem.
        
    Input:
        n:           Grid size.
        d_flag_name: Name of dielectric lattice.
    
    **kwargs:
        type:       Sub-directory for output files. Default is "chiral". 
        solver:     Eigen-solver, default is lobpcg_sep_nolock.
        indices:    Indices of translation vectors.

    (The dielectric constants are always default in bandgap)
         
    """
    
    nn = n * n * n

    # Load diel_info.
    ct, sym_points = diel.diel_info(d_flag)
    n_pt = sym_points.shape[0] - 1
    gap = GAP

    # Discrete lattice points.
    alphas = np.zeros((n_pt * gap, 3))
    for i in range(n_pt):
        alphas[(i + 1) * gap - 1,:] = sym_points[i + 1,:]
        for j in range(gap - 1):
            alphas[i * gap + j,:] = ((j + 1) * sym_points[i + 1,:] + (gap - j - 1) * sym_points[i, :]) / gap
    
    # Dielectric handle.
    Diels = eval("mfd."+type+"_handle")(n, d_flag, eps_opt = eps_opt)
    
    # Matrix blocks, we apply the default scaling (=1).
    d_fft, di_fft = mfd.fft_blocks(n, K, ct)
    
    # Path and keywords.
    path_bandgap = "../output/" + type + "/bandgap_" + d_flag + str(eps_opt) + ".json"
    var_name_it = d_flag + "_" + str(n) + "_iterations"
    var_name_fq = d_flag + "_" + str(n) + "_frequencies"

    # File Check.
    uncomputed_ind = None
    if not os.path.exists(path_bandgap):
        # New lattice bandgap plot.
        
        print("The bandgap of type ",d_flag," has no previous record.")
        gap_rec_it, gap_rec_fq = [[0] * 2] * (n_pt*gap), [[0] * NEV] * (n_pt * gap)
        gap_lib = {var_name_it: gap_rec_it, var_name_fq: gap_rec_fq}
        with open(path_bandgap,'w') as file:
            json.dump(gap_lib, file, indent = 4)
    else:
        with open(path_bandgap,'r') as file:
            gap_lib = json.load(file)
        if var_name_it in gap_lib.keys():
            # Previous record exists.
            print(f"{GREEN}Lattice type {type},{d_flag} with grid size n = {n} has a_fft previous record.{RESET}")
            gap_rec_it, gap_rec_fq = gap_lib[var_name_it], gap_lib[var_name_fq]
            
            # Get error, uncomputed index.
            err_ind = [i for i,a in enumerate(gap_rec_it) if a==[-1,-1]]
            if len(err_ind) > 0:
                print(f"{RED}Warning: Blow up results detected: {err_ind}.{RESET}")
                
            empty_ind = [i for i,a in enumerate(gap_rec_it) if a==[0,0]]
            if len(empty_ind) > 0:
                print(f"{YELLOW}Following indices remain uncomputed: {empty_ind}.{RESET}")
        
            if len(empty_ind) == 0 and len(err_ind) == 0:
                print(f"{GREEN}All indices of {type},{d_flag} have been computed without errors.{RESET}")
                return []
            
            uncomputed_ind = sorted(list(set(err_ind + empty_ind)))
            del err_ind, empty_ind
        else:
            # New grid size.
            
            print(f"{YELLOW}Lattice type {type},{d_flag} will be computed with a_fft new grid size n = {n}.{RESET}")
            
            gap_rec_it, gap_rec_fq = [[0] * 2] * (n_pt * gap), [[0] * NEV] * (n_pt*gap)
            gap_lib[var_name_it], gap_lib[var_name_fq] = gap_rec_it, gap_rec_fq
            
            with open(path_bandgap,'w') as file:
                json.dump(gap_lib, file, indent = 4)
    
    if indices is None:
        indices = list(range(n_pt*gap)) if uncomputed_ind is None else uncomputed_ind
    elif len(indices) == 0:
        return []
    if max(indices) >= n_pt*gap or min(indices) < 0:
        ValueError("Index is non-positive or is incompatible with gap size.")
    
    """
    Calculation of bandgap.
    """
    
    err_index = []
    pool = cp.get_default_memory_pool()
    
    # Main Loop: compute each lattice point.    
    for i in range(len(indices)):
        t_h = time.time()
        
        alpha = alphas[indices[i]] / SCAL       
        relax_opt, pnt = mfd.set_relaxation(alpha)
        m = NEV + round(NEV * relax_opt[1])
        
        if i == 0 or abs(indices[i] - indices[i-1])>1:
            x0 = cp.random.rand(3 * nn, m) + 1j * cp.random.rand(3 * nn, m)
        elif m <= x.shape[1]:
            x0 = cp.empty((3 * nn, m), dtype = cp.complex128)
            cp.copyto(x0, x[:, 0:m])
        elif m > x.shape[1]:
            n_new = m - x.shape[1]
            x0 = cp.concatenate((x, cp.random.rand(3 * nn, n_new) + 1j * cp.random.rand(3 * nn, n_new)), axis = 1)
        
        a_fft = cp.concatenate((d_fft[:nn]+1j*alpha[0]*di_fft[:nn],\
                           d_fft[nn:2*nn]+1j*alpha[1]*di_fft[nn:2*nn],\
                           d_fft[2*nn:]+1j*alpha[2]*di_fft[2*nn:] ))
        b_fft = (cp.concatenate(((a_fft[0:nn]*a_fft[0:nn].conj()).real,\
                            (a_fft[nn:2*nn]*a_fft[nn:2*nn].conj()).real,\
                            (a_fft[2*nn:]*a_fft[2*nn:].conj()).real)),\
                 cp.concatenate((a_fft[0:nn].conj()*a_fft[nn:2*nn],\
                            a_fft[0:nn].conj()*a_fft[2*nn:],a_fft[nn:2*nn].conj()*a_fft[2*nn:])))
        inv_fft = mfd.inverse_3_times_3_B(b_fft,pnt,relax_opt[0])
        
        a_fft /= SCAL
        b_fft = (pnt*b_fft[0]/SCAL/SCAL,pnt*b_fft[1]/SCAL/SCAL)
        inv_fft = (inv_fft[0]*SCAL*SCAL,inv_fft[1]*SCAL*SCAL)
        
        A_func, H_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, relax_opt[0])
        
        t_o = owari()
        print(f"Matrix blocks done, {t_o-t_h:<6.3f}s elapsed.")
        
        #lambdas_pnt, x, iters = solver(A_func, P_func, x0, NEV, tol=TOL/SCAL/SCAL)   
           
        try:
            lambdas_pnt, x, iters = solver(H_func, P_func, x0, NEV, tol=TOL/SCAL/SCAL)        
        
            print(f"Gap {indices[i]+1} out of {n_pt*gap} ({d_flag}),"
                  f"alpha = ({alpha[0]/pi:<6.3f}, {alpha[1]/pi:<6.3f}, {alpha[2]/pi:<6.3f})pi is computed.")
            print(f"Iterations = {int(iters[0])}, runtime = {iters[1]:<6.3f}s.\n")
        
            _ , lambdas_re = recompute_normalize_print(lambdas_pnt, x, A_func, relax_opt[0])
            
            gap_rec_it[indices[i]] = iters.tolist()
            gap_rec_fq[indices[i]] = lambdas_re.tolist()  
        
        except Exception:
            err_index.append(indices[i])
            lambdas_re = -1.0 * np.ones(NEV)
            x = cp.random.rand(3 * nn, m) + 1j * cp.random.rand(3 * nn, m)
            iters = np.array([-1.0, -1.0])

            gap_rec_it[indices[i]] = iters.tolist()
            gap_rec_fq[indices[i]] = lambdas_re.tolist()                
        
        del x0
        pool.free_all_blocks()
        t_h = time.time()
        gap_lib[var_name_it], gap_lib[var_name_fq] = gap_rec_it, gap_rec_fq
        
        with open(path_bandgap,'w') as file:
            json.dump(gap_lib, file, indent=4)
            
        t_o = owari()
        print(f"Gap info library ({d_flag}) is updated ({indices[i]+1}/{n_pt*gap}), time = {t_o-t_h:<6.3f}s.")
    
    if len(err_index) > 0:
        print(f"{RED}Error occurs to following indices:{RESET}")
        print(err_index)
    else:
        print(f"{GREEN}All indices computed correctly.{RESET}")
    
    return err_index

def test():

    indices = list(range(120))
    for i in range(5):
        indices = bandgap(120, FCC, type=TYPE1, indices = indices, solver = lobpcg_sep_softlock)
        if len(indices) == 0:
            print(f"{GREEN}Full bandgap computation is completed after {i} recomputings.")
            break

    return

def main():
    #  Order: CUDA_VISIBLE_DEVICES=4 python numerical_experiments.py

    #bandgap_wnk_check(120,d_flag=BCC_DG,ind=[139])
    
    eigen_1p(120, SC_C, np.array([pi,0,0]), nev=10, type = TYPE1, solver=lobpcg_sep_softlock)
    #bandgap(120, d_flag=SC_C, type=TYPE2)
    #bandgap(120, d_flag=FCC, type=TYPE2, indices=[60])
    #bandgap(120, d_flag=BCC_SG, type=TYPE2, indices=[20, 98, 100])
    #bandgap(120, d_flag=BCC_DG, type=TYPE0, indices=[39])

    #nev = 15
    #alpha_sample = np.array([0.01,0,0])*pi
    #eigen_1p(120, d_flag=SC_C, alpha=alpha_sample, type=TYPE2, nev=nev)
    #eigen_1p(120, d_flag=BCC_SG, alpha=alpha_sample, type=TYPE2, nev=nev)
    #eigen_1p(120, d_flag=BCC_DG, alpha=alpha_sample, type=TYPE2, nev=nev)
    #eigen_1p(120, d_flag=BCC_DG, alpha=alpha_sample, type=TYPE0, nev=nev)
    #eigen_1p(100, d_flag=BCC_DG, alpha=alpha_sample, type=TYPE0, nev=nev)

    #bandgap_history_check(120, d_flag=SC_C, type=TYPE2)
    #bandgap_history_check(120, d_flag=FCC, type=TYPE2)
    #bandgap_history_check(120, d_flag=BCC_SG, type=TYPE2)
    #bandgap_history_check(120, d_flag=BCC_SG, type=TYPE2)

    #_ = mfd.pseudochiral_crossdof_handle(120, eps_opt=1)
    return
    
    
if __name__ == '__main__':
    main()