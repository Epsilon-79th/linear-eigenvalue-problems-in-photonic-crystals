# -*- coding: utf-8 -*-
#
# Created on 2025-02-25 (Tuesday) at 14:48:00
#
# Author: Epsilon-79th
#
# Usage: Numerical experiments.
#

# Packages.
from time import time
import os,json,sys

import numpy as np
import cupy as cp

import discretization as mfd
import dielectric as diel

from lobpcg import *
from pcfft import *

from numpy import pi
from numpy.random import rand
from gpu_opts import *

# Default package: cupy.
owari = owari_cuda
NP = cp

# Global parameters.
EPS = 13.0                    # Dielectric constant.
K = 4                         # Stencil length.
M_CONV = 10                   # Desired eigenpairs.
SCAL = 1                      # SCALing coefficient.
TOL = 1e-5                    # Tolerance.

"""
Part 0: Initialization, recomputing, normalization, print.
"""

def uniform_initialization(n,d_flag_name,alpha,k=K,gpu_opt=True):
    
    """
    Usage:
        Trivial operations in matrix assembling.
    
    Input:
        n:           grid size.
        d_flag_name: name of lattice type.
        alpha:       lattice vector.
        gpu_opt:     gpu option (default==True).
    
    Output:
        a_fft:   fft diagonal blocks of A.
        b_fft:   fft diagonal blocks of B'B.
        diels:   a tuple contians indices and a coefficient.
        inv_fft: fft diagonal blocks of inv(AA'+pnt B'B).
        x0:      initial guess.
        shift:   ensuring the insingularity of the system.
        
    """
    
    t_h = time()

    relax_opt, pnt = mfd.set_relaxation(n,alpha)
    diel_ind, ct, _ = diel.diels_io(n, d_flag_name)

    t1 = time()
    a_fft, b_fft = mfd.fft_blocks(n,k,ct,alpha = alpha, gpu = gpu_opt)
    t1 = owari() -t1

    t2 = time()
    inv_fft = mfd.inverse_3_times_3_B(b_fft,pnt,relax_opt[0])
    t2 = owari() -t2

    t3 = time()
    a_fft /= SCAL
    b_fft = (pnt * b_fft[0] / SCAL / SCAL, pnt * b_fft[1] / SCAL / SCAL)
    inv_fft = (inv_fft[0] * SCAL * SCAL,inv_fft[1] * SCAL * SCAL)
    m = round(M_CONV * relax_opt[1]) + M_CONV
    t3 = owari() -t3

    t4 = time()
    if gpu_opt:
        x0 = cp.random.rand(3 * n ** 3,m) + 1j * cp.random.rand(3 * n ** 3,m)
    else:
        x0 = rand(3 * n ** 3,m) + 1j * rand(3 * n ** 3, m)
    t4 = owari() -t4

    t_o = owari()
    print(f"Matrix blocks done, {t_o - t_h:<6.3f}s elapsed.")
    print(f"t1 = {t1:<6.3f}s, t2 = {t2:<6.3f}s, t3 = {t3:<6.3f}s, t4 = {t4:<6.3f}s.")
    
    return a_fft, b_fft, diel_ind, inv_fft, x0, relax_opt[0]

def pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift = 0.0):
    
    """
    Usage: Matrix handle of A_func and P_func.
    """

    A_func = lambda x: AMA_BB(x, a_fft, b_fft, Diels, shift)
    P_func = lambda x: H_block(x, inv_fft)
    
    return A_func, P_func

def recompute_normalize_print(lambdas_pnt, x, A_func, shift = 0.0):
    
    t_h = time()
    adax = A_func(x)
    R = adax - x * lambdas_pnt
    
    lambdas_re = ((x.T.conj() @ adax).diagonal() / (x.T.conj() @ x).diagonal()).real
    if shift > 0.0:
        lambdas_pnt -= shift
        lambdas_re -= shift
        
    t_o = owari()
    
    print(f"Runtime for recomputing: {t_o - t_h:<6.3f}s.")
    print("| i  |   lambda   | lambda_re  | abs(lambda - lambda_re) | residual  |")
    for i in range(M_CONV):
        l1 = SCAL*np.sqrt(lambdas_pnt[i])
        l2 = SCAL*np.sqrt(lambdas_re[i])
        print(f"| {i + 1:<2d} | {l1:<10.6f} | {l2:<10.6f} |        {abs(l1-l2):<10.3e}       | {norm(R[:,i]):<6.3e} |")
        lambdas_pnt[i], lambdas_re[i] = l1, l2
    
    return lambdas_pnt[:M_CONV], lambdas_re[:M_CONV]

def print_standard_deviation(lambdas_pnt, lambdas_re):
    NP = arrtype(lambdas_pnt)
    sd_pnt, sd_re = NP.std(lambdas_pnt,axis=0), NP.std(lambdas_re,axis=0)
    print("\nStandard deviation of each eigenvalue:")
    print("| i  |  std_pnt  |  std_re   |")
    for i in range(M_CONV):
        print(f"| {i + 1:<2d} | {sd_pnt[i]:<6.3e} | {sd_re[i]:<6.3e} |")
        
        
"""
Part 0: Demo.
"""

def eigen_1p(n,d_flag_name,alpha):
    
    """
    Usage:
        Compute eigenvalues at one SINGLE lattice vector.
    
    Input:
        n:           grid size.
        d_flag_name: name of diel type.
        alpha:       lattice vector.
    """

    a_fft, b_fft, diel_ind, inv_fft, x0, shift = uniform_initialization(n,d_flag_name,alpha)
    Diels = lambda x: diel.diels_chiral(EPS)(x, diel_ind)
    A_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)

    lambdas_pnt, x, iters = lobpcg_sep_nolock(A_func, P_func, x0, M_CONV)
    del x0
    
    print(f"n = {n}, lattice type: {d_flag_name}, alpha = "
          f"[{alpha[0]/pi:<5.2f}, {alpha[1]/pi:<5.2f}, {alpha[2]/pi:<5.2f}] pi, "
          f"iter = {int(iters[0])}, runtime = {iters[1]:<6.3f}s.")
    
    recompute_normalize_print(lambdas_pnt, x, A_func)

    return
        

"""
Part 1: Bandgap.
"""

def bandgap_history_check(n, d_flag, simplified = False):
    if simplified:
        bandgap_name = "output_simplified/bandgap_"+d_flag+".json"
    else:
        bandgap_name = "output/bandgap_"+d_flag+".json"
    
    var_name_it = d_flag+"_" + str(n) + "_iterations"
    if not os.path.exists(bandgap_name):        
        print("The bandgap of type ",d_flag," has no previous record.")
    else:
        with open(bandgap_name,'r') as file:
            gap_lib = json.load(file)
        if var_name_it in gap_lib.keys():
            # Previous record exists.
            print(f"Lattice type {d_flag} with grid size n = {n} has a_fft previous record.")
            gap_rec_it = gap_lib[var_name_it]
            
            # Get error, uncomputed index.
            err_ind = [i for i,a in enumerate(gap_rec_it) if a==[-1,-1]]
            if len(err_ind) > 0:
                print(f"{RED}Warning: Blow up results detected: {err_ind}.{RESET}")
                
            empty_ind = [i for i,a in enumerate(gap_rec_it) if a==[0,0]]
            if len(empty_ind) > 0:
                print(f"{YELLOW}Following indices remain uncomputed: {empty_ind}.{RESET}")
        
            if len(empty_ind) == 0 and len(err_ind) == 0:
                print(f"{GREEN}All indices of {d_flag} have been computed.{RESET}")
        else:
            print(f"{YELLOW}Lattice type {d_flag} has a record for grid size n = {n}.{RESET}")
            
    return

def bandgap(n, d_flag, solver = lobpcg_sep_nolock, indices = None):
    
    """
    Usage:
        Compute the complete (or part of) bandgap w.r.t to a_fft given lattice material with 
        certain grid size. (uniformly tensor division).
        
    Input:
        n:           Grid size.
        d_flag_name: Name of dielectric lattice.
        gap:         Segmentation of Brillouin zone.
    
    **kwargs:
        indices:     Indices of translation vectors.
        chiral_flag: If true the Diels = diels_chiral.
         
    """
    
    # Segmentation of Brillouin zones.
    gap = 20
    
    nn = n * n * n
    diel_ind, ct, sym_points = diel.diels_io(n,d_flag)
  
    bandgap_name = "../output/bandgap_"+d_flag+".json"
    def Diel0(x_in):
        x = x_in.copy()
        x[diel_ind] /= EPS
        return x 
    Diels = Diel0
    
    d_fft, di_fft = mfd.fft_blocks(n, K, ct, gpu = True)
    n_pt = sym_points.shape[0]
    n_pt -= 1
    
    if not indices:
        # Default: compute the complete bandgap.
        indices = list(range(n_pt * gap))
    if max(indices) >= n_pt*gap or min(indices) < 0:
        ValueError("Index is non-positive or is incompatible with gap size.")
    
    # Discrete lattice points.
    alphas = np.zeros((n_pt * gap, 3))
    for i in range(n_pt):
        alphas[(i + 1) * gap - 1,:] = sym_points[i + 1,:]
        for j in range(gap - 1):
            alphas[i * gap + j,:] = ((j + 1) * sym_points[i + 1,:] + (gap - j - 1) * sym_points[i, :]) / gap

    # Keywords:
    var_name_it = d_flag+"_" + str(n) + "_iterations"
    var_name_fq = d_flag+"_" + str(n) + "_frequencies"
    
    if not os.path.exists(bandgap_name):
        # New lattice bandgap plot.
        
        print("The bandgap of type ",d_flag," has no previous record.")
        
        gap_rec_it, gap_rec_fq = [[0] * 2] * (n_pt*gap), [[0] * M_CONV] * (n_pt * gap)
        gap_lib = {var_name_it: gap_rec_it, var_name_fq: gap_rec_fq}
        
        with open(bandgap_name,'w') as file:
            json.dump(gap_lib, file, indent = 4)
    else:
        with open(bandgap_name,'r') as file:
            gap_lib = json.load(file)
        if var_name_it in gap_lib.keys():
            # Previous record exists.
            print(f"Lattice type {d_flag} with grid size n = {n} has a_fft previous record.")
            gap_rec_it, gap_rec_fq = gap_lib[var_name_it], gap_lib[var_name_fq]
            
            # Get error, uncomputed index.
            err_ind = [i for i,a in enumerate(gap_rec_it) if a==[-1,-1]]
            if len(err_ind) > 0:
                print(f"{RED}Warning: Blow up results detected: {err_ind}.{RESET}")
                
            empty_ind = [i for i,a in enumerate(gap_rec_it) if a==[0,0]]
            if len(empty_ind) > 0:
                print(f"{YELLOW}Following indices remain uncomputed: {empty_ind}.{RESET}")
        
            if len(empty_ind) == 0 and len(err_ind) == 0:
                print(f"{GREEN}All indices of {d_flag} have been computed.{RESET}")
            del err_ind, empty_ind
        else:
            # New grid size.
            
            print(f"Lattice type {d_flag} will be computed with a_fft new grid size n = {n}.")
            
            gap_rec_it, gap_rec_fq = [[0] * 2] * (n_pt * gap), [[0] * M_CONV] * (n_pt*gap)
            gap_lib[var_name_it], gap_lib[var_name_fq] = gap_rec_it, gap_rec_fq
            
            with open(bandgap_name,'w') as file:
                json.dump(gap_lib, file, indent = 4)
    
    
    """
    Calculation of bandgap.
    """
    
    err_index = []
    pool = cp.get_default_memory_pool()
    
    # Main Loop: compute each lattice point.    
    for i in range(len(indices)):
        t_h = time()
        
        alpha = alphas[indices[i]] / (2 * pi)       
        relax_opt, pnt = mfd.set_relaxation(n, alpha)
        m = M_CONV + round(M_CONV * relax_opt[1])
        
        if i == 0 or abs(indices[i] - indices[i-1])>1:
            x0 = cp.random.rand(3 * nn, m) + 1j * cp.random.rand(3 * nn, m)
        elif m <= x.shape[1]:
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
        
        A_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, relax_opt[0])
        
        t_o = time()
        print(f"Matrix blocks done, {t_o-t_h:<6.3f}s elapsed.")
        
        #lambdas_pnt, x, iters = solver(A_func, P_func, x0, M_CONV, tol=TOL/SCAL/SCAL)   
           
        try:
            lambdas_pnt, x, iters = solver(A_func, P_func, x0, M_CONV, tol=TOL/SCAL/SCAL)        
        
            print(f"Gap index {indices[i]+1} out of {n_pt*gap} ({d_flag}) is computed.")
            print(f"Iterations = {int(iters[0])}, runtime = {iters[1]:<6.3f}s.\n")
        
            _ , lambdas_re = recompute_normalize_print(lambdas_pnt, x, A_func, relax_opt[0])
            
            gap_rec_it[indices[i]] = iters.tolist()
            gap_rec_fq[indices[i]] = lambdas_re.tolist()  
        
        except Exception:
            err_index.append(indices[i])
            lambdas_re = -1.0 * np.ones((M_CONV))
            x = cp.random.rand(3 * nn, m) + 1j * cp.random.rand(3 * nn, m)
            iters = np.array([-1.0, -1.0])

            gap_rec_it[indices[i]] = iters.tolist()
            gap_rec_fq[indices[i]] = lambdas_re.tolist()                
        
        del x0
        pool.free_all_blocks()
        t_h = time()
        gap_lib[var_name_it], gap_lib[var_name_fq] = gap_rec_it, gap_rec_fq
        
        with open(bandgap_name,'w') as file:
            json.dump(gap_lib, file, indent=4)
            
        t_o = owari()
        print(f"Gap info library ({d_flag}) is updated ({indices[i]}/{n_pt*gap}), time = {t_o-t_h:<6.3f}s.")
    
    if len(err_index) > 0:
        print(f"{RED}Error occurs to following indices:{RESET}")
        print(err_index)
        
    else:
        print(f"{GREEN}All indices computed correctly.{RESET}")
    
    return len(err_index) == 0

def bandgap_pseudochiral(n, d_flag, diel_mat, solver = lobpcg_sep_nolock, indices = None):
    
    """
    Input:
        n, d_flag: Same as above.
        diel_mat:  3*3 Hermitian tensor, tuple containing diag and off diag coefficients.
    """
    
    # Segmentation of Brillouin zones.
    gap = 20
    
    nn = n * n * n
    diel_ind, ct, sym_points = diel.diels_io(n,d_flag)
    
    if norm(np.array(diel_mat[1])) == 0.0:
        raise("Input of off-diag coefficients shouldn't be zero.")
    
    # We aasume only one off-diag element is nonzero.
    if diel_mat[1][0] != 0.0:
        # e12 \neq 0.0
        def Diel0(x_in):
            x1,x2,x3 = x_in[:nn].copy(), x_in[nn:2*nn].copy(), x_in[2*nn:].copy()
            x1[diel_ind] = x1[diel_ind] * diel_mat[0][0] + x2[diel_ind] * diel_mat[1][0]
            x2[diel_ind] = x1[diel_ind] * diel_mat[1][0].conj() + x2[diel_ind] * diel_mat[0][1]
            x3[diel_ind] *= diel_mat[0][0]
            return cp.concatenate((x1,x2,x3),axis = 0)
        Diels = Diel0
    elif diel_mat[1][1] != 0.0:  
        # e13 \neq 0.0
        def Diel0(x_in):
            x1,x2,x3 = x_in[:nn].copy(), x_in[nn:2*nn].copy(), x_in[2*nn:].copy()
            x1[diel_ind] = x1[diel_ind] * diel_mat[0][0] + x3[diel_ind] * diel_mat[1][1]
            x2[diel_ind] *= diel_mat[0][1]
            x3[diel_ind] = x1[diel_ind] * diel_mat[1][1].conj() + x3[diel_ind] * diel_mat[1][1]
            return cp.concatenate((x1,x2,x3),axis = 0)
        Diels = Diel0
    else:
        # e23 \neq 0.0
        def Diel0(x_in):
            x1,x2,x3 = x_in[:nn].copy(), x_in[nn:2*nn].copy(), x_in[2*nn:].copy()
            x1[diel_ind] *= diel_mat[0][0]
            x2[diel_ind] = x2[diel_ind] * diel_mat[0][1] + x2[diel_ind] * diel_mat[1][2]
            x3[diel_ind] = x2[diel_ind] * diel_mat[1][2].conj() + x3[diel_ind]
            return cp.concatenate((x1,x2,x3),axis = 0)
        Diels = Diel0
    
    bandgap_name = "output_simplified/bandgap_"+d_flag+"pseudochiral1.json"
    
    d_fft, di_fft = mfd.fft_blocks(n, K, ct, gpu = True)
    n_pt = sym_points.shape[0]
    n_pt -= 1
    
    if not indices:
        # Default: compute the complete bandgap.
        indices = list(range(n_pt * gap))
    if max(indices) >= n_pt*gap or min(indices) < 0:
        ValueError("Index is non-positive or is incompatible with gap size.")
    
    # Discrete lattice points.
    alphas = np.zeros((n_pt * gap, 3))
    for i in range(n_pt):
        alphas[(i + 1) * gap - 1,:] = sym_points[i + 1,:]
        for j in range(gap - 1):
            alphas[i * gap + j,:] = ((j + 1) * sym_points[i + 1,:] + (gap - j - 1) * sym_points[i, :]) / gap

    # Keywords:
    var_name_it = d_flag+"_" + str(n) + "_iterations"
    var_name_fq = d_flag+"_" + str(n) + "_frequencies"
    
    if not os.path.exists(bandgap_name):
        # New lattice bandgap plot.
        
        print("The bandgap of type ",d_flag," has no previous record.")
        
        gap_rec_it, gap_rec_fq = [[0] * 2] * (n_pt*gap), [[0] * M_CONV] * (n_pt * gap)
        gap_lib = {var_name_it: gap_rec_it, var_name_fq: gap_rec_fq}
        
        with open(bandgap_name,'w') as file:
            json.dump(gap_lib, file, indent = 4)
    else:
        with open(bandgap_name,'r') as file:
            gap_lib = json.load(file)
        if var_name_it in gap_lib.keys():
            # Previous record exists.
            print(f"Lattice type {d_flag} with grid size n = {n} has a_fft previous record.")
            gap_rec_it, gap_rec_fq = gap_lib[var_name_it], gap_lib[var_name_fq]
            
            # Get error, uncomputed index.
            err_ind = [i for i,a in enumerate(gap_rec_it) if a==[-1,-1]]
            if len(err_ind) > 0:
                print(f"{RED}Warning: Blow up results detected: {err_ind}.{RESET}")
                
            empty_ind = [i for i,a in enumerate(gap_rec_it) if a==[0,0]]
            if len(empty_ind) > 0:
                print(f"{YELLOW}Following indices remain uncomputed: {empty_ind}.{RESET}")
        
            if len(empty_ind) == 0 and len(err_ind) == 0:
                print(f"{GREEN}All indices of {d_flag} have been computed.{RESET}")
            del err_ind, empty_ind
        else:
            # New grid size.
            
            print(f"Lattice type {d_flag} will be computed with a_fft new grid size n = {n}.")
            
            gap_rec_it, gap_rec_fq = [[0] * 2] * (n_pt * gap), [[0] * M_CONV] * (n_pt*gap)
            gap_lib[var_name_it], gap_lib[var_name_fq] = gap_rec_it, gap_rec_fq
            
            with open(bandgap_name,'w') as file:
                json.dump(gap_lib, file, indent = 4)
    
    
    """
    Calculation of bandgap.
    """
    
    err_index = []
    pool = cp.get_default_memory_pool()
    
    # Main Loop: compute each lattice point.    
    for i in range(len(indices)):
        t_h = time()
        
        alpha = alphas[indices[i]] / (2 * pi)       
        relax_opt, pnt = mfd.set_relaxation(n, alpha)
        m = M_CONV + round(M_CONV * relax_opt[1])
        
        if i == 0 or abs(indices[i] - indices[i-1])>1:
            x0 = cp.random.rand(3 * nn, m) + 1j * cp.random.rand(3 * nn, m)
        elif m <= x.shape[1]:
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
        
        A_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, relax_opt[0])
        
        t_o = time()
        print(f"Matrix blocks done, {t_o-t_h:<6.3f}s elapsed.")
        
        #lambdas_pnt, x, iters = solver(A_func, P_func, x0, M_CONV, tol=TOL/SCAL/SCAL)   
           
        try:
            lambdas_pnt, x, iters = solver(A_func, P_func, x0, M_CONV, tol=TOL/SCAL/SCAL)        
        
            print(f"Gap index {indices[i]+1} out of {n_pt*gap} ({d_flag}) is computed.")
            print(f"Iterations = {int(iters[0])}, runtime = {iters[1]:<6.3f}s.\n")
        
            _ , lambdas_re = recompute_normalize_print(lambdas_pnt, x, A_func, relax_opt[0])
            
            gap_rec_it[indices[i]] = iters.tolist()
            gap_rec_fq[indices[i]] = lambdas_re.tolist()  
        
        except Exception:
            err_index.append(indices[i])
            lambdas_re = -1.0 * np.ones((M_CONV))
            x = cp.random.rand(3 * nn, m) + 1j * cp.random.rand(3 * nn, m)
            iters = np.array([-1.0, -1.0])

            gap_rec_it[indices[i]] = iters.tolist()
            gap_rec_fq[indices[i]] = lambdas_re.tolist()                
        
        del x0
        pool.free_all_blocks()
        t_h = time()
        gap_lib[var_name_it], gap_lib[var_name_fq] = gap_rec_it, gap_rec_fq
        
        with open(bandgap_name,'w') as file:
            json.dump(gap_lib, file, indent=4)
            
        t_o = owari()
        print(f"Gap info library ({d_flag}) is updated ({indices[i]}/{n_pt*gap}), time = {t_o-t_h:<6.3f}s.")
    
    if len(err_index) > 0:
        print(f"{RED}Error occurs to following indices:{RESET}")
        print(err_index)
        
    else:
        print(f"{GREEN}All indices computed correctly.{RESET}")
    
    return len(err_index) == 0
        
"""
Part 2: Speedup.
"""

def speedup(Ns, diel_name, alpha = np.array([pi, pi, pi])):
    
    """
    Cupy speedup test.
    """
    
    # Packs = {gpu_pack, cpu_pack}. 
    # cpu: numpy,scipy. gpu: cupy,cupyx.
    
    runtime_name = "output/speedup_" + diel_name + ".json"
    
    if not os.path.exists(runtime_name):
        runtime_pack_lib={}
    else:
        with open(runtime_name,'r') as file:
            runtime_pack_lib=json.load(file)

    n_Ns  = len(Ns)
    iters = np.zeros((n_Ns,2,2))
    
    for i in range(n_Ns):
        a_fft, b_fft, diel_ind, inv_fft, x0, shift = uniform_initialization(Ns[i], diel_name, alpha)
        Diels = lambda x: diel.diels_chiral(EPS)(x, diel_ind)
        A_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)

        lambdas_pnt, x, iters[i,0,:] = lobpcg_sep_nolock(A_func, P_func, x0, M_CONV)
        print(f"\nN = {Ns[i]}, gpu (cupy/cupyx.scipy) is done computing, runtime = {iters[i,0,1]:<6.3f}s.\n")
        _ ,_ = recompute_normalize_print(lambdas_pnt, x, A_func)
        
        a_fft, b_fft, inv_fft, x0 = a_fft.get(), b_fft.get(), inv_fft.get(), x0.get()
        A_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)
        
        lambdas_pnt, x, iters[i,0,:] = lobpcg_sep_nolock(A_func, P_func, x0, M_CONV)
        print(f"\nN {Ns[i]}, cpu (numpy/scipy) is done computing, runtime {iters[i,1,1]:<6.3f}s.\n")
        _ ,_ = recompute_normalize_print(lambdas_pnt, x, A_func, shift)
        
        runtime_pack_lib["pack_cmp_"+str(Ns[i])]=\
            [iters[i,0,0],iters[i,1,1],iters[i,0,1],iters[i,1,1]/iters[i,0,1]]
            
        with open(runtime_name,'w') as file:
            json.dump(runtime_pack_lib,file,indent=4)

    print("\nTesting lattice type: ",diel_name)
    print("Runtime comparison using different linear algebra packages:")
    for i in range(n_Ns):
        print(f"n = {Ns[i]}, iterations = {int(iters[i,0,0])}, cputime = {iters[i,1,1]:<6.3f}s, gputime = {iters[i,0,1]:<6.3f}s, "
              f"ratio = {iters[i,1,1]/iters[i,0,1]:<6.3f}.")
      
    return


def main():
    #  Order: CUDA_VISIBLE_DEVICES=1 python numerical_experiment.py

    eigen_1p(120,"sc_curv",np.array([pi,pi,pi]))
    #bandgap(100,"sc_curv", indices=[39])
    #pool = cp.get_default_memory_pool()
    #pool.free_all_blocks()
    #bandgap(100,"sc_curv", chiral  = ["sc_curv_pseudochiral2", 13.0, 0.875])
    #bandgap(120, "bcc_single_gyroid", indices=[18, 20, 98, 100])
    #bandgap(120, "bcc_double_gyroid", indices=[18, 20, 98, 100])
    #bandgap(120, "fcc", indices=[60])
    #eigen_1p(100,"sc_curv",np.array([pi,pi,pi]))
    
    #bandgap_history_check(100,"sc_curv",simplified=True)
    #bandgap_history_check(120,"sc_curv",simplified=True)
    #bandgap_history_check(150,"sc_curv",simplified=True)
if __name__ == '__main__':
    main()