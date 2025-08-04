# -*- coding: utf-8 -*-
#
# Created on 2025-02-24 (Monday) at 20:34:41
#
# Author: Epsilon-79th
#
# Usage: Test functions of paper 2.
#

import numpy as np
from numpy import pi
import cupy as cp
import cmath
from cupyx.scipy.sparse import csr_matrix

import discretization as mfd
import dielectric as diel
from paper_1_test import *
from numerical_experiment import *
from gpu_opts import *
from lobpcg import *
from davidson import *

def pseudochiral_test_nedelec(n, alpha = np.array([pi, pi, pi])):
    
    """
    Test the pseudochiral model with different initial vectors.
    """
    
    nn = n * n * n
    a_fft, b_fft, e_dofs, inv_fft, x0, shift = uniform_initialization(n, "sc_curv", alpha)

    # Load volume and edge dofs.
    v_dofs = diel.diels_io(n, d_flag="sc_curv", dofs="volume")[0]
    n_v = v_dofs.shape[0]

    e_x = e_dofs[e_dofs < nn].astype(cp.int32)
    e_y = e_dofs[(e_dofs >= nn) & (e_dofs < 2*nn)].astype(cp.int32)
    e_z = e_dofs[e_dofs >= 2*nn].astype(cp.int32)

    #eps_loc = np.random.rand(3,3)+1j* np.random.rand(3,3)
    #eps_loc = (eps_loc + eps_loc.T.conj()) / 2+np.eye(3)*10.0

    eps_loc = np.array([[(1+0.875**2)**0.5, -1j*0.875, 0], [1j*0.875, (1+0.875**2)**0.5, 0], [0,0,1.0]], dtype=cp.complex128)/13.0

    vals = cp.tile(cp.array([3,3**0.5,3,3**0.5,3**0.5,1,3**0.5,1])/4, n_v)
    ve_x, ve_y, ve_z = cp.asarray(diel.mesh3d_offdiagonal_dofs(n, d_flag="sc_curv"))

    M12 = csr_matrix((vals * eps_loc[0,1], (cp.repeat(ve_x,2), cp.tile(ve_y,2))), shape=(nn, nn))
    M13 = csr_matrix((vals * eps_loc[0,2], (cp.repeat(ve_x,2), cp.tile(ve_z,2))), shape=(nn, nn))
    M23 = csr_matrix((vals * eps_loc[1,2], (cp.repeat(ve_y,2), cp.tile(ve_z,2))), shape=(nn, nn))
    
    def Diels(x_in):

        x = x_in.copy()

        # Digonal blocks.
        x*=4
        x[e_x] *= eps_loc[0,0]
        x[e_y] *= eps_loc[1,1]
        x[e_z] *= eps_loc[2,2]
        
        # Off-diagonal blocks.
        x[:nn] += M12 @ x_in[nn:2*nn] + M13 @ x_in[2*nn:]
        x[nn:2*nn] += M12.T.conj() @ x_in[:nn] + M23 @ x_in[2*nn:]
        x[2*nn:] += M13.T.conj() @ x_in[:nn] + M23.T.conj() @ x_in[nn:2*nn]
        return x
    
    nev=10
    x0_eig = cp.random.rand(3*nn, nev+4) + 1j * cp.random.rand(3*nn, nev+4)
    eigvals, _, _ = lobpcg_sep_nolock(Diels, lambda x: x, x0_eig, nev)
    print("Diels 10 smallest eigenvalues:", eigvals.get() if hasattr(eigvals, 'get') else eigvals)
    

    #A_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)
    
    #lambdas_pnt, x1, iters = lobpcg_sep_nolock(A_func, P_func, x0, M_CONV)
    #_ , lambdas1 = recompute_normalize_print(lambdas_pnt, x1, A_func, shift)
    
    
    
    return

def precision_cmp(n, diel_name, alpha = np.array([pi, pi, pi])):
    
    """
    Usage:
        Double precision & Single precision.
    
    Input:
        n: Size.
        diel_name:
    """
    
    a_fft, b_fft, diel_ind, inv_fft, x0, shift = uniform_initialization(n, diel_name, alpha)
    Diels = lambda x: diel.diels_chiral(EPS)(x, diel_ind)
    A_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)
    
    lambdas_double, x_double, iters_double = lobpcg_sep_nolock(A_func, P_func, x0, M_CONV)
    _ , lambdas_re_d = recompute_normalize_print(lambdas_double, x_double, A_func, shift)
    
    a_fft = a_fft.astype(cp.complex64)
    b_fft = (b_fft[0].real.astype(cp.float32), b_fft[1].astype(cp.complex64))
    
    x0 = cp.random.rand(3*n*n*n, M_CONV, dtype=cp.float32) + 1j * cp.random.rand(3*n*n*n, M_CONV, dtype=cp.float32)
    A_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft)
    
    lambdas_single, x_single, iters_single = lobpcg_sep_nolock(A_func, P_func, x0, M_CONV)
    _ , lambdas_re_s = recompute_normalize_print(lambdas_single, x_single, A_func)
    NP = arrtype(lambdas_re_d)
    l_diff, x_diff = NP.abs(lambdas_re_d - lambdas_re_s), norms(x_double - x_single) / norms(x_double)
    
    print(f"Double: ({int(iters_double[0])}, {iters_double[1]:<6.3f}s).")
    print(f"Double: ({int(iters_single[0])}, {iters_single[1]:<6.3f}s).")
    for i in range(M_CONV):
        print(f"i = {i + 1:<4d}, lambda_diff = {l_diff[i]:<6.3e}, x_diff = {x_diff[i]:<6.3e}")
    
    return

def eigenvector_cmp(n, diel_name, alpha = np.array([pi, pi, pi])):
    
    """
    Usage:
        Results with different initial vectors.
    
    This experiment focuses on eigenvectors with different initial guess.
    It implies that when eigenvalue is simple, the corresponding eigenvectors are linearly dependent.
    e.g: x0,1 --> x1, x0,2 --> x2, then |x1-zx2| is small, with z = re^(i\theta), r close to 1.
    """
    
    
    a_fft, b_fft, diel_ind, inv_fft, x0, shift = uniform_initialization(n, diel_name, alpha)
    Diels = lambda x: diel.diels_chiral(EPS)(x, diel_ind)
    A_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)
    lambdas, x1, _  = lobpcg_sep_nolock(A_func, P_func, x0, M_CONV)
    _ , lambdas1 = recompute_normalize_print(lambdas, x1, A_func, shift)
    
    n1, n2 = x0.shape
    x0 = cp.random.rand(n1, n2) + 1j * cp.random.rand(n1, n2)
    lambdas, x2, _  = lobpcg_sep_nolock(A_func, P_func, x0, M_CONV)
    _ , lambdas2 = recompute_normalize_print(lambdas, x1, A_func, shift)
    
    l_diff = cp.abs(lambdas1 - lambdas2)
    for i in range(M_CONV):
        z = x2[0, i] / x1[0, i]
        r, c = abs(z), cmath.phase(z)
        x_diff = cp.linalg.norm(x1[:, i] * x2[0, i] / x1[0, i] - x2[:, i])
        print(f"i = {i + 1:<4d}, lambda_diff = {l_diff[i]:<6.3e}, x_diff = {x_diff:<6.3e}, "
              f"<x1, x2> = ({r:<6.3f}, {c/pi:<6.3f}pi).")
    return

def eigensolver_test(n, diel_name, eigen_name, alpha = np.array([pi, pi, pi])):
    
    eps = 13.0+1.0j
    solver = eval(eigen_name)
    a_fft, b_fft, diel_ind, inv_fft, x0, shift = uniform_initialization(n, diel_name, alpha)
    Diels = lambda x: diel.diels_chiral(eps)(x, diel_ind)
    A_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)
    lambdas_pnt, x, iters = solver(A_func, P_func, x0, M_CONV)
    _ , _ = recompute_normalize_print(lambdas_pnt, x, A_func)
    
    print(f"Model: {diel_name}_{n}, iterations = {int(iters[0])}, runtime = {iters[1]:<6.3f}s.")
    
    return

def largek_cmp(Ns):
    
    m0, m = 5,7
    results = np.zeros((2,len(Ns)))
    
    for i in range(len(Ns)):
        N = Ns[i]
        k = int(N**(1/3))
        a_fft, b_fft, diel_ind, inv_fft, x0 , _ = uniform_initialization(N, "sc_curv", np.array([pi,pi,pi]),k=k)
        Diels = lambda x: diel.diels_chiral(13.0)(x, diel_ind)
        A_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift=0.0)
        lambdas, _ , _ = lobpcg_sep_nolock(A_func, P_func, x0[:,:m], m0)
        #_ , _ = recompute_normalize_print(lambdas, x, A_func)
        lambdas=SCAL*np.sqrt(lambdas)
        
        results[0,i], results[1,i] = np.sum(lambdas[:3])/3, np.sum(lambdas[3:])/2
        print(f"N = {N} is done computing.")
    
    print(results)  
    results.tofile("output/precision_test.bin")
    return

def main():
    #precision_cmp(100, "sc_curv")
    #eigenvector_cmp(100, 'sc_curv')
    pseudochiral_test_nedelec(100)
    #largek_cmp([80,100,120,150,180])
    #eigensolver_test(100, "sc_curv", "davidson_sep_nonhermitian")
    
if __name__ == "__main__":
    main()

