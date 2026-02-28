# -*- coding: utf-8 -*-
#
# Created on 2025-02-24 (Monday) at 20:34:41
#
# Author: Epsilon-79th
#
# Usage: Test functions of paper 2.
#        Pseudochiral.
#

import numpy as np
from numpy import pi
import cupy as cp
import cmath

from environment import *
import discretization as mfd
from numerical_experiments import *
from lobpcg import *
from orthogonalization import *

def global_precision_cmp(n, d_flag, alpha = np.array([pi, pi, pi])):
    
    """
    Usage:
        Double precision & Single precision.
    
    Warning:
        Global single precision may lead to instability, stagnation, nan.
    """
    
    a_fft, b_fft, inv_fft, x0, shift = uniform_initialization(n, d_flag, alpha)
    Diels = mfd.chiral_handle(n, d_flag)
    A_func, H_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)
    
    lambdas_d, x_d, iters_d = lobpcg_sep_softlock(H_func, P_func, x0, NEV)
    _ , lambdas_re_d = recompute_normalize_print(lambdas_d, x_d, A_func, shift)
    
    a_fft_s = a_fft.astype(cp.complex64)
    b_fft_s = (b_fft[0].real.astype(cp.float32), b_fft[1].astype(cp.complex64))
    inv_fft_s = (inv_fft[0].real.astype(cp.float32), inv_fft[1].astype(cp.complex64))
    
    x0_s = x0.astype(cp.complex64)
    A_func_s, H_func_s, P_func_s = pc_mfd_handle(a_fft_s, b_fft_s, Diels, inv_fft_s)
    
    lambdas_s, x_s, iters_s = lobpcg_sep_softlock(H_func_s, P_func_s, x0_s, NEV, longortho=True,singleprecision=True)
    _ , lambdas_re_s = recompute_normalize_print(lambdas_s, x_s, A_func_s)
    l_diff, x_diff = cp.abs(lambdas_re_d - lambdas_re_s), norms(x_d - x_s) / norms(x_d)
    
    print(f"Double: ({int(iters_d[0])}, {iters_d[1]:<6.3f}s).")
    print(f"Single: ({int(iters_s[0])}, {iters_s[1]:<6.3f}s).")
    for i in range(NEV):
        print(f"i = {i + 1:<4d}, lambda_diff = {l_diff[i]:<6.3e}, x_diff = {x_diff[i]:<6.3e}")
    
    return

def partial_precision_cmp(n, d_flag, alpha = np.array([pi, pi, pi])):

    """
    Usage:
        Double precision & Single precision.
    
    Single precision for preconditioning doesn't affect accuracy much.
    """
    
    a_fft, b_fft, inv_fft, x0, shift = uniform_initialization(n, d_flag, alpha)
    Diels = mfd.chiral_handle(n, d_flag)
    A_func, H_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)
    
    lambdas_d, x_d, iters_d = lobpcg_sep_softlock(H_func, P_func, x0, NEV)
    _ , lambdas_re_d = recompute_normalize_print(lambdas_d, x_d, A_func, shift)
    
    inv_fft_s = (inv_fft[0].real.astype(cp.float32), inv_fft[1].astype(cp.complex64))
    P_func_s = lambda x: H_block(x, inv_fft_s)    
    
    lambdas_s, x_s, iters_s = lobpcg_sep_softlock_mixedprecision(H_func, P_func_s, x0, NEV)
    _ , lambdas_re_s = recompute_normalize_print(lambdas_s, x_s, A_func)
    l_diff, x_diff = cp.abs(lambdas_re_d - lambdas_re_s), norms(x_d - x_s * x_d[0,:]/x_s[0,:])
    
    print(f"Double: ({int(iters_d[0])}, {iters_d[1]:<6.3f}s).")
    print(f"Single: ({int(iters_s[0])}, {iters_s[1]:<6.3f}s).")
    for i in range(NEV):
        print(f"i = {i + 1:<4d}, lambda_diff = {l_diff[i]:<6.3e}, x_diff = {x_diff[i]:<6.3e}")
    
    return

def eigenvector_cmp(n, d_flag, alpha = np.array([pi, pi, pi])):
    
    """
    Usage:
        Results with different initial vectors.
    
    This experiment focus on eigenvectors with different initial guess.
    It implies that when eigenvalue is simple, the corresponding eigenvectors are linearly dependent.
    e.g: x0,1 --> x1, x0,2 --> x2, then |x1-zx2| is small, with z = re^(i\theta), r close to 1.
    """
    
    a_fft, b_fft, inv_fft, x0, shift = uniform_initialization(n, d_flag, alpha)
    Diels = mfd.chiral_handle(n, d_flag)
    A_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)
    lambdas, x1, _  = lobpcg_sep_softlock(A_func, P_func, x0, NEV)
    _ , lambdas1 = recompute_normalize_print(lambdas, x1, A_func, shift)
    
    n1, n2 = x0.shape
    x0 = cp.random.rand(n1, n2) + 1j * cp.random.rand(n1, n2)
    lambdas, x2, _  = lobpcg_sep_softlock(A_func, P_func, x0, NEV)
    _ , lambdas2 = recompute_normalize_print(lambdas, x1, A_func, shift)
    
    l_diff = cp.abs(lambdas1 - lambdas2)
    for i in range(NEV):
        z = x2[0, i] / x1[0, i]
        r, c = abs(z), cmath.phase(z)
        x_diff = cp.linalg.norm(x1[:, i] * z - x2[:, i])
        print(f"i = {i + 1:<4d}, lambda_diff = {l_diff[i]:<6.2e}, x_diff = {x_diff:<6.2e}, "
              f"<x1, x2> = ({r:<6.2f}, {c/pi:<6.2f}pi).")
    return

def largek_cmp(Ns):
    
    m0, m = 4,5
    results = np.zeros(len(Ns))

    N2k = lambda N: round(16.30*np.log(N-10)-58.12)
    
    for i in range(len(Ns)):
        N = Ns[i]
        k = N2k(N)
        a_fft, b_fft, inv_fft, _ , _ = uniform_initialization(N, "sc_curv", np.array([pi,pi,pi]),k=k)
        x0 = cp.random.rand(3*N*N*N,m) + 1j * cp.random.rand(3*N*N*N,m)
        Diels = mfd.chiral_handle(N, "sc_curv")
        A_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift=0.0)
        lambdas, x , _ = lobpcg_sep_nolock(A_func, P_func, x0, m0)
        _ , lambdas = recompute_normalize_print(lambdas, x, A_func)
        results[i] = lambdas[2]

        #results[0,i], results[1,i] = np.sum(lambdas[:3])/3, np.sum(lambdas[3:])/2

        print(f"N = {N} is done computing.")
    
    henka = np.abs(results[1:] - results[:-1])
    for i in range(len(Ns)-1):
        print(f"{henka[i]:<6.3e}")
    
    return


"""
    Tests
"""

def edge_volume_index_cmp(n = 100, d_flag=SC_C):

    """
    Conclusion: when N=100, around 1% edge/volume index mismatch.
    """

    nn = n * n * n
    ind_e = diel.diel_io_index(n, d_flag, dofs = 'edge').get()
    ind_v = diel.diel_io_index(n, d_flag, dofs = 'volume').get()

    flag_v = np.zeros(nn, dtype=bool)
    flag_v[ind_v] = True

    flag_ex, flag_ey, flag_ez = [np.zeros(nn, dtype=bool) for _ in range(3)]
    flag_ex[ind_e[ind_e < nn]] = True
    flag_ey[ind_e[(ind_e >= nn) & (ind_e < 2*nn)] - nn] = True
    flag_ez[ind_e[ind_e >= 2*nn] - 2 * nn] = True

    n_xv = len(np.where(flag_ex != flag_v)[0])
    n_yv = len(np.where(flag_ey != flag_v)[0])
    n_zv = len(np.where(flag_ez != flag_v)[0])
    print(f"Number/Ratio of different x-edge and volume indices: {n_xv}, {n_xv/nn:<6.3e}.")
    print(f"Number/Ratio of different y-edge and volume indices: {n_yv}, {n_yv/nn:<6.3e}.")
    print(f"Number/Ratio of different z-edge and volume indices: {n_zv}, {n_zv/nn:<6.3e}.")

    print("When volume index is True,")
    for i1 in range(0,2):
        for i2 in range(0,2):
            for i3 in range(0,2):
                n_tmp = len(np.where((flag_v) & (flag_ex == i1) & (flag_ey == i2) & (flag_ez == i3))[0])
                print(f"({i1},{i2},{i3}), number = {n_tmp}.")
    
    n_e = [len(np.where(flag_ex!=flag_ey)[0]), len(np.where(flag_ex!=flag_ez)[0]), len(np.where(flag_ez!=flag_ey)[0])]
    print(f"Number of different edge-edge indices: {n_e}.")
    return

def dmat_cmp(n, types, d_flag = SC_C, k = 1):

    """
    Compare D-matrix generated by two methods.
    """

    dmat_1 = eval("mfd."+types[0]+"_handle")(n, d_flag, flag_mat=True)
    dmat_2 = eval("mfd."+types[1]+"_handle")(n, d_flag, flag_mat=True)

    print(f"{d_flag} lattice comparison: {types[0]} & {types[1]}.")
    diff_mat = dmat_1 - dmat_2
    diff_mat.eliminate_zeros()

    if diff_mat.nnz == 0:
        print("Two D-matrices are identical.")
        return

    diff = cp.abs(diff_mat.data)
    print(f"Info: size = {3*n*n*n}, nnz = {len(diff)}, fro = {cp.linalg.norm(diff):<6.3e}, "
          f"max_nz = {cp.max(diff):<6.3e}, min_nz = {cp.min(diff):<6.3e}.")
    
    diff_func = lambda x: diff_mat.getH() @ (diff_mat @ x)
    rho = power_method(diff_func, x0 = cp.random.rand(3*n*n*n)+1j*cp.random.rand(3*n*n*n))
    print(f"Spectrum radius = {rho**0.5:<6.3e}.")
    return

def check_sdd(mat_in):

    mat_diag = mat_in.diagonal()
    mat_abs = mat_in.copy()
    mat_abs.data = cp.abs(mat_in.data)
    sums = mat_abs.sum(axis=1).flatten()

    ind = np.where(mat_diag <= sums-mat_diag)[0]
    print(f"SDD not satisfied n_row = {len(ind)}.")

    return

def check_pseudochiral_crossdof_sdd(N, k, d_flag = SC_C):

    """
    Usage:
        Check whether crossdof_mal/her is SDD.
    """

    mat_her = mfd.pseudochiral_crossdof_handle(N, k=k, d_flag=d_flag, flag_mat=True)
    check_sdd(mat_her)

    return

def check_component_HPD(N, k, eps_no='0'):

    d3 = diel.diel_pseudochiral_const(eps_no)
    nn = N*N*N
    mat = mfd.pseudochiral_crossdof_handle(N, d_flag=SC_C, eps_no=eps_no,flag_mat=True)
    mat_diag=cp.ones(3*nn)
    mat_diag[:nn] = d3[0].real
    mat_diag[nn:2*nn] = d3[1].real
    mat_diag[2*nn:] = d3[2].real
    mat.setdiag(mat_diag)
    
    eig_s, iters = lobpcg_default(mat, prec = lambda x: x, nev = 2, rlx=4, info = True)
    print(f"Eig_s done, runtime = {iters[1]:<6.3f}s, {eig_s}.")

    return

def bandgap_pseudo_cmp(n, d_flag, eps_opt = 0):

    """
    Usage:
        Compare bandgaps of M matrix defined by pseudochiral trivial and crossdof.
    """

    path0 = "output/chiral/bandgap_" + d_flag + str(eps_opt) + ".json"
    path1 = "output/pseudochiral_trivial/bandgap_" + d_flag + str(eps_opt) + ".json"
    path2 = "output/pseudochiral_crossdof/bandgap_" + d_flag + str(eps_opt) + ".json"

    with open(path0, 'r') as file:
        gap_lib0 = json.load(file)
    with open(path1, 'r') as file:
        gap_lib1 = json.load(file)
    with open(path2, 'r') as file:
        gap_lib2 = json.load(file)
    
    var_name_it = d_flag + "_" + str(n) + "_iterations"
    var_name_fq = d_flag + "_" + str(n) + "_frequencies"
    iters0 = np.array(gap_lib0[var_name_it]) 
    iters1 = np.array(gap_lib1[var_name_it])
    iters2 = np.array(gap_lib2[var_name_it])

    fq0 = np.array(gap_lib1[var_name_fq])
    fq1 = np.array(gap_lib1[var_name_fq])
    fq2 = np.array(gap_lib2[var_name_fq])
    fq0 = fq1[np.abs(fq0) > 1e-5]
    fq1 = fq1[np.abs(fq1) > 1e-5]
    fq2 = fq2[np.abs(fq2) > 1e-5]
    fq_diff = np.abs(fq1 - fq2) / fq2

    print(f"max = {np.max(fq_diff):<6.3e}, min = {np.min(fq_diff):<6.3e}, mean = {np.mean(fq_diff):<6.3e}.")
    print(f"Average iterations of chiral: {np.mean(iters0[:,0]):<6.3f}, trivial: {np.mean(iters1[:,0]):<6.3f}, crossdof: {np.mean(iters2[:,0]):<6.3f}.")

    # Deviation
    print(f"Deviation of chiral: {np.std(iters0[:,0]):<6.3f},trivial: {np.std(iters1[:,0]):<6.3f}, crossdof: {np.std(iters2[:,0]):<6.3f}.")

    return

def compute_extreme_case(n, d_flag = SC_C, type = TYPE1):

    # A randomly chosen alpha with acute angles.
    alpha = np.array([pi/7, 3*pi/5, 4*pi/13])

    # A randomly chosen anisotropic eps_mat.
    eps_mat = np.array([[1/16,0,0],[0,1/64,0],[0,0,-1/256]])
    Umat = np.random.rand(3,3) + 1j * np.random.rand(3,3)
    Umat, _ = np.linalg.qr(Umat)
    eps_mat = Umat @ eps_mat @ Umat.T.conj()
    eps_mat = np.array([eps_mat[0,0], eps_mat[1,1], eps_mat[2,2], eps_mat[0,1], eps_mat[0,2], eps_mat[1,2]])

    a_fft, b_fft, inv_fft, x0, shift = uniform_initialization(n, d_flag, alpha)
    Diels = eval("mfd."+type+"_handle")(n, d_flag, eps_mat = eps_mat)
    A_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)
    lambdas, x, info = lobpcg_sep_softlock(A_func, P_func, x0, NEV, tol = 1e-9, iter_max = int(1e4), history=True)
    _ , lambdas = recompute_normalize_print(lambdas, x, A_func)
    print(f"Extreme case: {d_flag}, {type}, n={n}, iterations = {int(info[0])}, runtime = {info[1]:<6.3f}s.")

    path = "../output/" + type + "/info_" + d_flag + ".bin"
    info.tofile(path)

    return

def precision_test(d_flag=SC_C, alpha=np.array([pi,pi,pi])):

    """
    Usage:
        Grid N = 16,32,64,128.
    """

    N2k = lambda N: max(2,round(16.30*np.log(N-10)-58.12))

    Ns = [16,32,64,128]
    n_Ns=len(Ns)
    lambdas = cp.zeros((n_Ns,NEV))
    iters = np.zeros((n_Ns,2))
    
    for i in range(n_Ns):
        a_fft, b_fft, inv_fft, x0, shift = uniform_initialization(Ns[i],d_flag,alpha,k=5)
        Diels = mfd.pseudochiral_crossdof_handle(Ns[i], d_flag=d_flag)
        #Diels = mfd.chiral_handle(Ns[i],d_flag=d_flag)
        A_func, H_func, P_func = pc_mfd_handle(a_fft, b_fft, Diels, inv_fft, shift)
        lambdas[i,:], x, iters[i,:] = lobpcg_sep_softlock(H_func, P_func, x0, NEV)

        print(f"\nGrid size= {Ns[i]} is done computing.\n")
        lambdas[i,:], _ = recompute_normalize_print(lambdas[i,:], x, A_func, shift) 

    lambdas = lambdas.get()
    print(f"Convergence results ({d_flag}), {alpha}:")
    for i in range(n_Ns):
        print(f"n = {Ns[i]}, iterations = {int(iters[i,0])}, runtime = {iters[i,1]:<5.2f}s.")
    
    print("\nPrecision results:")
    for i in range(NEV):
        print(f"{i+1:<4d}:",end = " ")
        for j in range(n_Ns-1):
            print(f"{np.abs(lambdas[j+1,i]-lambdas[j,i]):<10.2e}", end = ",")
        print(f" average order = {np.log(np.abs(lambdas[1,i]-lambdas[0,i])/np.abs(lambdas[-1,i]-lambdas[-2,i]))/np.log(2)/(n_Ns-2):<6.2f}.")

    ord_avg = np.log(norm(lambdas[1,:]-lambdas[0,:]) / norm(lambdas[-1,:]-lambdas[-2,:]))/np.log(2)/(n_Ns-2)
    print(f"Global average order = {ord_avg:<6.2f}.")
    return

def main():

    compute_extreme_case(120, d_flag=SC_C, type = TYPE1)
    compute_extreme_case(120, d_flag=SC_C, type = TYPE2)
    return
    
if __name__ == "__main__":
    main()

