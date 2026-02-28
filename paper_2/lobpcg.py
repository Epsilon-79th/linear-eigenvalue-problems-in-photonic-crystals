# -*- coding: utf-8 -*-
#
# Created on 2025-03-10 (Monday) at 16:40:40
#
# Author: Epsilon-79th
#
# Usage: LOBPCG and its variants.
#        lobpcg_sep_softlock: RECOMMENDED for solving SEP !!!
# Optimization: _update_after_rr, done by cupy elementwise kernels.
# Note: all the inputs and outputs are cupy arrays.
# 

import cupy as cp
import numpy as np
import scipy as sc
import gc

from orthogonalization import *
from environment import *
from _kernels import *
owari = owari_cuda

r"""
Default: find several minimum eigenvalues of Ax=\lambda x,
         A is a scipy or cupy csr matrix, no preconditioner.
"""

def lobpcg_default(A, nev = 20, rlx = 4, prec = lambda x:x, eigvec = False, info = False, maxmin = "min"):

    """
    Input:
        A: a 2-tuple with function handle and size or cupy/scipy csr matrix.
        When maxmin = "max", preconditioner is trivial.
    """

    if isinstance(A, tuple):
        h_func,n = A[0], A[1]
    else:
        if isinstance(A, sc.sparse.csr.csr_matrix):
            A_gpu = cp.sparse.csr_matrix(A)
        else:
            A_gpu = A.copy()
        n = A_gpu.shape[0]
        h_func = lambda x: A_gpu @ x
    
    x0 = cp.random.rand(n, nev+rlx) + 1j * cp.random.rand(n, nev+rlx)
    if maxmin == "min":
        eigv, eigV , it = lobpcg_sep_softlock(h_func, prec, x0, nev, tol=TOL, maxiter=MAXITER)
    elif maxmin == "max":
        eigv, eigV , it = lobpcg_sep_max_nolock(h_func, x0, nev, tol=TOL, maxiter=MAXITER)
    else:
        raise ValueError("maxmin should be 'min' or 'max'.")

    if eigvec and info:
        return eigv, eigV, it
    elif eigvec and not info:
        return eigv, eigV
    elif not eigvec and info:
        return eigv, it 
    else:
        return eigv

r"""
Environment:
    GPU, cupy, cupyx.

Input:
    Function handle of A,B (Ax=\lamba Bx, for sep B=I), preconditioner P.
    Parameters of convergence: nev, tol, maxiter, etc.

Output:
    Eigenvalues, eigenvectors and iterations (times, time.time).

"""

def lobpcg_sep_nolock(
        h_func, p_func, x0, nev, 
        tol = TOL, maxiter = MAXITER, history = False, 
        longortho = False, singleprecision = False):

    """
    Usage:
        LOBPCG with softlocking, [X,W,P] stored as a whole in S.
    Input:
        h_func, p_func:  function handle of matrix/preconditioning.
        x0, nev:      initial guess and nev.
        tol, maxiter:    convergence parameters.
        history:         whether to return the residual history.
        longortho:       whether to use long orthogonalization in RR.
        singleprecision: use single precision in computation.
        (Warning1: when using single precision, all the handles should also support SP.)
        (Warning2: global single precision may lead to instability, stagnation, nan.)
    """

    t_h = time.time()
    m = x0.shape[1]
    if singleprecision:
        real_type, complex_type = cp.float32, cp.complex64
    else:
        real_type, complex_type = cp.float64, cp.complex128

    # Preallocation.
    res_his = cp.empty((maxiter), dtype = real_type)
    s, hs = [cp.empty((x0.shape[0], 3*m), dtype = complex_type) for _ in range(2)]
    if isinstance(x0, np.ndarray):  # GPU, cupy array.
        s[:,:m] = cp.asarray(x0)
    else:
        s[:,:m] = x0.copy()

    # Memory.
    pool = cp.get_default_memory_pool()

    # Orthogonalization of RR procedure.
    RR = rayleigh_ritz_qr_sep if longortho else rayleigh_ritz_chol_sep
    
    hs[:,:m] = h_func(s[:,:m])
    ss = hermitize(gemm('H', 'N', s[:,:m], s[:,:m]))
    shs = hermitize(gemm('H', 'N', s[:,:m], hs[:,:m]))
    lambdas, _, _ = RR(ss[:,:m], shs[:,:m])

    # Initialization runtime.
    t_initial = owari() - t_h
    print(f"Time for LOBPCG initialization: {t_initial:<6.2f}s.")
    
    # Total runtime of iterations.
    t_tot_h = time.time()
    for iter_ in range(maxiter):
        t_iter_h = time.time()
        
        # Residual, convergence and locking.
        t_h = time.time()
        cp.multiply(s[:,:m], lambdas, out = s[:,m:2*m])
        s[:,m:2*m] -= hs[:,:m]

        res_nrms = norms(s[:,m:2*m])
        res_his[iter_] = cp.linalg.norm(res_nrms[:nev]) # Record residuals of desired parts.
        print(f"Iter = {iter_:<4d}, res_nrm = {cp.linalg.norm(res_nrms):<6.2e},", end =  " ")

        # Check nan.
        if cp.isnan(res_nrms).any():
            raise ValueError(f"{RED}Nan occurs in residuals.{RESET}")
        
        # Convergence.
        if max(res_nrms[:nev]) < tol:
            print(f"{GREEN}convergence reached.{RESET}")
            break

        ind_act = cp.where(res_nrms > tol)[0]
        n_act = len(ind_act)
        n_loc = 2*m if iter_ == 0 else 3*m
        
        t_mul = owari() - t_h
        
        # Preconditioning.
        t_h = time.time()
        
        s[:,m:m+n_act] = p_func(s[:,m:2*m])
        hs[:,m:m+n_act] = h_func(s[:,m:2*m])
        
        t_fft = owari() - t_h
        
        # Rayleigh-Ritz: done by cholesky factorization without extra orthogonalization.
        # Shape: lambdas (n_loc,), eigvec (n_loc,n_loc).
        #lambdas, eigvec, t_rr = rayleigh_ritz_chol(s[:,:n_loc], hs[:,:n_loc])
        lambdas, eigvec, t_rr = RR(s[:,:n_loc], hs[:,:n_loc])
        lambdas, eigvec = lambdas[:m], eigvec[:,:m]
            
        t_h = time.time()
        
        # The updated P is always stored in 2m:2m+n_act.
        _sep_update_after_rr(s, hs, eigvec, m, n_act, iter_)

        pool.free_all_blocks()
        peak_usage = pool.used_bytes() / (1024 ** 2)
        
        t_mul += owari() - t_h
        t_iter = owari() - t_iter_h
        
        # Print
        print(f"n_act = {n_act:<3d}. Runtime = {t_iter:<6.2f}s, ratio of FFT, RR, MM = ({t_fft/t_iter*100:<5.2f}%, "
              f"{t_mul/t_iter*100:<5.2f}%, {t_rr/t_iter*100:<5.2f}%). Memory = {peak_usage:<6.2f}MiB.")
    
    del hs

    t_tot_o = owari()
    t_tot = t_tot_o - t_tot_h
    print(f"\nA complete procedure of lobpcg is done, {t_tot:<6.2f}s elapsed.")

    info = np.array([iter_, t_tot])
    if history:
        info = np.append(info, res_his[1:iter_].get())
        
    return lambdas[:m], s[:,:m], info


def lobpcg_sep_max_nolock(h_func, x0, nev, tol=TOL, maxiter=MAXITER):

    r"""
    Usage:
        Compute several maximum eigenvalues by: x=\lambda Ax.
    """

    if isinstance(x0, np.ndarray):
        x = cp.asarray(x0)
    else:
        x = x0.copy()
    
    m = x.shape[1]
    hx = h_func(x)
    lambdas, _ = cp.linalg.eigh(hermitize(cp.cublas.gemm('H', 'N', x, hx)))
    
    # Local general eigenproblem.
    GEP = GEP_chol
    
    # Memory.
    pool = cp.get_default_memory_pool()
    
    # Preallocation.
    w, hw, p, hp = [cp.empty_like(x0, dtype = complex) for _ in range(4)]
    qq, qhq = [cp.empty((3*m,3*m), dtype = complex) for _ in range(2)]
    
    t_tot_h = time.time()
    
    for iter_ in range(maxiter):
        t_iter_h = time.time()
        
        # Residual, convergence and locking.
        t_h = time.time()
        cp.multiply(hx, lambdas, out = w)
        w -= x

        res_nrms = norms(w)

        # Check nan.
        if cp.isnan(res_nrms).any():
            raise ValueError(f"{RED}Nan occurs in residuals.{RESET}")

        n_act = cp.sum(res_nrms > tol)
        if max(res_nrms[:nev]) < tol:
            break
        
        t_o = owari()
        t_mul = t_o - t_h
        
        # Preconditioning.
        t_h = time.time()
        
        hw = h_func(w)

        t_o = owari()
        t_fft = t_o - t_h
        
        # Rayleigh-Ritz    
        t_h = time.time()
        
        hermitize(cp.cublas.gemm('H', 'N', x, hx), out = qhq[:m,:m])
        hermitize(cp.cublas.gemm('H', 'N', x, x),  out = qq[:m, :m])
        cp.cublas.gemm('H', 'N', x, hw, out = qhq[:m, m:2*m])
        cp.cublas.gemm('H', 'N', x, w,  out = qq[:m,  m:2*m])
        hermitize(cp.cublas.gemm('H', 'N', w, hw), out = qhq[m:2*m,m:2*m])
        hermitize(cp.cublas.gemm('H', 'N', w, w),  out = qq[m:2*m, m:2*m])
        cp.conj(qq[:m, m:2*m], out = qq[m:2*m, :m].T)
        cp.conj(qhq[:m, m:2*m], out = qhq[m:2*m, :m].T)
        t_o = owari()
        t_mul = t_mul + t_o - t_h
        
        if iter_ > 0:
            t_h = time.time()
            
            cp.cublas.gemm('H', 'N', x, hp, out = qhq[:m, 2*m:])
            cp.cublas.gemm('H', 'N', x, p,  out = qq[:m,  2*m:])
            cp.cublas.gemm('H', 'N', w, hp, out = qhq[m:2*m, 2*m:])
            cp.cublas.gemm('H', 'N', w, p,  out = qq[m:2*m,  2*m:])
            hermitize(cp.cublas.gemm('H', 'N', p, hp), out = qhq[2*m:, 2*m:])
            hermitize(cp.cublas.gemm('H', 'N', p, p),  out = qq[2*m:,  2*m:])
            cp.conj(qq[:m, 2*m:], out = qq[2*m:,:m].T)
            cp.conj(qhq[:m, 2*m:], out = qhq[2*m:,:m].T)
            cp.conj(qq[m:2*m, 2*m:], out = qq[2*m:,m:2*m].T)
            cp.conj(qhq[m:2*m, 2*m:], out = qhq[2*m:,m:2*m].T)
            
            t_o = owari()
            t_mul = t_mul + t_o - t_h
            
            lambdas, eigvec, t_eigh = GEP(qq, qhq, slice=m)
        else:
            lambdas, eigvec, t_eigh = GEP(qq[:2*m, :2*m], qhq[:2*m, :2*m], slice=m)
            
        t_h = time.time()
        
        if iter_ > 0:
            cp.cublas.gemm('N', 'N', p, eigvec[2*m:,:], out = p)
            cp.cublas.gemm('N', 'N', w, eigvec[m:2*m,:], beta = 1.0, out = p)
            cp.cublas.gemm('N', 'N', hp, eigvec[2*m:,:], out = hp)
            cp.cublas.gemm('N', 'N', hw, eigvec[m:2*m,:], beta = 1.0, out = hp)
        else:
            cp.cublas.gemm('N', 'N', w, eigvec[m:2*m,:], out = p)
            cp.cublas.gemm('N', 'N', hw, eigvec[m:2*m,:], out = hp)
        
        cp.cublas.gemm('N', 'N', x, eigvec[:m,:], out = x)
        x += p
        cp.cublas.gemm('N', 'N', hx, eigvec[:m,:], out = hx)
        hx += hp
        
        peak_usage = pool.used_bytes() / (1024 ** 2)
        
        t_o = owari()
        t_mul = t_mul + t_o - t_h
        
        t_iter_o = owari()
        t_iter = t_iter_o - t_iter_h
        
        # Print
        print(f"Iter = {iter_ + 1:<4d}, residual = {cp.linalg.norm(res_nrms):<6.3e}, "
              f"n_act = {n_act}, t_iter = {t_iter:<6.2f}s, t_fft = {t_fft:<6.2f}s ({t_fft/t_iter*100:<5.2f}%), "
              f"t_mul = {t_mul:<6.2f}s ({t_mul/t_iter*100:<5.2f}%), t_eigh = {t_eigh:<6.2f}s, memory = {peak_usage:<6.2f}MiB.")
    
    del w, p, hx, hw, hp
    
    t_tot_o = owari()
    t_tot = t_tot_o - t_tot_h
    print(f"\nA complete procedure of lobpcg is done, {t_tot:<6.2f}s elapsed.")
    
    return 1.0/lambdas[:nev], x[:,:nev], np.array([iter_,t_tot])

def lobpcg_sep_softlock(
        h_func_in, p_func, x0, nev, shift = 0.0,
        tol = TOL, maxiter = MAXITER, history = False, 
        longortho = False, singleprecision = False,
        maxstagniter = 50):

    """
    Usage:
        LOBPCG with softlocking, [X,W,P] stored as a whole in S.
    Input:
        h_func, p_func:  function handle of matrix/preconditioning.
        x0, nev:         initial guess and nev.
        tol, maxiter:    convergence parameters.
        history:         whether to return the residual history.
        longortho:       whether to use long orthogonalization in RR.
        singleprecision: use single precision in computation.
        (Warning1: when using single precision, all the handles should also support SP.)
        (Warning2: global single precision may lead to instability, stagnation, nan.)
        maxstagniter: maximum allowed iterations of stagnation, raise if residual remain
                      above 1000 after maxstagniter iterations.

    Output:
        lambdas, x: length m, but only nev entries reach convergence.
        info: iterations and runtime.
    """

    t_h = time.time()
    m = x0.shape[1]
    if singleprecision:
        real_type, complex_type = cp.float32, cp.complex64
    else:
        real_type, complex_type = cp.float64, cp.complex128

    if shift == 0.0:
        h_func = h_func_in
    else:
        h_func = lambda x: h_func_in(x)+shift*x

    # Preallocation.
    res_his = cp.empty((maxiter), dtype = real_type)
    s, hs = [cp.empty((x0.shape[0], 3*m), dtype = complex_type) for _ in range(2)]
    if isinstance(x0, np.ndarray):  # GPU, cupy array.
        s[:,:m] = cp.asarray(x0)
    else:
        s[:,:m] = x0.copy()

    # Memory.
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()

    # Orthogonalization of RR procedure.
    RR = rayleigh_ritz_qr_sep if longortho else rayleigh_ritz_chol_sep
    
    hs[:,:m] = h_func(s[:,:m])
    ss = hermitize(gemm('H', 'N', s[:,:m], s[:,:m]))
    shs = hermitize(gemm('H', 'N', s[:,:m], hs[:,:m]))
    lambdas, _, _ = RR(ss[:,:m], shs[:,:m])

    # Initialization runtime.
    t_initial = owari() - t_h
    print(f"Time for LOBPCG initialization: {t_initial:<6.2f}s.")
    
    # Total runtime of iterations.
    t_tot_h = time.time()
    blowup = False
    for iter_ in range(maxiter):
        t_iter_h = time.time()
        
        # Residual, convergence and locking.
        t_h = time.time()
        cp.multiply(s[:,:m], lambdas, out = s[:,m:2*m])
        s[:,m:2*m] -= hs[:,:m]

        res_nrms = norms(s[:,m:2*m])
        res_his[iter_] = cp.linalg.norm(res_nrms[:nev]) # Record residuals of desired parts.

        ind_act = cp.where(res_nrms > tol)[0]
        n_act = len(ind_act)
        print(f"Iter = {iter_:<4d}, res_nrm = {cp.linalg.norm(res_nrms):<6.2e}, n_act = {n_act:<3d}.", end =  " ")

        if cp.isnan(res_nrms).any():
            del x0, s, hs, ss, shs, res_his
            print(f"{RED}Nan occurs in residuals.{RESET}")
            return None, None, None
        if (iter_ > maxstagniter and (res_nrms[0] > 1000 or res_nrms[0] > res_his[1])) or \
            (iter_ > 2*maxstagniter and res_nrms[0] > 50):
            # Stagnation. (probably blowup, but nan does not occur)
            if cp.linalg.norm(res_nrms[:nev]) < res_his[maxstagniter // 2]*0.1:
                print(f"{YELLOW}Stagnation warning.{RESET}")
            else:
                print(f"{YELLOW}Stagnation detected, probably blowup but no nan occurs.{RESET}")
                return None, None, None
        
        # Convergence.
        if max(res_nrms[:nev]) < tol:
            print(f"{GREEN}convergence reached.{RESET}")
            break

        if iter_>0:
            n_loc = m + 2*n_act
        else:
            n_loc = m + n_act
        
        t_mul = owari() - t_h
        
        # Locking (Push W, P to front blocks).
        t_h = time.time()
        if n_act < m:
            for i0 in range(n_act): # in-place copy.
                cp.copyto(s[:, m+i0], s[:, m+ind_act[i0]])
            for i0 in range(n_act):
                cp.copyto(s[:, m+n_act+i0], s[:, 2*m+ind_act[i0]])
                cp.copyto(hs[:, m+n_act+i0], hs[:, 2*m+ind_act[i0]])
        t_lock = owari() - t_h
        
        # Preconditioning.
        t_h = time.time()
        
        s[:,m:m+n_act] = p_func(s[:,m:m+n_act])
        hs[:,m:m+n_act] = h_func(s[:,m:m+n_act])
        
        t_fft = owari() - t_h
        
        # Rayleigh-Ritz: done by cholesky factorization without extra orthogonalization.
        # Shape: lambdas (n_loc,), eigvec (n_loc,n_loc).
        #lambdas, eigvec, t_rr = rayleigh_ritz_chol(s[:,:n_loc], hs[:,:n_loc])
        try:
            lambdas, eigvec, t_rr = RR(s[:,:n_loc], hs[:,:n_loc])
        except:
            del x0, s, hs, ss, shs, res_his
            gc.collect()
            pool.free_all_blocks()
            return None, None, None

        if cp.isnan(lambdas).any() or cp.isnan(eigvec).any():
            del x0, s, hs, ss, shs, res_his
            gc.collect()
            pool.free_all_blocks()
            print(f"{RED}Nan occurs after Rayleigh-Ritz procedure.{RESET}")
            return None, None, None
        lambdas, eigvec = lambdas[:m], eigvec[:,:m]
            
        t_h = time.time()
        
        # The updated P is always stored in 2m:2m+n_act.
        _sep_update_after_rr(s, hs, eigvec, m, n_act, iter_)

        pool.free_all_blocks()
        peak_usage = pool.used_bytes() / (1024 ** 2)
        
        t_mul += owari() - t_h
        t_iter = owari() - t_iter_h
        
        # Print
        print(f"Runtime = {t_iter:<6.2f}s, ratio of FFT, RR, MM, LOCK = ({t_fft/t_iter*100:<5.2f}%, "
              f"{t_mul/t_iter*100:<5.2f}%, {t_rr/t_iter*100:<5.2f}%, {t_lock/t_iter*100:<5.2f}%). "
              f"Memory = {peak_usage:<6.2f}MiB.")
    
    del hs
    
    t_tot_o = owari()
    t_tot = t_tot_o - t_tot_h
    print(f"\nA complete procedure of lobpcg is done, {t_tot:<6.2f}s elapsed.")

    info = np.array([iter_, t_tot])
    if history:
        info = np.append(info, res_his[1:iter_].get())
        
    return lambdas[:m]-shift, s[:,:m], info

def lobpcg_sep_softlock_mixedprecision(
        h_func, p_func, x0, nev, 
        tol = TOL, maxiter = MAXITER, history = False, 
        longortho = False):

    """
    Usage:
        LOBPCG with softlocking and mixed precision (precondition single precison).
    Input:
        h_func, p_func:  function handle of matrix/preconditioning.
        x0, nev:      initial guess and nev.
        tol, maxiter:    convergence parameters.
        history:         whether to return the residual history.
        longortho:       whether to use long orthogonalization in RR.
        (Warning: when using mixed precision, H_func should support SP.)
        
    """

    t_h = time.time()
    m = x0.shape[1]

    # Preallocation.
    res_his = cp.empty((maxiter), dtype = float)
    s, hs = [cp.empty((x0.shape[0], 3*m), dtype = complex) for _ in range(2)]
    if isinstance(x0, np.ndarray):  # GPU, cupy array.
        s[:,:m] = cp.asarray(x0)
    else:
        s[:,:m] = x0.copy()
    
    hs[:,:m] = h_func(s[:,:m])
    lambdas, _ = cp.linalg.eigh(hermitize(cp.cublas.gemm('H', 'N', s[:,:m], hs[:,:m])))
    
    # Memory.
    pool = cp.get_default_memory_pool()

    # Orthogonalization of RR procedure.
    RR = rayleigh_ritz_qr_sep if longortho else rayleigh_ritz_chol_sep
    t_initial = owari() - t_h
    print(f"Time for LOBPCG initialization: {t_initial:<6.2f}s.")
    
    t_tot_h = time.time()
    for iter_ in range(maxiter):
        t_iter_h = time.time()
        
        # Residual, convergence and locking.
        t_h = time.time()
        cp.multiply(s[:,:m], lambdas, out = s[:,m:2*m])
        s[:,m:2*m] -= hs[:,:m]

        res_nrms = norms(s[:,m:2*m])
        res_his[iter_] = cp.linalg.norm(res_nrms[:nev])

        # Check nan.
        if cp.isnan(res_nrms).any():
            raise ValueError(f"{RED}Nan occurs in residuals.{RESET}")
        
        # Convergence.
        ind_act = cp.where(res_nrms > tol)[0]
        n_act = len(ind_act)
        if iter_>0:
            n_loc = m + 2*n_act
        else:
            n_loc = m + n_act
        if max(res_nrms[:nev]) < tol:
            break
        t_mul = owari() - t_h
        
        # Locking (Push W, P to front blocks).
        t_h = time.time()
        if n_act < m:
            for i0 in range(n_act): # in-place copy.
                cp.copyto(s[:, m+i0], s[:, m+ind_act[i0]])
            for i0 in range(n_act):
                cp.copyto(s[:, m+n_act+i0], s[:, 2*m+ind_act[i0]])
                cp.copyto(hs[:, m+n_act+i0], hs[:, 2*m+ind_act[i0]])
        t_lock = owari() - t_h
        
        # Preconditioning (single precision).
        t_h = time.time()
        
        if n_act < m:
            s[:,m:m+n_act] = p_func(s[:,m:m+n_act].astype(cp.complex64)).astype(cp.complex128)
        else:
            s[:,m:2*m] = p_func(s[:,m:2*m].astype(cp.complex64)).astype(cp.complex128)
        hs[:,m:m+n_act] = h_func(s[:,m:m+n_act])
        
        t_fft = owari() - t_h
        
        # Rayleigh-Ritz: done by cholesky factorization without extra orthogonalization.
        # Shape: lambdas (n_loc,), eigvec (n_loc,n_loc).
        #lambdas, eigvec, t_rr = rayleigh_ritz_chol(s[:,:n_loc], hs[:,:n_loc])
        lambdas, eigvec, t_rr = RR(s[:,:n_loc], hs[:,:n_loc])
        lambdas, eigvec = lambdas[:m], eigvec[:,:m]
            
        t_h = time.time()
        
        # The updated P is always stored in 2m:2m+n_act.
        if iter_ > 0:   # p,hp.shape[1] == m.
            s[:,2*m:] = cp.cublas.gemm('N', 'N', s[:,m+n_act:n_loc], eigvec[m+n_act:,:])
            cp.cublas.gemm('N', 'N', s[:,m:m+n_act], eigvec[m:m+n_act,:], beta = 1.0, out = s[:,2*m:])
            hs[:,2*m:] = cp.cublas.gemm('N', 'N', hs[:,m+n_act:n_loc], eigvec[m+n_act:,:])
            cp.cublas.gemm('N', 'N', hs[:,m:m+n_act], eigvec[m:m+n_act,:], beta = 1.0, out = hs[:,2*m:])
        else:
            cp.cublas.gemm('N', 'N', s[:,m:m+n_act], eigvec[m:,:], out = s[:,2*m:])
            cp.cublas.gemm('N', 'N', hs[:,m:m+n_act], eigvec[m:,:], out = hs[:,2*m:])
        
        cp.cublas.gemm('N', 'N', s[:,:m], eigvec[:m,:], out = s[:,:m])
        s[:,:m] += s[:,2*m:]
        cp.cublas.gemm('N', 'N', hs[:,:m], eigvec[:m,:], out = hs[:,:m])
        hs[:,:m] += hs[:,2*m:]
        
        pool.free_all_blocks()
        peak_usage = pool.used_bytes() / (1024 ** 2)
        
        t_mul += owari() - t_h
        t_iter = owari() - t_iter_h
        
        # Print
        print(f"Iter = {iter_ + 1:<4d}, res_nrm = {cp.linalg.norm(res_nrms):<6.2e}, "
              f"n_act = {n_act:<3d}. Runtime = {t_iter:<6.2f}s, ratio of FFT, RR, MM, LOCK = ({t_fft/t_iter*100:<5.2f}%, "
              f"{t_mul/t_iter*100:<5.2f}%, {t_rr/t_iter*100:<5.2f}%, {t_lock/t_iter*100:<5.2f}%). "
              f"Memory = {peak_usage:<6.2f}MiB.")
    
    del hs
    #pool.free_all_blocks()
    #cp.cuda.set_allocator()
    
    t_tot_o = owari()
    t_tot = t_tot_o - t_tot_h
    print(f"\nA complete procedure of lobpcg is done, {t_tot:<6.2f}s elapsed.")

    info = np.array([iter_, t_tot])
    if history:
        info = np.append(info, res_his[1:iter_].get())
        
    return lambdas[:nev], s[:,:nev], info

"""
    LOBPCG from cupyx.scipy.sparse.linalg.
"""
def lobpcg_sep_cpxlinalg(h_func, p_func, x0, nev, tol = TOL, maxiter = MAXITER, history = False):

    """
    Usage:
        LOBPCG from cupyx.scipy.sparse.linalg.lobpcg.
    Input:
        h_func, p_func:  function handle of matrix/preconditioning.
        x0, nev:      initial guess and nev.
        tol, maxiter:    convergence parameters.
        history:         whether to return the residual history.
    Output:
        lambdas: eigenvalues.
        x:       eigenvectors.
        iter_:   iterations and runtime.
    """

    from cupyx.scipy.sparse import linalg as cpx_linalg

    t_h = time.time()
    if isinstance(x0, np.ndarray):
        x0_cpx = cp.asarray(x0)
    else:
        x0_cpx = x0.copy()
    
    size = x0_cpx.shape[0]
    h_op = cpx_linalg.LinearOperator((size, size), matvec = h_func, dtype = x0_cpx.dtype)
    p_op = cpx_linalg.LinearOperator((size, size), matvec = p_func, dtype = x0_cpx.dtype)

    t_init = owari() - t_h
    print(f"Time for LOBPCG initialization: {t_init:<6.2f}s.")

    t_tot_h = time.time()
    lambdas, x, info = cpx_linalg.lobpcg(
        A = h_op,
        X = x0_cpx[:, :nev],
        M = p_op,
        largest = False,
        tol = tol,
        maxiter = maxiter,
        retLambdaHistory = history
    )
    t_tot_o = owari()
    t_tot = t_tot_o - t_tot_h
    print(f"\nA complete procedure of lobpcg is done, {t_tot:<6.2f}s elapsed.")
    info_arr = np.array([info['numiter'], t_tot])
    if history:
        info_arr = np.append(info_arr, cp.asarray(info['residualNorms']).get())
    
    return lambdas, x, info_arr


"""
    LOBPCG GEP.
"""
def lobpcg_gep_softlock(
        h_func, m_func, p_func, x0, nev, 
        tol = TOL, maxiter = MAXITER, history = False, 
        longortho = False, singleprecision = False):

    """
    Usage:
        LOBPCG GEP with softlocking, [X,W,P] stored as a whole in S.
    Input:
        h_func, m_func, p_func:  function handle of stiffness, mass matrix/preconditioning.
        x0, nev:      initial guess and nev.
        tol, maxiter:    convergence parameters.
        history:         whether to return the residual history.
        longortho:       whether to use long orthogonalization in RR.
        singleprecision: use single precision in computation.
        (Warning1: when using single precision, all the handles should also support SP.)
        (Warning2: global single precision may lead to instability, stagnation, nan.)
    """

    t_h = time.time()
    m = x0.shape[1]
    if singleprecision:
        real_type, complex_type = cp.float32, cp.complex64
    else:
        real_type, complex_type = cp.float64, cp.complex128

    # Preallocation.
    res_his = cp.empty((maxiter), dtype = real_type)
    s, ms, hs = [cp.empty((x0.shape[0], 3*m), dtype = complex_type) for _ in range(3)]
    if isinstance(x0, np.ndarray):  # GPU, cupy array.
        s[:,:m] = cp.asarray(x0)
    else:
        s[:,:m] = x0.copy()

    # Memory.
    pool = cp.get_default_memory_pool()

    # Orthogonalization of RR procedure.
    RR = rayleigh_ritz_qr_gep if longortho else rayleigh_ritz_chol_gep
    
    hs[:,:m] = h_func(s[:,:m])
    ms[:,:m] = m_func(s[:,:m])
    lambdas, _, _ = RR(s[:,:m], ms[:,:m], hs[:,:m])
    
    # Initialization runtime.
    t_initial = owari() - t_h
    print(f"Time for LOBPCG initialization: {t_initial:<6.2f}s.")
    
    # Total runtime of iterations.
    t_tot_h = time.time()
    for iter_ in range(maxiter):
        t_iter_h = time.time()
        
        # Residual, convergence and locking.
        t_h = time.time()
        cp.multiply(ms[:,:m], lambdas, out = s[:,m:2*m])
        s[:,m:2*m] -= hs[:,:m]

        res_nrms = norms(s[:,m:2*m])
        res_his[iter_] = cp.linalg.norm(res_nrms[:nev]) # Record residuals of desired parts.
        print(f"Iter = {iter_:<4d}, res_nrm = {cp.linalg.norm(res_nrms):<6.2e},", end =  " ")

        # Check nan.
        if cp.isnan(res_nrms).any():
            raise ValueError(f"{RED}Nan occurs in residuals.{RESET}")
        
        # Convergence.
        if max(res_nrms[:nev]) < tol:
            print(f"{GREEN}convergence reached.{RESET}")
            break
        
        ind_act = cp.where(res_nrms > tol)[0]
        n_act = len(ind_act)
        if iter_>0:
            n_loc = m + 2*n_act
        else:
            n_loc = m + n_act

        t_mul = owari() - t_h
        
        # Locking (Push W, P to front blocks).
        t_h = time.time()
        if n_act < m:
            for i0 in range(n_act): # in-place copy.
                cp.copyto(s[:, m+i0], s[:, m+ind_act[i0]])
                cp.copyto(s[:, m+n_act+i0], s[:, 2*m+ind_act[i0]])
                cp.copyto(ms[:, m+n_act+i0], ms[:, 2*m+ind_act[i0]])
                cp.copyto(hs[:, m+n_act+i0], hs[:, 2*m+ind_act[i0]])
        t_lock = owari() - t_h
        
        # Preconditioning.
        t_h = time.time()
        
        s[:,m:m+n_act] = p_func(s[:,m:m+n_act])
        ms[:,m:m+n_act] = m_func(s[:,m:m+n_act])
        hs[:,m:m+n_act] = h_func(s[:,m:m+n_act])
        
        t_fft = owari() - t_h
        
        # Rayleigh-Ritz: done by cholesky factorization without extra orthogonalization.
        # Shape: lambdas (n_loc,), eigvec (n_loc,n_loc).
        #lambdas, eigvec, t_rr = rayleigh_ritz_chol(s[:,:n_loc], hs[:,:n_loc])
        lambdas, eigvec, t_rr = RR(s[:,:n_loc], ms[:,:n_loc], hs[:,:n_loc])
        lambdas, eigvec = lambdas[:m], eigvec[:,:m]
        
        t_h = time.time()
        
        # The updated P is always stored in 2m:2m+n_act.
        if iter_ > 0:   # p,hp.shape[1] == m.
            s[:,2*m:] = cp.cublas.gemm('N', 'N', s[:,m+n_act:n_loc], eigvec[m+n_act:,:])
            cp.cublas.gemm('N', 'N', s[:,m:m+n_act], eigvec[m:m+n_act,:], beta = 1.0, out = s[:,2*m:])
            ms[:,2*m:] = cp.cublas.gemm('N', 'N', ms[:,m+n_act:n_loc], eigvec[m+n_act:,:])
            cp.cublas.gemm('N', 'N', ms[:,m:m+n_act], eigvec[m:m+n_act,:], beta = 1.0, out = ms[:,2*m:])
            hs[:,2*m:] = cp.cublas.gemm('N', 'N', hs[:,m+n_act:n_loc], eigvec[m+n_act:,:])
            cp.cublas.gemm('N', 'N', hs[:,m:m+n_act], eigvec[m:m+n_act,:], beta = 1.0, out = hs[:,2*m:])
        else:
            cp.cublas.gemm('N', 'N', s[:,m:m+n_act], eigvec[m:,:], out = s[:,2*m:])
            cp.cublas.gemm('N', 'N', ms[:,m:m+n_act], eigvec[m:,:], out = ms[:,2*m:])
            cp.cublas.gemm('N', 'N', hs[:,m:m+n_act], eigvec[m:,:], out = hs[:,2*m:])
        
        cp.cublas.gemm('N', 'N', s[:,:m], eigvec[:m,:], out = s[:,:m])
        s[:,:m] += s[:,2*m:]
        cp.cublas.gemm('N', 'N', ms[:,:m], eigvec[:m,:], out = ms[:,:m])
        ms[:,:m] += ms[:,2*m:]
        cp.cublas.gemm('N', 'N', hs[:,:m], eigvec[:m,:], out = hs[:,:m])
        hs[:,:m] += hs[:,2*m:]
        
        pool.free_all_blocks()
        peak_usage = pool.used_bytes() / (1024 ** 2)
        
        t_mul += owari() - t_h
        t_iter = owari() - t_iter_h
        
        # Print
        print(f"n_act = {n_act:<3d}. Runtime = {t_iter:<6.2f}s, ratio of FFT, RR, MM, LOCK = ({t_fft/t_iter*100:<5.2f}%, "
              f"{t_mul/t_iter*100:<5.2f}%, {t_rr/t_iter*100:<5.2f}%, {t_lock/t_iter*100:<5.2f}%). "
              f"Memory = {peak_usage:<6.2f}MiB.")
    
    del ms, hs
    #pool.free_all_blocks()
    #cp.cuda.set_allocator()
    
    t_tot_o = owari()
    t_tot = t_tot_o - t_tot_h
    print(f"\nA complete procedure of lobpcg is done, {t_tot:<6.2f}s elapsed.")

    info = np.array([iter_, t_tot])
    if history:
        info = np.append(info, res_his[1:iter_].get())
        
    return lambdas[:nev], s[:,:nev], info


"""

    Block steepest descent.

"""

def descent_sep(h_func, p_func, x0 ,nev, tol = TOL, maxiter = MAXITER, history = False):
    
    r"""
    Usage:
        Two-term recurrence CG-type iteration (LOBPCG: three-term)
    Model: 
        Ax = \lambda x.
    
    Input:
        h_func: function handle of matrix A.
        p_func: function handle of preconditioning P.
        x0:     initial guess.
        nev:    number of desired eigenpairs.
    
    **kwargs: 
        tolerance and maximum iterations.
    
    Output:
        lambdas: eigenvalues.
        x:       eigenvectors.
        iter_:   iterations and time.time.
        
    """ 
    
    """
    initialization.
    """
    
    t_h = time.time()
    
    # Local eigensolver.
    GEP = GEP_chol
    
    nn, m = cp.shape(x0)
    nn = round(nn / 3)
    
    # Initialization.     
    x = x0.copy()
    hx = h_func(x)    
    lambdas, _ = cp.linalg.eigh(hermitize(x.T.conj() @ hx))

    # Memory.
    pool = cp.get_default_memory_pool()
    res_his = cp.empty((maxiter), dtype = float)
    
    t_o = owari()
    print(f"Time for initialization: {t_o - t_h:<6.2f}s.")
    
    """
    Main loop.
    """
    
    t_tot_h = time.time()
    
    for iter_ in range(maxiter):
        t_iter_h = time.time()
        
        # Residual, convergence and locking.
        t_h = time.time()
        w = hx - x * lambdas
        
        res_nrms = norms(w)
        res_his[iter_] = cp.linalg.norm(res_nrms[:nev])

        # Check nan.
        if cp.isnan(res_nrms).any():
            raise ValueError(f"{RED}Nan occurs in residuals.{RESET}")

        ind_act = cp.where(res_nrms > tol)[0]
        n_act = len(ind_act)

        if n_act <= m - nev:
            break
        
        t_o = owari()
        t_mul = t_o - t_h
        
        # Preconditioning.
        t_h = time.time()
        w = p_func(w[:,ind_act])
        hw = h_func(w)

        t_o=owari()
        t_fft = t_o - t_h
        
        # Rayleigh-Ritz    
        t_h = time.time()
        t11, g11 = hermitize(x.T.conj() @ hx), hermitize(x.T.conj() @ x)
        t12, g12 = x.T.conj() @ hw, x.T.conj() @ w
        t22, g22 = hermitize(w.T.conj() @ hw), hermitize(w.T.conj() @ w)
        
        t_o = owari()
        t_mul = t_mul + t_o - t_h
        
        lambdas, eigvec, t_eigh = GEP(cp.concatenate((cp.concatenate((t11,t12),axis=1),cp.concatenate((t12.T.conj(),t22),axis=1)),axis=0),\
                                      cp.concatenate((cp.concatenate((g11,g12),axis=1),cp.concatenate((g12.T.conj(),g22),axis=1)),axis=0))
        
        t_h = time.time()
        
        lambdas = lambdas[0:m]
        eigvec  = eigvec[:,0:m]
        x[:] = x @ eigvec[0:m,:] + w @ eigvec[m:m+n_act,:]
        hx[:] = hx @ eigvec[0:m,:] + hw @ eigvec[m:m+n_act,:]
        
        pool.free_all_blocks()
        peak_usage = pool.used_bytes() / (1024 ** 2)

        t_o = owari()
        t_mul = t_mul + t_o-t_h
        
        t_iter_o = owari()
        t_iter = t_iter_o - t_iter_h
        
        # Print        
        print(f"Iter = {iter_:<4d}, residual = {cp.linalg.norm(res_nrms):<6.3e}, "
              f"n_act = {n_act}, t_iter = {t_iter:<6.2f}s, t_fft = {t_fft:<6.2f}s  ({t_fft/t_iter*100:<5.2f}%), "
              f"t_mul = {t_mul:<6.2f}s  ({t_mul/t_iter*100:<5.2f}%), t_eigh = {t_eigh:<6.2f}s, "
              f"memory = {peak_usage:<6.2f}MiB.")
    
    del w,hx,hw
    
    t_tot_o = owari()
    t_tot = t_tot_o - t_tot_h
    print(f"\nA complete procedure of lobpcg is done, {t_tot:<6.2f}s elapsed.")
    
    info = np.array([iter_, t_tot])
    if history:
        info = np.append(info, res_his[1:iter_].get())

def descent_gep(h_func, b_func, p_func, x0, nev, tol = TOL, maxiter = MAXITER, history = False):

    r"""
    Usage:
        Steepest descent for generalized eigenvalue problems (GEP).
    Model: 
        Ax=\lambda Bx.
    
    Input:
        h_func:   function handle of matrix A.
        b_func:   function handle of matrix B (SPD/HPD).
        p_func:   function handle of preconditioning P.
        x0:       initial guess.
        nev:   number of desired eigenpairs.
    
    **kwargs: 
        tolerance and maximum iterations.
   
    Output:
        lambdas: eigenvalues.
        x:       eigenvectors.
        iter_:   iterations and time.time.
    """ 
    
    """
    LOBPCG: initialization.
    """
    
    t_h = time.time()
    
    # Local eigensolver.
    GEP = GEP_cpu
    
    nn, m = cp.shape(x0)
    nn = round(nn/3)
    
    # Initialization.    
    x = x0.copy()
    hx = h_func(x)
    mx = b_func(x)   
    lambdas, _ , _ = GEP_chol(hermitize(x.T.conj() @ hx), hermitize(x.T.conj() @ mx))
    res_his = cp.empty((maxiter), dtype = float)
    
    t_o = owari()
    print(f"Time for initialization: {t_o - t_h:<6.2f}s.")
    
    """
    LOBPCG: Main loop.
    """
    
    t_tot_h = time.time()
    for iter_ in range(maxiter):
        t_iter_h = time.time()
        
        # Residual, convergence and locking.
        t_h = time.time()
        w = hx - mx * lambdas
        
        res_nrms = norms(w)
        res_his[iter_] = cp.linalg.norm(res_nrms[:nev])

        # Check nan.
        if cp.isnan(res_nrms).any():
            raise ValueError(f"{RED}Nan occurs in residuals.{RESET}")

        ind_act = cp.where(res_nrms > tol)[0]
        n_act = len(ind_act)

        if n_act <= m - nev:
            break
        
        t_o = owari()
        t_mul = t_o - t_h
        
        # Preconditioning.
        t_h = time.time()
        w = p_func(w[:,ind_act])
        hw = h_func(w)
        mw = b_func(w)

        t_o = owari()
        t_fft = t_o - t_h
        
        # Rayleigh-Ritz    
        
        t_h = time.time()
        t11,g11 = hermitize(x.T.conj() @ hx), hermitize(x.T.conj() @ mx)
        t12,g12 = x.T.conj() @ hw, x.T.conj() @ mw
        t22,g22 = hermitize(w.T.conj() @ hw), hermitize(w.T.conj() @ mw)

        t_o = owari()
        t_mul = t_mul + t_o - t_h
        
        lambdas, eigvec, t_eigh = GEP(cp.concatenate((cp.concatenate((t11,t12),axis=1),cp.concatenate((t12.T.conj(),t22),axis=1)),axis=0),\
                                      cp.concatenate((cp.concatenate((g11,g12),axis=1),cp.concatenate((g12.T.conj(),g22),axis=1)),axis=0))
        
        t_h = time.time()
        
        lambdas = lambdas[0:m]
        eigvec = eigvec[:,0:m]
        
        x = x @ eigvec[0:m,:] + w @ eigvec[m:m + n_act,:]
        hx = hx @ eigvec[0:m,:] + hw @ eigvec[m:m + n_act,:]
        mx = mx @ eigvec[0:m,:] + mw @ eigvec[m:m + n_act,:]

        t_o = owari()
        t_mul = t_mul + t_o - t_h
        
        t_iter_o = owari()
        t_iter = t_iter_o - t_iter_h
        
        # Print
        print(f"Iter = {iter_:<4d}, residual = {cp.linalg.norm(res_nrms):<6.3e}, "
              f"n_act = {n_act}, t_iter = {t_iter:<6.2f}s, t_fft = {t_fft:<6.2f}s ({t_fft/t_iter*100:<5.2f}%), "
              f"t_mul = {t_mul:<6.2f}s ({t_mul/t_iter*100:<5.2f}%), t_eigh = {t_eigh:<6.2f}s.")
    
    del w,hx,hw
    
    t_tot_o = owari()
    t_tot = t_tot_o - t_tot_h
    print(f"\nA complete procedure of lobpcg is done, {t_tot:<6.2f}s elapsed.")
    
    info = np.array([iter_, t_tot])
    if history:
        info = np.append(info, res_his[1:iter_].get())

def lobpcg4svd_sep(h_func, h_herm_func, p_func, x0, nev, tol = TOL, maxiter = MAXITER, history = False):
    """
    Usage:
        LOBPCG for computing several smallest singular values/vectors.
    Input:
        h_func, a_herm_func:   function handle of matrix A and A.T.conj().
        p_func:                function handle of preconditioning P.
        x0:                    initial guess.
        nev:                number of desired singular values/vectors.
    **kwargs: 
        tolerance and maximum iterations.
    Output:
        sigmas:  singular values.
        x:       right singular vectors.
        info:    iterations and time.time.
    """

    """
    LOBPCG with softlocking, [X,W,P] stored as a whole in S.
    """

    t_h = time.time()
    m = x0.shape[1]

    # Preallocation.
    res_his = cp.empty((maxiter), dtype = float)
    s, hs = [cp.empty((x0.shape[0], 3*m), dtype = complex) for _ in range(2)]
    hhx = cp.empty_like(x0)
    if isinstance(x0, np.ndarray):  # GPU, cupy array.
        s[:,:m] = cp.asarray(x0)
    else:
        s[:,:m] = x0.copy()
    
    hs[:,:m] = h_func(s[:,:m])
    hs[:,:m], r = cp.linalg.qr(hs[:,:m])
    
    cp.multiply(s[:, :m], v, out = s[:, :m]) 
    
    # Memory.
    pool = cp.get_default_memory_pool()

    # Residual.
    hhx = h_herm_func(hs[:,:m])
    cp.multiply(s[:,:m], lambdas*lambdas, out = s[:,m:2*m])
    s[:,m:2*m] -= hhx
    n_act = m
    
    t_initial = owari() - t_h
    print(f"Time for LOBPCG initialization: {t_initial:<6.2f}s.")
    
    t_tot_h = time.time()
    for iter_ in range(maxiter):
        t_iter_h = time.time()
        
        # Preconditioning.
        t_h = time.time()
        
        if n_act < m:
            s[:,m:m+n_act] = p_func(s[:,m:m+n_act])
        else:
            s[:,m:2*m] = p_func(s[:,m:2*m])
        hs[:,m:m+n_act] = h_func(s[:,m:m+n_act])
        
        t_fft = owari() - t_h
        
        # Rayleigh-Ritz SVD.
        # Shape: lambdas (n_loc,), eigvec (n_loc,n_loc).
        t_h = time.time()
        s[:,:n_loc], _ = cp.linalg.qr(s[:,:n_loc])
        hs[:,:n_loc], r = cp.linalg.qr(hs[:,:n_loc])
        _ , lambdas, v = cp.linalg.svd(r)
        lambdas, v = lambdas[:m], v[:,:m]
        t_rr = owari() - t_h
            
        t_h = time.time()
        
        # Update X,P.
        gemm('N','N', s, v, out = s[:,:m])
        gemm('N','N', hs, v, out = hs[:,:m])
        gemm('N','N', s[:,m:], v[m:,:], out = s[:,m+n_act:])
        gemm('N','N', hs[:,m:], v[m:,:], out = hs[:,m+n_act:])
        t_mul = owari() - t_h

        t_h = time.time()
        hhx = h_herm_func(hs[:,:m])
        t_fft += owari() - t_h
        cp.multiply(s[:,:m], lambdas*lambdas, out = s[:,m:2*m])
        s[:,m:2*m] -= hhx

        # Residual, convergence and locking.
        t_h = time.time()
        res_nrms = norms(s[:,m:2*m])
        res_his[iter_] = cp.linalg.norm(res_nrms[:nev])

        # Check nan.
        if cp.isnan(res_nrms).any():
            raise ValueError(f"{RED}Nan occurs in residuals.{RESET}")
        
        # Convergence.
        ind_act = cp.where(res_nrms > tol)[0]
        n_act = len(ind_act)
        if iter_>0:
            n_loc = m + 2*n_act
        else:
            n_loc = m + n_act
        if max(res_nrms[:nev]) < tol:
            break
        t_mul = owari() - t_h
        
        # Locking (Push W, P to front blocks).
        t_h = time.time()
        if n_act < m:
            for i0 in range(n_act): # in-place copy.
                cp.copyto(s[:, m+i0], s[:, m+ind_act[i0]])
                cp.copyto(s[:, m+n_act+i0], s[:, 2*m+ind_act[i0]])
                cp.copyto(hs[:, m+n_act+i0], hs[:, 2*m+ind_act[i0]])
        t_lock = owari() - t_h

        pool.free_all_blocks()
        peak_usage = pool.used_bytes() / (1024 ** 2)
        t_iter = owari() - t_iter_h
        
        # Print
        print(f"Iter = {iter_ + 1:<4d}, res_nrm = {cp.linalg.norm(res_nrms):<6.2e}, "
              f"n_act = {n_act:<3d}. Runtime = {t_iter:<6.2f}s, ratio of FFT, RR, MM, LOCK = ({t_fft/t_iter*100:<5.2f}%, "
              f"{t_mul/t_iter*100:<5.2f}%, {t_rr/t_iter*100:<5.2f}%, {t_lock/t_iter*100:<5.2f}%). "
              f"Memory = {peak_usage:<6.2f}MiB.")
    
    del hs
    #pool.free_all_blocks()
    #cp.cuda.set_allocator()
    
    t_tot_o = owari()
    t_tot = t_tot_o - t_tot_h
    print(f"\nA complete procedure of lobpcg is done, {t_tot:<6.2f}s elapsed.")

    info = np.array([iter_, t_tot])
    if history:
        info = np.append(info, res_his[1:iter_].get())
        
    return lambdas[:nev], s[:,:nev], info

"""
Interior.
"""

def _sep_update_after_rr(s, hs, eigvec, m, n_act, iter_):

    #rows, cols = s.shape
    #grid_size = rows * m
    n_loc = m+2*n_act
    if iter_ > 0:
        s[:,2*m:] = cp.cublas.gemm('N', 'N', s[:,m+n_act:n_loc], eigvec[m+n_act:,:])
        cp.cublas.gemm('N', 'N', s[:,m:m+n_act], eigvec[m:m+n_act,:], beta = 1.0, out = s[:,2*m:])
        hs[:,2*m:] = cp.cublas.gemm('N', 'N', hs[:,m+n_act:n_loc], eigvec[m+n_act:,:])
        cp.cublas.gemm('N', 'N', hs[:,m:m+n_act], eigvec[m:m+n_act,:], beta = 1.0, out = hs[:,2*m:])
        #lobpcg_sep_update_kernel(eigvec, m, n_act, cols, s, hs, size=grid_size)
    else:
        
        cp.cublas.gemm('N', 'N', s[:,m:m+n_act], eigvec[m:,:], out = s[:,2*m:])
        cp.cublas.gemm('N', 'N', hs[:,m:m+n_act], eigvec[m:,:], out = hs[:,2*m:])
        #lobpcg_sep_update_iter0_kernel(eigvec, m, n_act, cols, s, hs, size=grid_size)
    
    cp.cublas.gemm('N', 'N', s[:,:m], eigvec[:m,:], out = s[:,:m])
    s[:,:m] += s[:,2*m:]
    cp.cublas.gemm('N', 'N', hs[:,:m], eigvec[:m,:], out = hs[:,:m])
    hs[:,:m] += hs[:,2*m:]
    
    return

"""
Tests
"""

def test():

    return 


def main():

    test()

if __name__ == '__main__':
    main()