# -*- coding: utf-8 -*-
#
# Created on 2025-03-10 (Monday) at 16:40:40
#
# Author: Epsilon-79th
#
# Usage: LOBPCG and its variants.
#

import cupy as cp
from cupy.cublas import gemm
import numpy as np
import scipy as sc
from time import time

from eigensolver import *   # Local eigensolvers.
from gpu_opts import *
owari = owari_opt(str(cp))

TOL = 1e-5
ITER_MAX = 500

r"""
Environment:
    GPU, cupy, cupyx.

Input:
    Function handle of A,B (Ax=\lamba Bx, for sep B=I), preconditioner P.
    Parameters of convergence: nev, tol, iter_max, etc.

Output:
    Eigenvalues, eigenvectors and iterations (times, time).

"""

def lobpcg_sep_nolock(a_func, p_func, x0, m_conv, tol=TOL, iter_max=ITER_MAX):
    
    if isinstance(x0, np.ndarray):
        x = cp.asarray(x0)
    else:
        x = x0.copy()
    
    m = x.shape[1]
    hx = a_func(x)
    lambdas, _ = cp.linalg.eigh(hermitize(gemm('H', 'N', x, hx)))
    
    # Local general eigenproblem.
    GEP = GEP_chol
    
    # Memory.
    pool = cp.get_default_memory_pool()
    
    # Preallocation.
    w, hw, p, hp = [cp.empty_like(x0, dtype = complex) for _ in range(4)]
    qq, qhq = [cp.empty((3*m,3*m), dtype = complex) for _ in range(2)]
    
    t_tot_h = time()
    
    for iter_ in range(iter_max):
        t_iter_h = time()
        
        # Residual, convergence and locking.
        t_h = time()
        cp.multiply(x, lambdas, out = w)
        w -= hx

        res_nrms = norms(w)
        n_act = cp.sum(res_nrms > tol)
        if max(res_nrms[:m_conv]) < tol:
            break
        
        t_o = owari()
        t_mul = t_o - t_h
        
        # Preconditioning.
        t_h = time()
        
        w = p_func(w)
        hw = a_func(w)

        t_o = owari()
        t_fft = t_o - t_h
        
        # Rayleigh-Ritz    
        t_h = time()
        
        hermitize(gemm('H', 'N', x, hx), out = qhq[:m,:m])
        hermitize(gemm('H', 'N', x, x),  out = qq[:m, :m])
        gemm('H', 'N', x, hw, out = qhq[:m, m:2*m])
        gemm('H', 'N', x, w,  out = qq[:m,  m:2*m])
        hermitize(gemm('H', 'N', w, hw), out = qhq[m:2*m,m:2*m])
        hermitize(gemm('H', 'N', w, w),  out = qq[m:2*m, m:2*m])
        cp.conj(qq[:m, m:2*m], out = qq[m:2*m, :m].T)
        cp.conj(qhq[:m, m:2*m], out = qhq[m:2*m, :m].T)
        t_o = owari()
        t_mul = t_mul + t_o - t_h
        
        if iter_ > 0:
            t_h = time()
            
            gemm('H', 'N', x, hp, out = qhq[:m, 2*m:])
            gemm('H', 'N', x, p,  out = qq[:m,  2*m:])
            gemm('H', 'N', w, hp, out = qhq[m:2*m, 2*m:])
            gemm('H', 'N', w, p,  out = qq[m:2*m,  2*m:])
            hermitize(gemm('H', 'N', p, hp), out = qhq[2*m:, 2*m:])
            hermitize(gemm('H', 'N', p, p),  out = qq[2*m:,  2*m:])
            cp.conj(qq[:m, 2*m:], out = qq[2*m:,:m].T)
            cp.conj(qhq[:m, 2*m:], out = qhq[2*m:,:m].T)
            cp.conj(qq[m:2*m, 2*m:], out = qq[2*m:,m:2*m].T)
            cp.conj(qhq[m:2*m, 2*m:], out = qhq[2*m:,m:2*m].T)
            
            t_o = owari()
            t_mul = t_mul + t_o - t_h
            
            lambdas, eigvec, t_eigh = GEP(qhq, qq, slice=m)
        else:
            lambdas, eigvec, t_eigh = GEP(qhq[:2*m, :2*m], qq[:2*m, :2*m], slice=m)
            
        t_h = time()
        
        if iter_ > 0:
            gemm('N', 'N', p, eigvec[2*m:,:], out = p)
            gemm('N', 'N', w, eigvec[m:2*m,:], beta = 1.0, out = p)
            gemm('N', 'N', hp, eigvec[2*m:,:], out = hp)
            gemm('N', 'N', hw, eigvec[m:2*m,:], beta = 1.0, out = hp)
        else:
            gemm('N', 'N', w, eigvec[m:2*m,:], out = p)
            gemm('N', 'N', hw, eigvec[m:2*m,:], out = hp)
        
        gemm('N', 'N', x, eigvec[:m,:], out = x)
        x += p
        gemm('N', 'N', hx, eigvec[:m,:], out = hx)
        hx += hp
        
        peak_usage = pool.used_bytes() / (1024 ** 2)
        
        t_o = owari()
        t_mul = t_mul + t_o - t_h
        
        t_iter_o = owari()
        t_iter = t_iter_o - t_iter_h
        
        # Print
        print(f"Iter = {iter_ + 1:<4d}, residual = {cp.linalg.norm(res_nrms):<6.3e}, "
              f"n_act = {n_act}, t_iter = {t_iter:<6.3f}s, t_fft = {t_fft:<6.3f}s, "
              f"t_mul = {t_mul:<6.3f}s, t_eigh = {t_eigh:<6.3f}s, memory = {peak_usage:<6.3f}MiB.")
    
    del w, p, hx, hw, hp
    
    t_tot_o = owari()
    t_tot = t_tot_o - t_tot_h
    print(f"\nA complete procedure of lobpcg is done, {t_tot:<6.3f}s elapsed.")
    
    return lambdas[:m_conv], x[:,:m_conv], np.array([iter_,t_tot])


def lobpcg_sep_max_nolock(a_func, x0, m_conv, tol=TOL, iter_max=ITER_MAX):

    r"""
    Compute several maximum eigenvalues by: x=\lambda Ax.
    """

    if isinstance(x0, np.ndarray):
        x = cp.asarray(x0)
    else:
        x = x0.copy()
    
    m = x.shape[1]
    hx = a_func(x)
    lambdas, _ = cp.linalg.eigh(hermitize(gemm('H', 'N', x, hx)))
    
    # Local general eigenproblem.
    GEP = GEP_chol
    
    # Memory.
    pool = cp.get_default_memory_pool()
    
    # Preallocation.
    w, hw, p, hp = [cp.empty_like(x0, dtype = complex) for _ in range(4)]
    qq, qhq = [cp.empty((3*m,3*m), dtype = complex) for _ in range(2)]
    
    t_tot_h = time()
    
    for iter_ in range(iter_max):
        t_iter_h = time()
        
        # Residual, convergence and locking.
        t_h = time()
        cp.multiply(hx, lambdas, out = w)
        w -= x

        res_nrms = norms(w)
        n_act = cp.sum(res_nrms > tol)
        if max(res_nrms[:m_conv]) < tol:
            break
        
        t_o = owari()
        t_mul = t_o - t_h
        
        # Preconditioning.
        t_h = time()
        
        hw = a_func(w)

        t_o = owari()
        t_fft = t_o - t_h
        
        # Rayleigh-Ritz    
        t_h = time()
        
        hermitize(gemm('H', 'N', x, hx), out = qhq[:m,:m])
        hermitize(gemm('H', 'N', x, x),  out = qq[:m, :m])
        gemm('H', 'N', x, hw, out = qhq[:m, m:2*m])
        gemm('H', 'N', x, w,  out = qq[:m,  m:2*m])
        hermitize(gemm('H', 'N', w, hw), out = qhq[m:2*m,m:2*m])
        hermitize(gemm('H', 'N', w, w),  out = qq[m:2*m, m:2*m])
        cp.conj(qq[:m, m:2*m], out = qq[m:2*m, :m].T)
        cp.conj(qhq[:m, m:2*m], out = qhq[m:2*m, :m].T)
        t_o = owari()
        t_mul = t_mul + t_o - t_h
        
        if iter_ > 0:
            t_h = time()
            
            gemm('H', 'N', x, hp, out = qhq[:m, 2*m:])
            gemm('H', 'N', x, p,  out = qq[:m,  2*m:])
            gemm('H', 'N', w, hp, out = qhq[m:2*m, 2*m:])
            gemm('H', 'N', w, p,  out = qq[m:2*m,  2*m:])
            hermitize(gemm('H', 'N', p, hp), out = qhq[2*m:, 2*m:])
            hermitize(gemm('H', 'N', p, p),  out = qq[2*m:,  2*m:])
            cp.conj(qq[:m, 2*m:], out = qq[2*m:,:m].T)
            cp.conj(qhq[:m, 2*m:], out = qhq[2*m:,:m].T)
            cp.conj(qq[m:2*m, 2*m:], out = qq[2*m:,m:2*m].T)
            cp.conj(qhq[m:2*m, 2*m:], out = qhq[2*m:,m:2*m].T)
            
            t_o = owari()
            t_mul = t_mul + t_o - t_h
            
            lambdas, eigvec, t_eigh = GEP(qq, qhq, slice=m)
        else:
            lambdas, eigvec, t_eigh = GEP(qq[:2*m, :2*m], qhq[:2*m, :2*m], slice=m)
            
        t_h = time()
        
        if iter_ > 0:
            gemm('N', 'N', p, eigvec[2*m:,:], out = p)
            gemm('N', 'N', w, eigvec[m:2*m,:], beta = 1.0, out = p)
            gemm('N', 'N', hp, eigvec[2*m:,:], out = hp)
            gemm('N', 'N', hw, eigvec[m:2*m,:], beta = 1.0, out = hp)
        else:
            gemm('N', 'N', w, eigvec[m:2*m,:], out = p)
            gemm('N', 'N', hw, eigvec[m:2*m,:], out = hp)
        
        gemm('N', 'N', x, eigvec[:m,:], out = x)
        x += p
        gemm('N', 'N', hx, eigvec[:m,:], out = hx)
        hx += hp
        
        peak_usage = pool.used_bytes() / (1024 ** 2)
        
        t_o = owari()
        t_mul = t_mul + t_o - t_h
        
        t_iter_o = owari()
        t_iter = t_iter_o - t_iter_h
        
        # Print
        print(f"Iter = {iter_ + 1:<4d}, residual = {cp.linalg.norm(res_nrms):<6.3e}, "
              f"n_act = {n_act}, t_iter = {t_iter:<6.3f}s, t_fft = {t_fft:<6.3f}s, "
              f"t_mul = {t_mul:<6.3f}s, t_eigh = {t_eigh:<6.3f}s, memory = {peak_usage:<6.3f}MiB.")
    
    del w, p, hx, hw, hp
    
    t_tot_o = owari()
    t_tot = t_tot_o - t_tot_h
    print(f"\nA complete procedure of lobpcg is done, {t_tot:<6.3f}s elapsed.")
    
    return lambdas[:m_conv], x[:,:m_conv], np.array([iter_,t_tot])


def lobpcg_sep_softlock(a_func, p_func, x0, m_conv, tol = TOL, iter_max = ITER_MAX):
    
    if isinstance(x0, np.ndarray):
        x = cp.asarray(x0)
    else:
        x = x0.copy()
    
    m = x.shape[1]
    hx = a_func(x)
    lambdas, _ = cp.linalg.eigh(hermitize(gemm('H', 'N', x, hx)))
    
    # Local general eigenproblem.
    GEP = GEP_chol
    
    # Memory.
    pool = cp.get_default_memory_pool()
    
    # Preallocation.
    w, hw, p, hp = [cp.empty_like(x0, dtype = complex) for _ in range(4)]
    qq, qhq = [cp.empty((3*m,3*m), dtype = complex) for _ in range(2)]
    
    t_tot_h = time()
    #cp.cuda.set_allocator(None)
    for iter_ in range(iter_max):
        t_iter_h = time()
        
        # Residual, convergence and locking.
        t_h = time()
        cp.multiply(x, lambdas, out = w)
        w -= hx

        res_nrms = norms(w)
        
        # Convergence.
        ind_act = cp.where(res_nrms > tol)[0]
        n_act = len(ind_act)
        n_loc = m + 2*n_act
        if max(res_nrms[:m_conv]) < tol:
            break
        
        t_o = owari()
        t_mul = t_o - t_h
        
        # Locking (Usually locking can't occur at 1st iteration).
        t_h = time()
        if n_act < m:
            for i0 in range(n_act): # in-place copy.
                cp.copyto(w[:, i0], w[:, ind_act[i0]])
                cp.copyto(p[:, i0], p[:, ind_act[i0]])
                cp.copyto(hp[:, i0], hp[:, ind_act[i0]])
        t_o = owari()
        t_lock = t_o - t_h
        
        # Preconditioning.
        t_h = time()
        
        if n_act < m:
            w[:,:n_act] = p_func(w[:,:n_act])
        else:
            w = p_func(w)
        hw[:,:n_act] = a_func(w[:,:n_act])
        
        t_o = owari()
        t_fft = t_o - t_h
        
        # Rayleigh-Ritz    
        t_h = time()
        
        hermitize(gemm('H', 'N', x, hx), out = qhq[:m,:m])
        hermitize(gemm('H', 'N', x, x),  out = qq[:m, :m])
        gemm('H', 'N', x, hw[:,:n_act], out = qhq[:m, m:m+n_act])
        gemm('H', 'N', x, w[:,:n_act],  out = qq[:m,  m:m+n_act])
        hermitize(gemm('H', 'N', w[:,:n_act], hw[:,:n_act]), out = qhq[m:m+n_act,m:m+n_act])
        hermitize(gemm('H', 'N', w[:,:n_act], w[:,:n_act]),  out = qq[m:m+n_act, m:m+n_act])
        cp.conj(qq[:m, m:m+n_act], out = qq[m:m+n_act, :m].T)
        cp.conj(qhq[:m, m:m+n_act], out = qhq[m:m+n_act, :m].T)
        t_o = owari()
        t_mul = t_mul + t_o - t_h
        
        if iter_ > 0:
            t_h = time()
            
            gemm('H', 'N', x, hp[:,:n_act], out = qhq[:m, m+n_act:n_loc])
            gemm('H', 'N', x, p[:,:n_act],  out = qq[:m,  m+n_act:n_loc])
            gemm('H', 'N', w[:,:n_act], hp[:,:n_act], out = qhq[m:m+n_act, m+n_act:n_loc])
            gemm('H', 'N', w[:,:n_act], p[:,:n_act],  out = qq[m:m+n_act,  m+n_act:n_loc])
            hermitize(gemm('H', 'N', p[:,:n_act], hp[:,:n_act]), out = qhq[m+n_act:n_loc, m+n_act:n_loc])
            hermitize(gemm('H', 'N', p[:,:n_act], p[:,:n_act]),  out = qq[m+n_act:n_loc,  m+n_act:n_loc])
            cp.conj(qq[:m, m+n_act:n_loc], out = qq[m+n_act:n_loc,:m].T)
            cp.conj(qhq[:m, m+n_act:n_loc], out = qhq[m+n_act:n_loc,:m].T)
            cp.conj(qq[m:m+n_act, m+n_act:n_loc], out = qq[m+n_act:n_loc,m:m+n_act].T)
            cp.conj(qhq[m:m+n_act, m+n_act:n_loc], out = qhq[m+n_act:n_loc,m:m+n_act].T)
            
            t_o = owari()
            t_mul = t_mul + t_o - t_h
            
            lambdas, eigvec, t_eigh = GEP(qhq[:n_loc, :n_loc], qq[:n_loc, :n_loc], slice=m)
        else:
            lambdas, eigvec, t_eigh = GEP(qhq[:m+n_act, :m+n_act], qq[:m+n_act, :m+n_act], slice=m)
            
        t_h = time()
        
        if iter_ > 0:
            #gemm('N', 'N', p[:,:n_act], eigvec[m+n_act:,:], out = p)
            p[:] = gemm('N', 'N', p[:,:n_act], eigvec[m+n_act:,:])
            gemm('N', 'N', w[:,:n_act], eigvec[m:m+n_act,:], beta = 1.0, out = p)
            #gemm('N', 'N', hp[:,:n_act], eigvec[m+n_act:,:], out = hp)
            hp[:] = gemm('N', 'N', hp[:,:n_act], eigvec[m+n_act:,:])
            gemm('N', 'N', hw[:,:n_act], eigvec[m:m+n_act,:], beta = 1.0, out = hp)
        else:
            gemm('N', 'N', w[:,:n_act], eigvec[m:,:], out = p)
            gemm('N', 'N', hw[:,:n_act], eigvec[m:,:], out = hp)
        
        gemm('N', 'N', x, eigvec[:m,:], out = x)
        x += p
        gemm('N', 'N', hx, eigvec[:m,:], out = hx)
        hx += hp
        
        pool.free_all_blocks()
        peak_usage = pool.used_bytes() / (1024 ** 2)
        
        t_o = owari()
        t_mul = t_mul + t_o - t_h
        
        t_iter_o = owari()
        t_iter = t_iter_o - t_iter_h
        
        # Print
        print(f"Iter = {iter_ + 1:<4d}, residual = {cp.linalg.norm(res_nrms):<6.3e}, "
              f"n_act = {n_act}, t_iter = {t_iter:<6.3f}s, t_fft = {t_fft:<6.3f}s, "
              f"t_mul = {t_mul:<6.3f}s, t_eigh = {t_eigh:<6.3f}s, t_lock = {t_lock:<6.3f}s, "
              f"memory = {peak_usage:<6.3f}MiB.")
    
    del w, p, hx, hw, hp
    #pool.free_all_blocks()
    #cp.cuda.set_allocator()
    
    t_tot_o = owari()
    t_tot = t_tot_o - t_tot_h
    print(f"\nA complete procedure of lobpcg is done, {t_tot:<6.3f}s elapsed.")
    
    return lambdas[:m_conv], x[:,:m_conv], np.array([iter_,t_tot])

"""

    Block steepest descent.

"""

def descent_sep(a_func,p_func,x0,m_conv, tol = TOL, iter_max = ITER_MAX):
    
    r"""
    Usage:
        Two-term recurrence CG-type iteration (LOBPCG: three-term)
    Model: 
        Ax=\lambda x.
    
    Input:
        a_func:   function handle of matrix A.
        p_func:   function handle of preconditioning P.
        x0:       initial guess.
        m_conv:   number of desired eigenpairs.
    
    **kwargs: 
        tolerance and maximum iterations.
    
    Output:
        lambdas: eigenvalues.
        x:       eigenvectors.
        iter_:   iterations and time.
        
    """ 
    
    r"""
    initialization.
    """
    
    t_h = time()
    
    # Matrix manipulations.
    NP = arrtype(x0)
    
    # Timing (cpu/gpu).
    owari = owari_opt(str(NP))
    
    # Local eigensolver.
    if NP == cp:
        GEP = GEP_chol
    else:
        GEP = GEP_cpu
    
    nn, m = NP.shape(x0)
    nn = round(nn / 3)
    
    # Initialization.     
    x = x0.copy()
    hx = a_func(x)    
    lambdas, _ = NP.linalg.eigh(hermitize(x.T.conj() @ hx)) 
    
    t_o = owari()
    print(f"Time for initialization: {t_o - t_h:<6.3f}s.")
    
    """
    Main loop.
    """
    
    t_tot_h = time()
    
    for iter_ in range(iter_max):
        t_iter_h = time()
        
        # Residual, convergence and locking.
        t_h = time()
        w = hx - x * lambdas
        
        res_nrms, x_nrms = norms(w), norms(x)

        ind_act = NP.where(res_nrms/ x_nrms>tol)[0]
        n_act = len(ind_act)

        if n_act <= m - m_conv:
            break
        
        t_o = owari()
        t_mul = t_o - t_h
        
        # Preconditioning.
        t_h = time()
        w = p_func(w[:,ind_act])
        hw = a_func(w)

        t_o=owari()
        t_fft = t_o - t_h
        
        # Rayleigh-Ritz    
        t_h = time()
        t11, g11 = hermitize(x.T.conj() @ hx), hermitize(x.T.conj() @ x)
        t12, g12 = x.T.conj() @ hw, x.T.conj() @ w
        t22, g22 = hermitize(w.T.conj() @ hw), hermitize(w.T.conj() @ w)
        
        t_o = owari()
        t_mul = t_mul + t_o - t_h
        
        lambdas, eigvec, t_eigh = GEP(NP.concatenate((NP.concatenate((t11,t12),axis=1),NP.concatenate((t12.T.conj(),t22),axis=1)),axis=0),\
                                      NP.concatenate((NP.concatenate((g11,g12),axis=1),NP.concatenate((g12.T.conj(),g22),axis=1)),axis=0))
        
        t_h = time()
        
        lambdas = lambdas[0:m]
        eigvec  = eigvec[:,0:m]
        x[:] = x @ eigvec[0:m,:] + w @ eigvec[m:m+n_act,:]
        hx[:] = hx @ eigvec[0:m,:] + hw @ eigvec[m:m+n_act,:]

        t_o = owari()
        t_mul = t_mul + t_o-t_h
        
        t_iter_o = owari()
        t_iter = t_iter_o - t_iter_h
        
        # Print        
        print(f"Iter = {iter_:<4d}, residual = {NP.linalg.norm(res_nrms):<6.3e}, "
              f"n_act = {n_act}, t_iter = {t_iter:<6.3f}s, t_fft = {t_fft:<6.3f}s, "
              f"t_mul = {t_mul:<6.3f}s, t_eigh = {t_eigh:<6.3f}s.")
    
    del w,hx,hw
    
    t_tot_o = owari()
    t_tot = t_tot_o - t_tot_h
    print(f"\nA complete procedure of lobpcg is done, {t_tot:<6.3f}s elapsed.")
    
    return lambdas[0:m_conv], x[:,:m_conv], np.array([iter_,t_tot])

def descent_gep(a_func,b_func,p_func,x0,m_conv, tol = TOL, iter_max = ITER_MAX):

    r"""
    Usage:
        Steepest descent for generalized eigenvalue problems (GEP).
    Model: 
        Ax=\lambda Bx.
    
    Input:
        a_func:   function handle of matrix A.
        b_func:   function handle of matrix B (SPD/HPD).
        p_func:   function handle of preconditioning P.
        x0:       initial guess.
        m_conv:   number of desired eigenpairs.
    
    **kwargs: 
        tolerance and maximum iterations.
   
    Output:
        lambdas: eigenvalues.
        x:       eigenvectors.
        iter_:   iterations and time.
    """ 
    
    r"""
    LOBPCG: initialization.
    """
    
    t_h = time()
    
    # Matrix manipulations.
    NP = arrtype(x0)
    
    # Timing (cpu/gpu).
    owari = owari_opt(str(NP))
    
    # Local eigensolver.
    if NP == cp:
        GEP = GEP_chol
    else:
        GEP = GEP_cpu
    
    nn, m = NP.shape(x0)
    nn = round(nn/3)
    
    # Initialization.    
    x = x0.copy()
    hx = a_func(x)
    mx = b_func(x)   
    lambdas, _ , _ = GEP_chol(hermitize(x.T.conj() @ hx), hermitize(x.T.conj() @ mx))
    
    t_o = owari()
    print(f"Time for initialization: {t_o - t_h:<6.3f}s.")
    
    """
    LOBPCG: Main loop.
    """
    
    t_tot_h = time()
    for iter_ in range(iter_max):
        t_iter_h = time()
        
        # Residual, convergence and locking.
        t_h = time()
        w = hx - mx * lambdas
        
        res_nrms,x_nrms = norms(w),norms(x)

        ind_act = NP.where(res_nrms / x_nrms > tol)[0]
        n_act = len(ind_act)

        if n_act <= m - m_conv:
            break
        
        t_o = owari()
        t_mul = t_o - t_h
        
        # Preconditioning.
        t_h = time()
        w = p_func(w[:,ind_act])
        hw = a_func(w)
        mw = b_func(w)

        t_o = owari()
        t_fft = t_o - t_h
        
        # Rayleigh-Ritz    
        
        t_h = time()
        t11,g11 = hermitize(x.T.conj() @ hx), hermitize(x.T.conj() @ mx)
        t12,g12 = x.T.conj() @ hw, x.T.conj() @ mw
        t22,g22 = hermitize(w.T.conj() @ hw), hermitize(w.T.conj() @ mw)

        t_o = owari()
        t_mul = t_mul + t_o - t_h
        
        lambdas, eigvec, t_eigh = GEP(NP.concatenate((NP.concatenate((t11,t12),axis=1),NP.concatenate((t12.T.conj(),t22),axis=1)),axis=0),\
                                      NP.concatenate((NP.concatenate((g11,g12),axis=1),NP.concatenate((g12.T.conj(),g22),axis=1)),axis=0))
        
        t_h = time()
        
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
        print(f"Iter = {iter_:<4d}, residual = {NP.linalg.norm(res_nrms):<6.3e}, "
              f"n_act = {n_act}, t_iter = {t_iter:<6.3f}s, t_fft = {t_fft:<6.3f}s, "
              f"t_mul = {t_mul:<6.3f}s, t_eigh = {t_eigh:<6.3f}s.")
    
    del w,hx,hw
    
    t_tot_o = owari()
    t_tot = t_tot_o - t_tot_h
    print(f"\nA complete procedure of lobpcg is done, {t_tot:<6.3f}s elapsed.")
    
    return lambdas[:m_conv], x[:,:m_conv], np.array([iter_,t_tot])


def test_max_min():



    return 


def main():

    test_max_min()

if __name__ == '__main__':
    main()