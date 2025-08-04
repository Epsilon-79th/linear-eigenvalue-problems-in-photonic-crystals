# -*- coding: utf-8 -*-
#
# Created on 2025-03-25 (Tuesday) at 22:37:39
#
# Author: Epsilon-79th
#
# Usage: Davidson.
#

from time import time
from time import time

import numpy as np
import scipy as sc
import cupy as cp

from gpu_opts import *
from eigensolver import *

TOL = 1e-5
ITER_MAX = 1000

owari = owari_opt(str(cp))

def davidson_sep_locking(a_func,p_func,x0,m_conv, nblock = 4, tol = TOL, iter_max = ITER_MAX):
    
    r"""
    Usage:
        Davidson method for simple eigenvalue problems (SEP).
        Model: Ax=\lambda x.
    
    Input:
        a_func:   function handle of matrix A.
        p_func:   function handle of preconditioning P.
        x0:       initial guess.
        m_conv:   number of desired eigenpairs.
    
    **kwargs: 
        tolerance and maximum iterations.
    
    
    Output.
        lambdas: eigenvalues.
        x:       eigenvectors.
        iter_:   iterations and time.
        
    """
    
    r"""
    LOBPCG: initialization.
    """
    
    t_h = time()
    
    # Local eigensolver.
    GEP = GEP_cpu

    m = x0.shape[1]
    nsub_max, nsub = nblock * m, m
    q, hq = [cp.empty((x0.shape[0], nsub_max), dtype = complex) for _ in range(2)]
    q[:,:m], _ = cp.linalg.qr(x0)
    hq[:,:m] = a_func(q[:,:m])
    
    s, hs = cp.identity(m), hermitize(q[:,:m].T.conj() @ hq[:,:m])
    lambdas, v = cp.linalg.eigh(hs) 
    p = hq[:,:m] - q[:,:m] * lambdas
    r_nrms = norms(p)
    
    memory_pool = cp.get_default_memory_pool()
    
    t_o = owari()
    print(f"Time for initialization:{t_o - t_h:<6.3f}s.")
    
    r"""
    Davidson: Main loop.
    """
    
    t_tot_h = time()
    
    for iter_ in range(0,iter_max):
        t_iter_h = time()
        
        # Residual, convergence and locking.
        t_h = time()

        ind_act = cp.where(r_nrms>tol)[0]
        n_act = len(ind_act)

        if max(r_nrms[:m_conv]) < tol:
            break
        elif nsub + n_act > nsub_max:
            # shrink subspace.
            q[:,:m], hq[:,:m] = q[:,:nsub] @ v[:,:m], hq[:,:nsub] @ v[:,:m]
            s,hs = cp.identity(m), v[:,:m].T.conj() @ (hs @ v[:,:m])
            nsub = m
            print("Restart.")
        
        t_o = owari()
        t_else = t_o - t_h
        
        # Preconditioning.
        t_h = time()
        
        # locking.
        if n_act < m:
            for i in range(n_act):
                cp.copyto(p[:,i], p[:,ind_act[i]])
        
        if n_act < m:
            print(f"usage = {memory_pool.used_bytes() / (1024 ** 2):<6.3f}MiB.")
        #p = p_func(p)
        p[:,:n_act] = p_func(p[:,:n_act])
        
        p[:,:n_act] /= norms(p[:,:n_act])
        cp.copyto(q[:, nsub:nsub+n_act], p[:,:n_act])
        
        if n_act < m:
            print(f"usage = {memory_pool.used_bytes() / (1024 ** 2):<6.3f}MiB.")
        
        hq[:, nsub:nsub+n_act] = a_func(p[:,:n_act])
        t_o = owari()
        t_fft = t_o - t_h
        
        if n_act < m:
            print(f"usage = {memory_pool.used_bytes() / (1024 ** 2):<6.3f}MiB.")
        # Local.
        t_h = time()
        g2 = p[:,:n_act].T.conj() @ q[:,:nsub]
        g3 = hermitize(p[:,:n_act].T.conj() @ p[:,:n_act])
        s = cp.concatenate((cp.concatenate((s,g2.T.conj()), axis = 1), cp.concatenate((g2,g3),axis = 1)), axis = 0)
        t_o = owari()
        t_else += t_o - t_h
 
        t_h = time()
        t2 = p[:,:n_act].T.conj() @ hq[:,:nsub]
        t3 = hermitize(p[:,:n_act].T.conj() @ hq[:, nsub:nsub+n_act])
        hs = cp.concatenate((cp.concatenate((hs,t2.T.conj()), axis = 1), cp.concatenate((t2,t3),axis = 1)), axis = 0)
        t_o = owari()
        t_else += t_o - t_h
        
        t_h = time()
        lambdas,v, _ = GEP(hs,s)
        lambdas = lambdas[:m]
        t_o = owari()
        t_rr = t_o - t_h
        
        t_h = time()
        
        nsub += n_act
        p = hq[:,:nsub] @ v[:,:m] - (q[:,:nsub] @ v[:,:m]) * lambdas
        r_nrms = norms(p)
        t_o = owari()
        t_else += t_o - t_h
        
        peak_usage = memory_pool.used_bytes() / (1024 ** 2)
        
        t_iter_o = owari()
        t_iter = t_iter_o - t_iter_h

        # Print.
        print(f"Iter = {iter_+1}, residual = {norm(r_nrms):<6.3e}, n_act = {n_act}, " 
              f"t_iter = {t_iter:<6.3f}s, t_fft = {t_fft:<6.3f}s",
              f"t_RR = {t_rr:<6.3f}s, t_else = {t_else:<6.3f}s, memory = {peak_usage:<6.3f}MiB.")   
    
    t_tot_o = owari()
    t_tot = t_tot_o - t_tot_h
    print(f"Davidson for SEP is completed, {t_tot:>6.3f}s elapsed.")

    return lambdas[:m_conv], q[:,:nsub] @ v[:,:m_conv], [iter_,t_tot]


def davidson_sep(a_func,p_func,x0,m_conv, nblock_max = 4, tol = TOL, iter_max = ITER_MAX):
    
    t_h = time()
    owari = owari_opt('cp')
    memory_pool = cp.get_default_memory_pool()
    
    nn, m = x0.shape
    x = cp.empty(x0.shape, dtype = complex)     # Eigenvectors.
    v, hv = [cp.empty((nn, nblock_max * m), dtype = complex) for _ in range(2)]     # Search space.
    p = cp.empty((nn, m), dtype = complex)      # Store both a_func(v) and direction.
    
    lambdas = cp.empty(nblock_max * m, dtype = float)
    H_loc, M_loc, y = [cp.empty((nblock_max * m, nblock_max * m), dtype = complex) for _ in range(3)]
    v[:,:m] = x0.copy()
    hv[:,:m] = a_func(v[:,:m])
    
    H_loc[:m,:m] = hermitize(v[:,:m].T.conj()@hv[:,:m])
    M_loc[:m,:m] = hermitize(v[:,:m].T.conj()@v[:,:m])
    
    nblock = 1
    t_o = owari()
    print(f"Runtime of initialization: {t_o - t_h:<6.3f}s.")
    
    t_tot_h = time()
    for iter_ in range(iter_max):
        t_iter_h = time()
        t_h = time()
        lambdas[:nblock*m], y[:nblock*m,:nblock*m], _ = GEP_cpu(H_loc[:nblock*m,:nblock*m], M_loc[:nblock*m,:nblock*m])
        x = v[:, :nblock*m] @ y[:nblock*m,:m]
        t_o = owari()
        t_rr = t_o - t_h
        
        # Residual and convergence.
        t_h = time()
        p = hv[:, :nblock*m] @ y[:nblock*m,:m] - x * lambdas[:m]
        r_nrms = norms(p)
        n_act = len(cp.where(r_nrms > tol)[0])
        if max(r_nrms[:m_conv]) < tol:
            break
        
        # Shrink if necessary.
        if nblock == nblock_max:
            nblock = 1
            v[:,:m] = x.copy()
            hv[:,:m] = hv @ y[:,:m]
            M_loc[:m,:m] = cp.eye(m)
            H_loc[:m,:m] = hermitize(y[:,:m].T.conj() @ H_loc @ y[:,:m])
            print("Restart.")
            #continue
        t_o = owari()
        t_else_1 = t_o - t_h
        
        # Precondition.
        t_h = time()
        p = p_func(p)
        t_o = owari()
        t_fft = t_o - t_h
        
        # Orthogonalization and expansion of search space v.
        t_h = time()
        cp.copyto(v[:, nblock*m:(nblock+1)*m], p)
        t_o = owari()
        t_else_2 = t_o - t_h
        
        t_h = time()
        hv[:,nblock*m:(nblock+1)*m] = a_func(p) #hp
        t_o = owari()
        t_fft += t_o - t_h
        
        # Expansion of local matrix.
        t_h = time()
        H_loc[:nblock*m, nblock*m:(nblock+1)*m] = v[:,:nblock*m].T.conj() @ hv[:,nblock*m:(nblock+1)*m]
        H_loc[nblock*m:(nblock+1)*m, :nblock*m] = H_loc[:nblock*m, nblock*m:(nblock+1)*m].T.conj()
        H_loc[nblock*m:(nblock+1)*m, nblock*m:(nblock+1)*m] = hermitize(p.T.conj() @ hv[:,nblock*m:(nblock+1)*m])
        
        M_loc[:nblock*m, nblock*m:(nblock+1)*m] = v[:,:nblock*m].T.conj() @ p
        M_loc[nblock*m:(nblock+1)*m, :nblock*m] = M_loc[:nblock*m, nblock*m:(nblock+1)*m].T.conj()
        M_loc[nblock*m:(nblock+1)*m, nblock*m:(nblock+1)*m] = hermitize(p.T.conj() @ p)
        t_o = owari()
        t_else_3 = t_o - t_h
        
        nblock += 1
        peak_usage = memory_pool.used_bytes() / (1024 ** 2)
        t_iter_o = owari()
        
        # Print.
        print(f"Iter = {iter_+1}, residual = {norm(r_nrms):<6.3e}, n_act = {n_act}, " 
              f"t_iter = {t_iter_o-t_iter_h:<6.3f}s, t_fft = {t_fft:<6.3f}s",
              f"t_RR = {t_rr:<6.3f}s, memory = {peak_usage:<6.3f}MiB.",
              f"t_else = ({t_else_1:<6.3f}s, {t_else_2:<6.3f}s, {t_else_3:<6.3f}s)")
        
    t_tot_o = owari()
    return lambdas[:m_conv], x[:,:m_conv], [iter_, t_tot_o - t_tot_h]

"""
Non-Hermtian Eigenproblem (NHSEP).
"""

def davidson_sep_nonhermitian(a_func,p_func,x0,m_conv, nblock_max = 4, tol = TOL, iter_max = ITER_MAX):
    
    t_h = time()
    owari = owari_opt('cp')
    memory_pool = cp.get_default_memory_pool()
    
    nn, m = x0.shape
    x = cp.empty(x0.shape, dtype = complex)     # Eigenvectors.
    v, hv = [cp.empty((nn, nblock_max * m), dtype = complex) for _ in range(2)]     # Search space.
    p = cp.empty((nn, m), dtype = complex)      # Store both a_func(v) and direction.
    
    lambdas = cp.empty(nblock_max * m, dtype = float)
    H_loc, M_loc, y = [cp.empty((nblock_max * m, nblock_max * m), dtype = complex) for _ in range(3)]
    v[:,:m] = x0.copy()
    hv[:,:m] = a_func(v[:,:m])
    
    H_loc[:m,:m] = v[:,:m].T.conj()@hv[:,:m]
    M_loc[:m,:m] = hermitize(v[:,:m].T.conj()@v[:,:m])
    
    nblock = 1
    t_o = owari()
    print(f"Runtime of initialization: {t_o - t_h:<6.3f}s.")
    
    t_tot_h = time()
    for iter_ in range(iter_max):
        t_iter_h = time()
        t_h = time()
        lambdas[:nblock*m], y[:nblock*m,:nblock*m], _ = GEP_cpu(H_loc[:nblock*m,:nblock*m], M_loc[:nblock*m,:nblock*m], herm = False)
        x = v[:, :nblock*m] @ y[:nblock*m,:m]
        t_o = owari()
        t_rr = t_o - t_h
        
        # Residual and convergence.
        t_h = time()
        p = hv[:, :nblock*m] @ y[:nblock*m,:m] - x * lambdas[:m]
        r_nrms = norms(p)
        n_act = len(cp.where(r_nrms > tol)[0])
        if max(r_nrms[:m_conv]) < tol:
            break
        
        # Shrink if necessary.
        if nblock == nblock_max:
            nblock = 1
            v[:,:m] = x.copy()
            hv[:,:m] = hv @ y[:,:m]
            M_loc[:m,:m] = cp.eye(m)
            H_loc[:m,:m] = y[:,:m].T.conj() @ H_loc @ y[:,:m]
            print("Restart.")
            #continue
        t_o = owari()
        t_else_1 = t_o - t_h
        
        # Precondition.
        t_h = time()
        p = p_func(p)
        t_o = owari()
        t_fft = t_o - t_h
        
        # Orthogonalization and expansion of search space v.
        t_h = time()
        cp.copyto(v[:, nblock*m:(nblock+1)*m], p)
        t_o = owari()
        t_else_2 = t_o - t_h
        
        t_h = time()
        hv[:,nblock*m:(nblock+1)*m] = a_func(p) #hp
        t_o = owari()
        t_fft += t_o - t_h
        
        # Expansion of local matrix.
        t_h = time()
        H_loc[:nblock*m, nblock*m:(nblock+1)*m] = v[:,:nblock*m].T.conj() @ hv[:,nblock*m:(nblock+1)*m]
        H_loc[nblock*m:(nblock+1)*m, :nblock*m] = p.T.conj() @ hv[:,:nblock*m]
        H_loc[nblock*m:(nblock+1)*m, nblock*m:(nblock+1)*m] = p.T.conj() @ hv[:,nblock*m:(nblock+1)*m]
        
        M_loc[:nblock*m, nblock*m:(nblock+1)*m] = v[:,:nblock*m].T.conj() @ p
        M_loc[nblock*m:(nblock+1)*m, :nblock*m] = M_loc[:nblock*m, nblock*m:(nblock+1)*m].T.conj()
        M_loc[nblock*m:(nblock+1)*m, nblock*m:(nblock+1)*m] = hermitize(p.T.conj() @ p)
        t_o = owari()
        t_else_3 = t_o - t_h
        
        nblock += 1
        peak_usage = memory_pool.used_bytes() / (1024 ** 2)
        t_iter_o = owari()
        
        # Print.
        print(f"Iter = {iter_+1}, residual = {norm(r_nrms):<6.3e}, n_act = {n_act}, " 
              f"t_iter = {t_iter_o-t_iter_h:<6.3f}s, t_fft = {t_fft:<6.3f}s",
              f"t_RR = {t_rr:<6.3f}s, memory = {peak_usage:<6.3f}MiB.",
              f"t_else = ({t_else_1:<6.3f}s, {t_else_2:<6.3f}s, {t_else_3:<6.3f}s)")
        
    t_tot_o = owari()
    return lambdas[:m_conv], x[:,:m_conv], [iter_, t_tot_o - t_tot_h]
    
    return
