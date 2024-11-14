# -*- coding: utf-8 -*-
#
# Created on 2024-06-10 (Monday) at 18:05:57
#
# Author: Epsilon-79th
#
# Usage: Self-defined eigensolvers.
#

from time import time

import numpy as np
import scipy as sc
import cupy as cp

import discretization as mfd
from discretization import A_fft,H_fft
from gpu_opts import norm,norms,dots,arrtype,owari_opt,owari_cuda

TOL = 1e-5
ITER_MAX = 1000

def default_kwargs_eigen_solver():
    
    return dict(
        tol = TOL,
        iter_max = ITER_MAX
    )

def descent_sep(a_func,p_func,x0,m_conv,**kwargs):
    
    """
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
    
    Output：
        lambdas: eigenvalues.
        x:       eigenvectors.
        iter_:   iterations and time.
        
    """ 
    
    
    """
    LOBPCG: initialization.
    """
    
    t_h=time()
    
    # Defaults.
    opts = default_kwargs_eigen_solver()
    opts.update(kwargs)
    
    # Shift, tolerance, maximum iterations.
    tol,iter_max = opts["tol"],opts["iter_max"]
    
    # Matrix manipulations.
    NP = arrtype(x0)
    MUL,HM,VM = NP.dot,NP.hstack,NP.vstack
    
    # Timing (cpu/gpu).
    owari = owari_opt(str(NP))
    
    # Local eigensolver.
    if NP == cp:
        GEP = GEP_chol
    else:
        GEP = sc.linalg.eigh
    
    # Dielectric setting.
    Diels = mfd.scalar_prod
    
    nn,m = NP.shape(x0)
    nn = round(nn/3)
    
    # Initialization.
    #Q=NP.zeros((3*n,3*m),dtype=complex)         
    x = x0
    del x0
    hx = a_func(x)    
    lambdas,__ = NP.linalg.eigh(MUL(x.T.conj(),hx)) 
    
    t_o = owari()
    print("Time for initialization:",'%6.3f'%(t_o - t_h),"s.")
    
    """
    Main loop.
    """
    
    t_tot_h=time()
    
    for iter_ in range(iter_max):
        t_iter_h=time()
        
        # Residual, convergence and locking.
        t_h=time()
        w=hx-x*lambdas
        
        res_nrms,x_nrms=norms(w),norms(x)

        ind_act=NP.where(res_nrms/x_nrms>tol)[0]
        n_act=len(ind_act)

        if n_act<=m-m_conv:
            break
        
        t_o=owari()
        t_mul=t_o-t_h
        
        # Preconditioning.
        t_h=time()
        w=p_func(w[:,ind_act])
        hw=a_func(w)

        t_o=owari()
        t_fft=t_o-t_h
        
        # Rayleigh-Ritz    
        
        t_h=time()
        t11,g11=mfd.hermitize(MUL(x.T.conj(),hx)),mfd.hermitize(MUL(x.T.conj(),x))
        t12,g12=MUL(x.T.conj(),hw),MUL(x.T.conj(),w)
        t22,g22=mfd.hermitize(MUL(w.T.conj(),hw)),mfd.hermitize(MUL(w.T.conj(),w))
        
        t_o=owari()
        t_mul=t_mul+t_o-t_h
        
        lambdas,eigvec,t_eigh=GEP(VM((HM((t11,t12)),HM((t12.T.conj(),t22)))),\
                                  VM((HM((g11,g12)),HM((g12.T.conj(),g22)))))
        
        t_h=time()
        
        lambdas=lambdas[0:m]
        eigvec=eigvec[:,0:m]
        x=MUL(x,eigvec[0:m,:])+MUL(w,eigvec[m:m+n_act,:])
        hx=MUL(hx,eigvec[0:m,:])+MUL(hw,eigvec[m:m+n_act,:])

        t_o=owari()
        t_mul=t_mul+t_o-t_h
        
        t_iter_o=owari()
        t_iter=t_iter_o-t_iter_h
        
        # Print        
        print(f"Iter={iter_}, residual={NP.linalg.norm(res_nrms):<6.3e}, n_act={n_act}, "
              f"t_iter={t_iter:<6.3f}s, t_fft={t_fft:<6.3f}s, t_mul={t_mul:<6.3f}s, "
              f"t_eigh={t_eigh:<6.3f}s.")
    
    del w,hx,hw
        
    lambdas = lambdas.get()
    
    t_tot_o = owari()
    t_tot = t_tot_o - t_tot_h
    print("\nA complete procedure of lobpcg is done, ",'%6.3f'%t_tot,"s elapsed.")
    
    return lambdas[0:m_conv],x,np.array([iter_,t_tot])

def descent_gep(a_func,b_func,p_func,x0,m_conv,**kwargs):

    """
    Usage:
        LOBPCG program for generalized eigenvalue problems (GEP).
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
    
    """
    LOBPCG: initialization.
    """
    
    t_h = time()
    
    # Defaults.
    opts = default_kwargs_eigen_solver()
    opts.update(kwargs)
    
    # Shift, tolerance, maximum iterations.
    tol,iter_max = opts["tol"],opts["iter_max"]
    
    # Matrix manipulations.
    NP = arrtype(x0)
    MUL,HM,VM = NP.dot,NP.hstack,NP.vstack
    
    # Timing (cpu/gpu).
    owari = owari_opt(str(NP))
    
    # Local eigensolver.
    if NP == cp:
        GEP = GEP_chol
    else:
        GEP = sc.linalg.eigh
    
    # Dielectric setting.
    Diels = mfd.scalar_prod
    
    nn,m = NP.shape(x0)
    nn=round(nn/3)
    
    # Initialization.    
    x = x0
    del x0
    hx = a_func(x)
    mx = b_func(x)   
    lambdas,__=NP.linalg.eigh(MUL(x.T.conj(),hx).get(),(x.T.conj(),mx).get)
    
    t_o=owari()
    print("Time for initialization:",'%6.3f'%(t_o - t_h),"s.")
    
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
        t11,g11 = mfd.hermitize(MUL(x.T.conj(),hx)),mfd.hermitize(MUL(x.T.conj(),mx))
        t12,g12 = MUL(x.T.conj(),hw),MUL(x.T.conj(),mw)
        t22,g22 = mfd.hermitize(MUL(w.T.conj(),hw)),mfd.hermitize(MUL(w.T.conj(),mw))

        t_o = owari()
        t_mul = t_mul + t_o - t_h
        
        lambdas,eigvec,t_eigh = GEP(VM((HM((t11,t12)),HM((t12.T.conj(),t22)))),\
                                    VM((HM((g11,g12)),HM((g12.T.conj(),g22)))))
        
        t_h = time()
        
        lambdas = lambdas[0:m]
        eigvec = eigvec[:,0:m]
        
        x = MUL(x,eigvec[0:m,:])+MUL(w,eigvec[m:m + n_act,:])
        hx = MUL(hx,eigvec[0:m,:])+MUL(hw,eigvec[m:m + n_act,:])
        mx = MUL(mx,eigvec[0:m,:])+MUL(mw,eigvec[m:m + n_act,:])

        t_o = owari()
        t_mul = t_mul + t_o - t_h
        
        t_iter_o = owari()
        t_iter = t_iter_o - t_iter_h
        
        # Print
        print(f"Iter={iter_}, residual={NP.linalg.norm(res_nrms):<6.3e}, "
              f"n_act={n_act}, t_iter={t_iter:<6.3f}s, t_fft={t_fft:<6.3f}s, "
              f"t_mul={t_mul:<6.3f}s, t_eigh={t_eigh:<6.3f}s.")
    
    del w,hx,hw
        
    lambdas = lambdas.get()
    
    t_tot_o = owari()
    t_tot = t_tot_o - t_tot_h
    print(f"\nA complete procedure of lobpcg is done, {t_tot:<6.3f}s elapsed.")
    
    return lambdas[0:m_conv],x,np.array([iter_,t_tot])

def lobpcg_sep(a_func,p_func,x0,m_conv,**kwargs):
    
    """
    Usage:
        LOBPCG program for simple eigenvalue problems (SEP).
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
    
    """
    LOBPCG: initialization.
    """
    
    t_h = time()
    
    # Defaults.
    opts = default_kwargs_eigen_solver()
    opts.update(kwargs)
    
    # Shift, tolerance, maximum iterations.
    tol_lobpcg,iter_max = opts["tol"],opts["iter_max"]
    
    # Matrix manipulations.
    NP = arrtype(x0)
    MUL,HM,VM = NP.dot,NP.hstack,NP.vstack
    
    # Timing (cpu/gpu).
    owari = owari_opt(str(NP))
    
    # Local eigensolver.
    if NP == cp:
        GEP = GEP_chol
    else:
        GEP = sc.linalg.eigh
    
    # Initialization.
    x,m = x0,x0.shape[1]
    del x0
    hx = a_func(x)    
    lambdas,__ = NP.linalg.eigh(MUL(x.T.conj(),hx))    
    p,hp = [],[]
    
    t_o = owari()
    print("Time for initialization:",'%6.3f'%(t_o-t_h),"s.")
    
    """
    LOBPCG: Main loop.
    """
    
    t_tot_h = time()
    
    for iter_ in range(iter_max):
        t_iter_h = time()
        
        # Residual, convergence and locking.
        t_h = time()
        w = hx - x * lambdas
        
        res_nrms,x_nrms = norms(w),norms(x)

        ind_act = NP.where(res_nrms / x_nrms > tol_lobpcg)[0]
        n_act = len(ind_act)

        if n_act <= m - m_conv:
            break
        
        t_o = owari()
        t_mul = t_o - t_h
        
        # Preconditioning.
        t_h = time()
        w = p_func(w[:,ind_act])
        hw = a_func(w)

        t_o = owari()
        t_fft = t_o - t_h
        
        # Rayleigh-Ritz    
        
        t_h = time()
        t11,g11 = mfd.hermitize(MUL(x.T.conj(),hx)),mfd.hermitize(MUL(x.T.conj(),x))
        t12,g12 = MUL(x.T.conj(),hw),MUL(x.T.conj(),w)
        t22,g22 = mfd.hermitize(MUL(w.T.conj(),hw)),mfd.hermitize(MUL(w.T.conj(),w))

        t_o = owari()
        t_mul = t_mul + t_o - t_h
        
        if iter_ > 0:
            t_h = time()
            p = p[:,ind_act]
            hp = hp[:,ind_act]
            t13,g13 = MUL(x.T.conj(),hp),MUL(x.T.conj(),p)
            t23,g23 = MUL(w.T.conj(),hp),MUL(w.T.conj(),p)
            t33,g33 = mfd.hermitize(MUL(p.T.conj(),hp)),mfd.hermitize(MUL(p.T.conj(),p))
            t_o = owari()
            t_mul = t_mul + t_o - t_h
            
            lambdas,eigvec,t_eig = GEP(VM((HM((t11,t12,t13)),\
                                       HM((t12.T.conj(),t22,t23)),\
                                       HM((t13.T.conj(),t23.T.conj(),t33)))),\
                                       VM((HM((g11,g12,g13)),\
                                       HM((g12.T.conj(),g22,g23)),\
                                       HM((g13.T.conj(),g23.T.conj(),g33)))))
        else:
            lambdas,eigvec,t_eigh = GEP(VM((HM((t11,t12)),HM((t12.T.conj(),t22)))),\
                                        VM((HM((g11,g12)),HM((g12.T.conj(),g22)))))
        
        t_h = time()
        
        lambdas = lambdas[0:m]
        eigvec = eigvec[:,0:m]
        
        if iter_ > 0:
            p = MUL(w,eigvec[m:m + n_act,:]) + MUL(p,eigvec[m + n_act:,:])
            hp = MUL(hw,eigvec[m:m + n_act,:]) + MUL(hp,eigvec[m + n_act:,:])
        else:
            p = MUL(w,eigvec[m:m + n_act,:])
            hp = MUL(hw,eigvec[m:m + n_act,:])
        
        x = MUL(x,eigvec[0:m,:]) + p
        hx = MUL(hx,eigvec[0:m,:]) + hp

        t_o = owari()
        t_mul = t_mul + t_o - t_h
        
        t_iter_o = owari()
        t_iter = t_iter_o - t_iter_h
        
        # Print
        print(f"Iter={iter_}, residual={NP.linalg.norm(res_nrms):<6.3e}, "
              f"n_act={n_act}, t_iter={t_iter:<6.3f}s, t_fft={t_fft:<6.3f}s, "
              f"t_mul={t_mul:<6.3f}, t_eigh={t_eigh:<6.3f}s.")
    
    del w,p,hx,hw,hp
        
    lambdas = lambdas.get()
    
    t_tot_o = owari()
    t_tot = t_tot_o - t_tot_h
    print(f"\nA complete procedure of lobpcg is done, {t_tot:<6.3f}s elapsed.")
    
    return lambdas[0:m_conv],x,np.array([iter_,t_tot])

def lobpcg_gep(a_func,b_func,p_func,x0,m_conv,**kwargs):

    """
    Usage:
        LOBPCG program for generalized eigenvalue problems (GEP).
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
    
    """
    LOBPCG: initialization.
    """
    
    t_h = time()
    
    # Defaults.
    opts = default_kwargs_eigen_solver()
    opts.update(kwargs)
    
    # Shift, tolerance, maximum iterations.
    tol_lobpcg,iter_max = opts["tol"],opts["iter_max"]
    
    # Matrix manipulations.
    NP = arrtype(x0)
    MUL,HM,VM = NP.dot,NP.hstack,NP.vstack
    
    # Timing (cpu/gpu).
    owari = owari_opt(str(NP))
    
    # Local eigensolver.
    if NP == cp:
        GEP = GEP_chol
    else:
        GEP = sc.linalg.eigh
    
    # Initialization.         
    x,m = x0,x0.shape[1]
    del x0
    hx = a_func(x)
    mx = b_func(x)   
    lambdas,__ = NP.linalg.eigh(MUL(x.T.conj(),hx).get(),(x.T.conj(),mx).get)    
    p,mp,hp = [],[],[]
    
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

        ind_act = NP.where(res_nrms / x_nrms > tol_lobpcg)[0]
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
        t11,g11 = mfd.hermitize(MUL(x.T.conj(),hx)),mfd.hermitize(MUL(x.T.conj(),mx))
        t12,g12 = MUL(x.T.conj(),hw),MUL(x.T.conj(),mw)
        t22,g22 = mfd.hermitize(MUL(w.T.conj(),hw)),mfd.hermitize(MUL(w.T.conj(),mw))

        t_o = owari()
        t_mul = t_mul + t_o - t_h
        
        if iter_ > 0:
            t_h = time()
            p = p[:,ind_act]
            hp = hp[:,ind_act]
            t13,g13 = MUL(x.T.conj(),hp),MUL(x.T.conj(),mp)
            t23,g23 = MUL(w.T.conj(),hp),MUL(w.T.conj(),mp)
            t33,g33 = mfd.hermitize(MUL(p.T.conj(),hp)),mfd.hermitize(MUL(p.T.conj(),mp))
            t_o = owari()
            t_mul = t_mul + t_o - t_h
            
            lambdas,eigvec,t_eig = GEP(VM((HM((t11,t12,t13)),\
                                       HM((t12.T.conj(),t22,t23)),\
                                       HM((t13.T.conj(),t23.T.conj(),t33)))),\
                                       VM((HM((g11,g12,g13)),\
                                       HM((g12.T.conj(),g22,g23)),\
                                       HM((g13.T.conj(),g23.T.conj(),g33)))))
        else:
            lambdas,eigvec,t_eigh = GEP(VM((HM((t11,t12)),HM((t12.T.conj(),t22)))),\
                                        VM((HM((g11,g12)),HM((g12.T.conj(),g22)))))
        
        t_h = time()
        
        lambdas = lambdas[0:m]
        eigvec = eigvec[:,0:m]
        if iter_ > 0:
            p = MUL(w,eigvec[m:m + n_act,:])+MUL(p,eigvec[m + n_act:,:])
            hp = MUL(hw,eigvec[m:m + n_act,:])+MUL(hp,eigvec[m + n_act:,:])
            mp = MUL(mw,eigvec[m:m + n_act,:])+MUL(mp,eigvec[m + n_act:,:])
        else:
            p = MUL(w,eigvec[m:m + n_act,:])
            hp = MUL(hw,eigvec[m:m + n_act,:])
            mp = MUL(mw,eigvec[m:m + n_act,:])
        
        x = MUL(x,eigvec[0:m,:]) + p
        hx = MUL(hx,eigvec[0:m,:]) + hp
        mx = MUL(mx,eigvec[0:m,:]) + mp

        t_o = owari()
        t_mul = t_mul + t_o - t_h
        
        t_iter_o = owari()
        t_iter = t_iter_o - t_iter_h
        
        # Print
        print(f"Iter={iter_}, residual={NP.linalg.norm(res_nrms):<6.3e}, "
              f"n_act={n_act}, t_iter={t_iter:<6.3f}s, t_fft={t_fft:<6.3f}s, "
              f"t_mul={t_mul:<6.3f}s, t_eigh={t_eigh:<6.3f}s.")
    
    del w,p,hx,hw,hp
        
    lambdas = lambdas.get()
    
    t_tot_o = owari()
    t_tot = t_tot_o - t_tot_h
    print(f"\nA complete procedure of lobpcg is done, {t_tot:<6.3f}s elapsed.")
    
    return lambdas[0:m_conv],x,np.array([iter_,t_tot])

def lobpcg_PCs_mfd(a_fft,b_fft,diels,inv_fft,x0,m_conv,**kwargs):
    
    """
    Usage:
        LOBPCG program for computing bandgap of photonic crystals (PCs).
        Discretization: mimetic finite difference (MFD) method, also known as 'Yee scheme'.
    
    Input:
        a_fft: FFT diagonalization of A.
        b_fft: FFT diagonalization of pnt B'B (upper blocks).
        diels: diagonal matrix of dielectric coefficient.
    
            diels allows two types of inputs:
                diels 1: a tuple contians indices and a coefficient;
                diels 2: an 1D coefficient array of length 3N^3. 
    
        INV: FFT inverse of AA'+pnt B'B+shift.
        x0: initial guess.
        m_conv: number of desired eigenpairs.
        
    **kwargs: 
        shift (default 0) ,tolerance and maximum iterations.
    
    
    Output:
        lambda_0: eigenvalues of penalty scheme.
        lambda_re: recomputing eigenvalues.    
        iter_: iterations and time.
        x: eigenvectors.

    """
    
    """
    LOBPCG: initialization.
    """
    
    t_h = time()
    
    # Defaults.
    opts = {"shift": 0, "tol": 1e-4, "iter_max": 1000}
    opts.update(kwargs)

    # Shift, tolerance, maximum iterations.
    shift,tol_lobpcg,iter_max = opts["shift"],opts["tol"],opts["iter_max"]
    
    # Matrix manipulations.
    NP = arrtype(x0)
    MUL,HM,VM = NP.dot,NP.hstack,NP.vstack
    
    # Timing (cpu/gpu).
    owari = owari_opt(str(NP))
    
    # Local eigensolver.
    if NP == cp:
        GEP = GEP_chol
    else:
        GEP = sc.linalg.eigh
    
    # Dielectric setting.
    Diels = mfd.scalar_prod
    
    nn,m = NP.shape(x0)
    nn = round(nn/3)
    
    # Initialization.
    x = x0
    del x0
    hx = A_fft(Diels(A_fft(x,-a_fft.conj()),diels),a_fft)+H_fft(x,b_fft) + shift * x    
    lambdas,__ = NP.linalg.eigh(MUL(x.T.conj(),hx))    
    p,hp = [],[]
    
    t_o = owari()
    print(f"Time for initialization:{t_o - t_h:<6.3e}s.")
    
    """
    LOBPCG: Main loop.
    """
    
    t_tot_h = time()
    
    for iter_ in range(iter_max):
        t_iter_h = time()
        
        # Residual, convergence and locking.
        t_h = time()
        w = hx - x * lambdas
        
        res_nrms,x_nrms = norms(w),norms(x)

        ind_act = NP.where(res_nrms / x_nrms > tol_lobpcg)[0]
        n_act = len(ind_act)

        if n_act <= m - m_conv:
            break
        
        t_o = owari()
        t_mul = t_o - t_h
        
        # Preconditioning.
        t_h = time()
        w = H_fft(w[:,ind_act],inv_fft)

        hw = A_fft(Diels(A_fft(w,-a_fft.conj()),diels),a_fft)+H_fft(w,b_fft)+shift*w

        t_o = owari()
        t_fft = t_o - t_h
        
        # Rayleigh-Ritz    
        
        t_h = time()
        t11,g11 = mfd.hermitize(MUL(x.T.conj(),hx)),mfd.hermitize(MUL(x.T.conj(),x))
        t12,g12 = MUL(x.T.conj(),hw),MUL(x.T.conj(),w)
        t22,g22 = mfd.hermitize(MUL(w.T.conj(),hw)),mfd.hermitize(MUL(w.T.conj(),w))

        t_o = owari()
        t_mul = t_mul + t_o - t_h
        
        if iter_ > 0:
            t_h = time()
            p = p[:,ind_act]
            hp = hp[:,ind_act]
            t13,g13 = MUL(x.T.conj(),hp),MUL(x.T.conj(),p)
            t23,g23 = MUL(w.T.conj(),hp),MUL(w.T.conj(),p)
            t33,g33 = mfd.hermitize(MUL(p.T.conj(),hp)),mfd.hermitize(MUL(p.T.conj(),p))
            t_o = owari()
            t_mul = t_mul + t_o - t_h
            
            lambdas,eigvec,t_eig = GEP(VM((HM((t11,t12,t13)),\
                                       HM((t12.T.conj(),t22,t23)),\
                                       HM((t13.T.conj(),t23.T.conj(),t33)))),\
                                       VM((HM((g11,g12,g13)),\
                                       HM((g12.T.conj(),g22,g23)),\
                                       HM((g13.T.conj(),g23.T.conj(),g33)))))
        else:
            lambdas,eigvec,t_eigh = GEP(VM((HM((t11,t12)),HM((t12.T.conj(),t22)))),\
                                        VM((HM((g11,g12)),HM((g12.T.conj(),g22)))))
        
        t_h = time()
        
        lambdas = lambdas[0:m]
        eigvec = eigvec[:,0:m]
        if iter_ > 0:
            p=MUL(w,eigvec[m:m+n_act,:])+MUL(p,eigvec[m+n_act:,:])
            hp=MUL(hw,eigvec[m:m+n_act,:])+MUL(hp,eigvec[m+n_act:,:])
        else:
            p=MUL(w,eigvec[m:m+n_act,:])
            hp=MUL(hw,eigvec[m:m+n_act,:])
        x=MUL(x,eigvec[0:m,:])+p
        hx=MUL(hx,eigvec[0:m,:])+hp

        t_o=owari()
        t_mul=t_mul+t_o-t_h
        
        t_iter_o=owari()
        t_iter=t_iter_o-t_iter_h
        
        # Print
        print("Iter=",iter_,", residual=",'%6.3e'%(NP.linalg.norm(res_nrms)),\
              ", n_act=",n_act,", t_iter=",'%6.3f'%t_iter,"s, t_fft=",\
              '%6.3f'%t_fft,"s, t_mul=",'%6.3f'%t_mul,"t_eigh=",'%6.3f'%t_eigh,"s.")
    
    del w,p,hx,hw,hp
    
    # Recomputing.
    t_h=time()
    x_=x[:,0:m_conv]
 
    adax=Diels(A_fft(x_,-a_fft.conj()),diels[0],np.sqrt(diels[1]))
    adax=MUL(adax.T.conj(),adax)
    lambdas_re=(NP.diag(adax)/NP.diag(MUL(x_.T.conj(),x_))).real    
    lambdas,lambdas_re=lambdas.get(),lambdas_re.get()
    
    t_o=owari()
    print("\nTime for postprocessing (recomputing): ",'%6.3f'%(t_o-t_h),"s.")
    
    t_tot_o=owari()
    t_tot=t_tot_o-t_tot_h
    print("\nA complete procedure od eigen_lobpcg_gpu is done, ",'%6.3f'%t_tot,"s elapsed.")
    
    return lambdas[0:m_conv]-shift,lambdas_re,np.array([iter_,t_tot]),x

"""
Generalized eigenvalue problem (GEP).
If on cpu, the scipy.linalg.eigh allows two inputs thus GEP is directly solved.
If on gpu, the scipy.linalg.eigh ONLY allows single argin, which means GEP can't be explicitly solved
           on gpudevice. Now we give two solutions: the 1st is to use '.get()' to drag the arrays 
           back to cpu then scipy.linalg.eigh is available. The 2nd is to transform GEP into SEP via
           Cholesky factorization. 

When the relatively 'small' GEP is in a large scale, the communication costs of gpuarray to cpuarray
might be more expensive than chol. 
"""

def GEP_get(T,G):
    # GEP on gpu 1: cupy to scipy, eigh(), scipy to cupy.
    
    t_h=time()
    T,G=T.get(),G.get() 
    lambdas,eigvec=sc.linalg.eigh(T,G)
    lambdas,eigvec=cp.asarray(lambdas),cp.asarray(eigvec)
    t_o=owari_cuda()

    return lambdas,eigvec,t_o-t_h

def GEP_chol(T,G):
    # GEP on gpu 2: GEP to SEP via cholesky.
    
    t_h=time()
    L=cp.linalg.inv(cp.linalg.cholesky(G))
    T=cp.dot(cp.dot(L,T),L.T.conj())        
    lambdas,eigvec=cp.linalg.eigh(T)
    eigvec=cp.dot(L.T.conj(),eigvec)        
    t_o=owari_cuda()

    return lambdas,eigvec,t_o-t_h

def davidson_sep(a_func,p_func,x0,m_conv,**kwargs):
    
    """
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
    
    """
    LOBPCG: initialization.
    """
    
    t_h = time()
    
    # Defaults.
    opts = default_kwargs_eigen_solver()
    opts.update(kwargs)
    
    # Shift, tolerance, maximum iterations.
    tol,iter_max = opts["tol"],opts["iter_max"]
    
    # Matrix manipulations.
    NP = arrtype(x0)
    MUL,HM,VM = NP.dot,NP.hstack,NP.vstack
    
    # Timing (cpu/gpu).
    owari = owari_opt(str(NP))
    
    # Local eigensolver.
    if NP == cp:
        GEP = GEP_chol
    else:
        GEP = sc.linalg.eigh

    x,m = x0,x0.shape[1]
    del x0
    x,__ = NP.linalg.qr(x)
    hx = a_func(x)
    
    hs = MUL(x.T.conj(),hx)
    lambdas,v = NP.linalg.eigh(hs) 
    r = hx - x * lambdas
    r_nrms = norms(r)
    
    s = NP.identity(m)
    q,hq,u = x,hx,v
    
    nsub_max,nsub = 3*m,m
    
    t_o = owari()
    print(f"Time for initialization:{t_o - t_h:<6.3f}s.")
    
    """
    Davidson: Main loop.
    """
    
    t_tot_h = time()
    
    for iter_ in range(0,iter_max):
        t_iter_h = time()
        
        # Residual, convergence and locking.
        t_h = time()

        ind_act = NP.where(r_nrms>tol)[0]
        n_act = len(ind_act)
        nsub = nsub + n_act

        if n_act <= m - m_conv:
            break
        elif nsub > nsub_max:
            # shrink subspace.
            q,hq = MUL(q,u),MUL(hq,u)
            s,hs = NP.identity(m),MUL(u.T.conj(),MUL(hs,u))
            nsub = m
        
        t_o = time()
        t_mul = t_o - t_h
        
        # Preconditioning.
        x = p_func(r[:,ind_act])
        x /= norms(x)
        
        # Local.
        g2 = MUL(x.T.conj(),q)
        g3 = MUL(x.T.conj(),x)
        
        s = VM((HM((s,g2.T.conj())),HM((g2,g3))))
        hx = a_func(x)
        
        t2 = MUL(x.T.conj(),hq)
        t3 = MUL(x.T.conj(),hx)
        
        hs = VM((HM((hs,t2.T.conj())),HM((t2,t3))))
        lambdas,v,t_eigh = GEP(hs,s)
        
        lambdas = lambdas[:m]
        u = v[:,:m]
        
        q = HM((q,x))
        hq = HM((hq,hx))
        r = MUL(hq,u) - MUL(q,u) * lambdas
        r_nrms = norms(r)
        
        t_iter_o = owari()
        t_iter = t_iter_o - t_iter_h
        
        # Print
        
        print(f"Iter={iter_}, residual={norm(r_nrms):<6.3e}, "
              f"n_act={n_act}, t_iter={t_iter:<6.3f}s, t_mul={t_mul:<6.3f}s, "
              f"t_eigh={t_eigh:<6.3f}s.")
            
    x = MUL(q,u)    
    
    t_tot_o = owari()
    t_tot = t_tot_o - t_tot_h
    print(f"Davidson for SEP is completed, {t_tot:>6.3f}s elapsed.")

    return lambdas,x,[iter_,t_tot]

def jd_sep(a_func,p_func,x0,m_conv,**kwargs):
    
    """
    Usage:
        A Jacobi-Davidson method for simple eigenvalue problems (SEP).
        Model: Ax=\lambda x.
    
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
    
    """
    LOBPCG: initialization.
    """
    
    t_h = time()
    
    # Defaults.
    opts = default_kwargs_eigen_solver()
    opts.update(kwargs)
    
    # Shift, tolerance, maximum iterations.
    tol,iter_max = opts["tol"],opts["iter_max"]
    
    # Matrix manipulations.
    NP = arrtype(x0)
    MUL,HM,VM = NP.dot,NP.hstack,NP.vstack
    
    # Timing (cpu/gpu).
    owari = owari_opt(str(NP))
    
    # Local eigensolver.
    if NP == cp:
        GEP = GEP_chol
    else:
        GEP = sc.linalg.eigh

    x,m = x0,x0.shape[1]
    del x0
    x,__ = NP.linalg.qr(x)
    hx = a_func(x)
    
    hs = MUL(x.T.conj(),hx)
    lambdas,v = NP.linalg.eigh(hs) 
    r = hx - x * lambdas
    r_nrms = norms(r)
    
    s = NP.identity(m)
    q,hq,u = x,hx,v
    
    nsub_max,nsub = 3*m,m
    
    t_o = owari()
    print(f"Time for initialization:{t_o - t_h:<6.3f}s.")
    
    """
    Jacobi-Davidson: Main loop.
    """
    
    t_tot_h = time()
    
    for iter_ in range(iter_max):
        t_iter_h = time()
        
        # Residual, convergence and locking.
        t_h = time()

        ind_act = NP.where(r_nrms>tol)[0]
        n_act = len(ind_act)
        nsub += n_act

        if n_act <= m - m_conv:
            break
        elif nsub > nsub_max:
            # shrink subspace.
            q,hq = MUL(q,u),MUL(hq,u)
            s,hs = NP.identity(m),MUL(u.T.conj(),MUL(hs,u))
            nsub = m
        
        t_o = time()
        t_mul = t_o - t_h
        
        # Preconditioning.
        x = x[:,ind_act]
        r = p_func(r[:,ind_act])
        px = p_func(x)
        
        x = px * dots(x,r) / dots(x,px) - r
        x /= norms(x)
        
        # Local.
        g2 = MUL(x.T.conj(),q)
        g3 = MUL(x.T.conj(),x)
        
        s = VM((HM((s,g2.T.conj())),HM((g2,g3))))
        hx = a_func(x)
        
        t2 = MUL(x.T.conj(),hq)
        t3 = MUL(x.T.conj(),hx)
        
        hs = VM((HM((hs,t2.T.conj())),HM((t2,t3))))
        lambdas,v,t_eigh = GEP(hs,s)
        
        lambdas = lambdas[:m]
        u = v[:,:m]
        
        q = HM((q,x))
        hq = HM((hq,hx))
        r = MUL(hq,u) - MUL(q,u) * lambdas
        r_nrms = norms(r)
        
        t_iter_o = owari()
        t_iter = t_iter_o - t_iter_h
        
        # Print
        
        print(f"Iter={iter_}, residual={norm(r_nrms):<6.3e}, "
              f"n_act={n_act}, t_iter={t_iter:<6.3f}s, t_mul={t_mul:<6.3f}, "
              f"t_eigh={t_eigh:<6.3f}s.")
            
    x=MUL(q,u)    
    
    t_tot_o = owari()
    t_tot = t_tot_o - t_tot_h
    print(f"Davidson for SEP is completed, {t_tot:>6.3f}s elapsed.")

    return lambdas,x,[iter_,t_tot]

