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
from my_norm import norms
from discretization import A_fft,H_fft
from gpu_opts import arrtype,owari_opt,owari_cuda


def PCs_mfd_lobpcg(a_fft,b_fft,diels,inv_fft,x0,m_conv,**kwargs):
    
    """
    LOBPCG program for computing bandgap of photonic crystals (PCs).
    Discretization: mimetic finite difference (MFD) method, also known as 'Yee scheme'.
    """
    
    """
    Input.
    """
    # a_fft: FFT diagonalization of A.
    # b_fft: FFT diagonalization of pnt B'B (upper blocks).
    # diels: diagonal matrix of dielectric coefficient.
    
    # diels allows two types of inputs:
    # diels 1: a tuple contians indices and a coefficient;
    # diels 2: an 1D coefficient array of length 3N^3. 
    
    # INV: FFT inverse of AA'+pnt B'B+shift.
    # x0: initial guess.
    # m_conv: number of desired eigenpairs.
    # opts: a dict contains shift (default 0) ,name of package, tolerance
    #       and maximum iterations.
    
    """
    Output.
    """ 
    # lambda_0: eigenvalues of penalty scheme.
    # lambda_re: recomputing eigenvalues.    
    # iter_: iterations and time.
    # x: eigenvectors.

    """
    LOBPCG: initialization.
    """
    
    t_h=time()
    
    # Defaults.
    opts = {"shift": 0, "tol": 1e-4, "iter_max": 1000}
    opts.update(kwargs)

    # Shift, tolerance, maximum iterations.
    shift,tol_lobpcg,iter_max=opts["shift"],opts["tol"],opts["iter_max"]
    
    # Matrix manipulations.
    NP=arrtype(x0)
    MUL,HM,VM=NP.dot,NP.hstack,NP.vstack
    
    # Timing (cpu/gpu).
    owari=owari_opt(str(NP))
    
    # Local eigensolver.
    if NP==cp:
        GEP=GEP_chol
    else:
        GEP=sc.linalg.eigh
    
    # Dielectric setting.
    Diels=mfd.scalar_prod
    
    nn,m=NP.shape(x0)
    nn=round(nn/3)
    
    # Initialization.
    #Q=NP.zeros((3*n,3*m),dtype=complex)         
    x=x0
    del x0
    hx=A_fft(Diels(A_fft(x,-a_fft.conj()),diels),a_fft)+H_fft(x,b_fft)+shift*x    
    lambdas,__=NP.linalg.eigh(MUL(x.T.conj(),hx))    
    p,hp=[],[]
    
    t_o=owari()
    print("Time for initialization:",'%6.3f'%(t_o-t_h),"s.")
    
    """
    LOBPCG: Main loop.
    """
    
    t_h=time()
    
    for iter_ in range(iter_max):
        t_iter_h_flag=time()
        
        # Residual, convergence and locking.
        t_h_flag=time()
        w=hx-x*lambdas
        
        res_nrms,x_nrms=norms(w),norms(x)

        ind_act=NP.where(res_nrms/x_nrms>tol_lobpcg)[0]
        n_act=len(ind_act)

        if n_act<=m-m_conv:
            break
        
        t_o_flag=owari()
        t_mul=t_o_flag-t_h_flag
        
        # Preconditioning.
        t_h_flag=time()
        w=H_fft(w[:,ind_act],inv_fft)

        hw=A_fft(Diels(A_fft(w,-a_fft.conj()),diels),a_fft)+H_fft(w,b_fft)+shift*w

        t_o_flag=owari()
        t_fft=t_o_flag-t_h_flag
        
        # Rayleigh-Ritz    
        
        t_h_flag=time()
        t11,g11=MUL(x.T.conj(),hx),MUL(x.T.conj(),x)
        t12,g12=MUL(x.T.conj(),hw),MUL(x.T.conj(),w)
        t22,g22=MUL(w.T.conj(),hw),MUL(w.T.conj(),w)

        t_o_flag=owari()
        t_mul=t_mul+t_o_flag-t_h_flag
        
        if iter_>0:
            t_h_flag=time()
            p=p[:,ind_act]
            hp=hp[:,ind_act]
            t13,g13=MUL(x.T.conj(),hp),MUL(x.T.conj(),p)
            t23,g23=MUL(w.T.conj(),hp),MUL(w.T.conj(),p)
            t33,g33=MUL(p.T.conj(),hp),MUL(p.T.conj(),p)
            t_o_flag=owari()
            t_mul=t_mul+t_o_flag-t_h_flag
            
            lambdas,eigvec,t_eig=GEP(VM((HM((t11,t12,t13)),\
                                     HM((t12.T.conj(),t22,t23)),\
                                     HM((t13.T.conj(),t23.T.conj(),t33)))),\
                                     VM((HM((g11,g12,g13)),\
                                     HM((g12.T.conj(),g22,g23)),\
                                     HM((g13.T.conj(),g23.T.conj(),g33)))))
        else:
            lambdas,eigvec,t_eig=GEP(VM((HM((t11,t12)),HM((t12.T.conj(),t22)))),\
                                     VM((HM((g11,g12)),HM((g12.T.conj(),g22)))))
        
        t_h_flag=time()
        
        lambdas=lambdas[0:m]
        eigvec=eigvec[:,0:m]
        if iter_>0:
            p=MUL(w,eigvec[m:m+n_act,:])+MUL(p,eigvec[m+n_act:,:])
            hp=MUL(hw,eigvec[m:m+n_act,:])+MUL(hp,eigvec[m+n_act:,:])
        else:
            p=MUL(w,eigvec[m:m+n_act,:])
            hp=MUL(hw,eigvec[m:m+n_act,:])
        x=MUL(x,eigvec[0:m,:])+p
        hx=MUL(hx,eigvec[0:m,:])+hp

        t_o_flag=owari()
        t_mul=t_mul+t_o_flag-t_h_flag
        
        t_iter_o_flag=owari()
        t_iter=t_iter_o_flag-t_iter_h_flag
        
        # Print
        print("Iter=",iter_,", residual=",'%6.3e'%(NP.linalg.norm(res_nrms)),\
              ", n_act=",n_act,", t_iter=",'%6.3f'%t_iter,"s, t_fft=",\
              '%6.3f'%t_fft,"s, t_mul=",'%6.3f'%t_mul,"t_eigh=",'%6.3f'%t_eig,"s.")
    
    del w,p,hx,hw,hp
    
    # Recomputing.
    t_h_flag=time()
    x_=x[:,0:m_conv]
 
    adax=Diels(A_fft(x_,-a_fft.conj()),diels[0],np.sqrt(diels[1]))
    adax=MUL(adax.T.conj(),adax)
    lambdas_re=(NP.diag(adax)/NP.diag(MUL(x_.T.conj(),x_))).real    
    lambdas,lambdas_re=lambdas.get(),lambdas_re.get()
    
    t_o_flag=owari()
    print("\nTime for postprocessing (recomputing): ",'%6.3f'%(t_o_flag-t_h_flag),"s.")
    
    t_o=owari()
    t_tot=t_o-t_h
    print("\nA complete procedure od eigen_lobpcg_gpu is done, ",'%6.3f'%t_tot,"s elapsed.")
    
    return lambdas[0:m_conv]-shift,lambdas_re,np.array([iter_,t_tot]),x


# GEP on gpu 1: cupy to scipy, eigh(), scipy to cupy.
def GEP_local(T,G):
    
    t_h=time()
    T,G=T.get(),G.get() 
    lambdas,eigvec=sc.linalg.eigh(T,G)
    lambdas,eigvec=cp.asarray(lambdas),cp.asarray(eigvec)
    t_o=owari_cuda()

    return lambdas,eigvec,t_o-t_h

# GEP on gpu 2: GEP to SEP via cholesky.
def GEP_chol(T,G):
    
    t_h=time()
    L=cp.linalg.inv(cp.linalg.cholesky(G))
    T=cp.dot(cp.dot(L,T),L.T.conj())        
    lambdas,eigvec=cp.linalg.eigh(T)
    eigvec=cp.dot(L.T.conj(),eigvec)        
    t_o=owari_cuda()

    return lambdas,eigvec,t_o-t_h