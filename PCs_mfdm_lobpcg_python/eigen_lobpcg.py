# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 11:42:05 2024

@author: Epsilon-79th.
"""

import numpy as np
import scipy as sc
import cupy as cp

import discretization as mfd
from numpy import pi
from discretization import norms_gpu
from discretization import A_fft
from discretization import H_fft
    
#from scipy.linalg import norm

#from numpy.random import rand
#from numpy import pi

from time import time

# Local GEP 1: cupy to scipy, eigh(), scipy to cupy.
def GEP(T,G,pack_name="np"):
    
    if pack_name[0:2]!="cp":
        lambdas,U=sc.linalg.eigh(T,G)
    else:
        t_h=time()
        T,G=T.get(),G.get()
        t_o=time()
        t1=t_o-t_h
        
        t_h=time()
        lambdas,U=sc.linalg.eigh(T,G)
        t_o=time()
        t2=t_o-t_h
        
        t_h=time()
        lambdas,U=cp.asarray(lambdas),cp.asarray(U)
        t_o=time()
        t1=t1+t_o-t_h
    t_o=time()
    return lambdas,U,t1,t2

# Local GEP 2: GEP to SEP via cholesky.
def GEP_chol(T,G,pack_name="np"):
    
    if pack_name[0:2]!="cp":
        lambdas,U=sc.linalg.eigh(T,G)
    else:
        t_h=time()
        L=cp.linalg.inv(cp.linalg.cholesky(G))
        T=cp.dot(cp.dot(L,T),L.T.conj())
        t_o=time()
        t1=t_o-t_h
        
        t_h=time()
        lambdas,U=cp.linalg.eigh(T)
        t_o=time()
        t2=t_o-t_h
        
        t_h=time()
        U=cp.dot(L.T.conj(),U)
        t_o=time()
        t1=t1+t_o-t_h
    t_o=time()
    return lambdas,U,t1,t2


def PCs_mfd_lobpcg(A,B,Ms,INV,X0,m_conv,\
                   options={"shift":0,"pack_name":"np","tol":1e-5,\
                            "iter_max":1000}):
    
    """
    LOBPCG program, linear algebra package: numpy, scipy, cupy/cupyx.
    """
    
    """
    Input.
    """
    # A: FFT diagonalization of A.
    # B: FFT diagonalization of pnt B'B (upper blocks).
    # Ms: diagonal matrix of dielectric coefficient.
    
    # Ms allows two types of inputs:
    # Ms 1: a tuple contians indices and a coefficient;
    # Ms 2: an 1D coefficient array of length 3N^3. 
    
    # INV: FFT inverse of AA'+pnt B'B+shift.
    # X0: initial guess.
    # m_conv: number of desired eigenpairs.
    # options: a dict contains shift (default 0) ,name of package, tolerance
    #          and maximum iterations.
    
    """
    Output.
    """ 
    # lambda_0: eigenvalues of penalty scheme.
    # lambda_re: recomputing eigenvalues.    
    # iter_: iterations and time.
    # X: eigenvectors.

    """
    LOBPCG: initialization.
    """
    
    t_h=time()
    
    # Options.
    shift=options["shift"]
    pack_name=options["pack_name"]
    
    # Matrix manipulations.
    NP=eval(pack_name[0:2])
    MUL=eval(pack_name[0:2]+".dot")
    HM=eval(pack_name[0:2]+".hstack")
    VM=eval(pack_name[0:2]+".vstack")
    
    # Local eigensolver.
    GEP_local=GEP_chol
    
    # Dielectric setting.
    if type(Ms)==tuple:
        diels=mfd.scalar_prod
    else:
        diels=mfd.x_mul_y
        
    # Tolerance.
    tol_lobpcg=options["tol"]

    # Maximum lobpcg iterations.
    iter_max=options["iter_max"]
    
    n,m=NP.shape(X0)
    
    n=round(n/3)
    #N=round(n**(1/3))
    
    if pack_name[0:2]=="cp":
        X,A,B,INV=cp.asarray(X0),cp.asarray(A),cp.asarray(B),\
                  cp.asarray(INV)        
    else:
        X=X0
    
    del X0
    HX=A_fft(diels(A_fft(X,-A.conj(),pack_name),Ms),A,pack_name)\
       +H_fft(X,B,pack_name)+shift*X
    
    lambdas,__=NP.linalg.eigh(MUL(X.T.conj(),HX)) 
    
    P=[]
    HP=[]
    
    t_o=time()
    print("Time for initialization:",'%6.3f'%(t_o-t_h),"s.")
    
    """
    LOBPCG: Main loop.
    """
    
    t_h=time()
    
    for iter_ in range(iter_max):
        t_iter_h_flag=time()
        
        # Residual, convergence and locking.
        t_h_flag=time()
        W=HX-X*lambdas
        
        R_nrms=norms_gpu(W,pack_name)
        X_nrms=norms_gpu(X,pack_name)

        ind_act=NP.where(R_nrms/X_nrms>tol_lobpcg)[0]
        n_act=len(ind_act)

        if n_act<=m-m_conv:
            break

        t_o_flag=time()
        t_mul=t_o_flag-t_h_flag
        
        # Preconditioning.
        t_h_flag=time()
        W=H_fft(W[:,ind_act],INV,pack_name)

        HW=A_fft(diels(A_fft(W,-A.conj(),pack_name),Ms),A,pack_name)\
           +H_fft(W,B,pack_name)+shift*W

        t_o_flag=time()
        t_fft=t_o_flag-t_h_flag
        
        # Rayleigh-Ritz      
        
        t_h_flag=time()
        t11,g11=MUL(X.T.conj(),HX),MUL(X.T.conj(),X)
        t12,g12=MUL(X.T.conj(),HW),MUL(X.T.conj(),W)
        t22,g22=MUL(W.T.conj(),HW),MUL(W.T.conj(),W)
        t_o_flag=time()
        t_mul=t_mul+t_o_flag-t_h_flag
        
        if iter_>0:
            t_h_flag=time()
            P=P[:,ind_act]
            HP=HP[:,ind_act]
            t13,g13=MUL(X.T.conj(),HP),MUL(X.T.conj(),P)
            t23,g23=MUL(W.T.conj(),HP),MUL(W.T.conj(),P)
            t33,g33=MUL(P.T.conj(),HP),MUL(P.T.conj(),P)
            t_o_flag=time()
            t_mul=t_mul+t_o_flag-t_h_flag
            
            lambdas,U,t_eig1,t_eig2=GEP_local(VM((HM((t11,t12,t13)),\
                                    HM((t12.T.conj(),t22,t23)),\
                                    HM((t13.T.conj(),t23.T.conj(),t33)))),\
                                VM((HM((g11,g12,g13)),\
                                    HM((g12.T.conj(),g22,g23)),\
                                    HM((g13.T.conj(),g23.T.conj(),g33)))),pack_name)
        else:
            lambdas,U,t_eig1,t_eig2=GEP_local(VM((HM((t11,t12)),HM((t12.T.conj(),t22)))),\
                                VM((HM((g11,g12)),HM((g12.T.conj(),g22)))),pack_name)
        
        t_h_flag=time()
        
        lambdas=lambdas[0:m]
        U=U[:,0:m]
        if iter_>0:
            P=MUL(W,U[m:m+n_act,:])+MUL(P,U[m+n_act:,:])
            HP=MUL(HW,U[m:m+n_act,:])+MUL(HP,U[m+n_act:,:])
        else:
            P=MUL(W,U[m:m+n_act,:])
            HP=MUL(HW,U[m:m+n_act,:])
        X=MUL(X,U[0:m,:])+P
        HX=MUL(HX,U[0:m,:])+HP
        t_o_flag=time()
        t_mul=t_mul+t_o_flag-t_h_flag
        
        t_iter_o_flag=time()
        t_iter=t_iter_o_flag-t_iter_h_flag
        
        # Print
        print("Iter=",iter_,", residual=",'%6.3e'%(NP.linalg.norm(R_nrms)),\
              ", n_act=",n_act,", t_iter=",'%6.3f'%t_iter,"s, t_fft=",\
              '%6.3f'%t_fft,"s, t_mul=",'%6.3f'%t_mul,"t_eigh=",'%6.3f'%t_eig1,\
              '%6.3f'%t_eig2,")s.")
    
    
    del W,P,HX,HW,HP
    # Recomputing.
    X_=X[:,0:m_conv]
    
    if type(Ms)==tuple:
        Ms=(Ms[0],NP.sqrt(Ms[1]))
    else:
        Ms=np.sqrt(Ms)    
    AMAX=diels(A_fft(X_,-A.conj(),pack_name),Ms)
    AMAX=MUL(AMAX.T.conj(),AMAX)
    lambdas_re=(NP.diag(AMAX)/NP.diag(MUL(X_.T.conj(),X_))).real    
    
    t_o=time()
    print('%6.3f'%(t_o-t_h),"s elapsed.")
    
    return lambdas[0:m_conv]-shift,lambdas_re,iter_,X



def simple_rim_sym_test(A,hanni,eps):
    
    A=(A+A.T.conj())/2
    
    N=len(A)
    m=50
    
    CNT=[(hanni[0]+hanni[1])/2]
    R0=(hanni[1]-hanni[0])/2
    d0=0.2
    
    while R0>eps:
        CNT_new=[]
        for i in range(len(CNT)):
            C0=CNT[i]
            f=np.random.rand(N)
            f=f/np.linalg.norm(f)
            Pf=np.zeros(N)
            for j in range(m):
                theta=np.exp(1j*2*j*pi/m)
                Pf=Pf+np.linalg.solve(A-(C0+R0*theta)*np.identity(N),f)*theta
            Pf=Pf*R0/(2*pi)
            Pf_nrm=np.linalg.norm(Pf)
            print("Between [",'%6.3f'%(C0-R0),",",'%6.3f'%(C0+R0),"], |Pf|=",\
                  '%6.3f'%Pf_nrm,".")
            if Pf_nrm>d0:
                CNT_new.append(C0-R0/2)
                CNT_new.append(C0+R0/2)
        
        CNT=CNT_new
        print("\nR0=",'%6.3f'%(R0)," already seached.\n")
        R0=R0/2

    return CNT