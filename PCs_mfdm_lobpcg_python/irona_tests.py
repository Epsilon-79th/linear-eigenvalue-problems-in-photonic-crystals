# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:02:27 2024

@author: 11034
"""

"""

色んなテスト
various tests.


"""

import time,json
import numpy as np
import scipy as sc
import cupy as cp
import cupyx.scipy as cpx
import discretization as mfd
import dielectric as diel

from discretization import A_fft,H_fft
from numpy.random import rand
from numpy import pi


# json type file i/o.
def json_test():

    arr={'x':np.random.rand(10,3).tolist()}
    arr['y']="String."
    arr['z']=pi

    with open('test.json','w') as f:
        json.dump(arr,f,indent=4)

    with open('diel_info.json','r') as f:
        arr2=json.load(f)
    
    print(arr2["CT_bcc"])
    return
    
def fft_test(N,m,is_complex=True):
    n=N**3
    
    if is_complex:
        X=rand(3*n,m)+1j*rand(3*n,m)
        data_type="complex"
    else:
        X=rand(3*n,m)
        data_type="real"
    
    # numpy
    
    t_h=time.time()
    X=X.reshape(N,N,N,3*m,order='F')
    X=np.fft.fftn(X,axes=(0,1,2))
    X=np.fft.ifftn(X,axes=(0,1,2))
    X=X.reshape(3*n,m,order='F')
    t_o=time.time()
    
    t_np=t_o-t_h
    
    # scipy
    if not is_complex:
        X=np.real(X)
        
    
    t_h=time.time()
    X=X.reshape(N,N,N,3*m,order='F')
    X=sc.fft.fftn(X,axes=(0,1,2))
    X=sc.fft.ifftn(X,axes=(0,1,2))
    X=X.reshape(3*n,m,order='F')
    t_o=time.time()
    
    t_sc=t_o-t_h
    
    # cupy
    X=cp.asarray(X)
    if not is_complex:
        X=np.real(X)
    
    
    t_h=time.time()
    X=X.reshape(N,N,N,3*m,order='F')
    X=cp.fft.fftn(X,axes=(0,1,2))
    X=cp.fft.ifftn(X,axes=(0,1,2))
    X=X.reshape(3*n,m,order='F')
    t_o=time.time()
    
    t_cp=t_o-t_h
    
    # cupyx
    if not is_complex:
        X=np.real(X)
    
    
    t_h=time.time()
    X=X.reshape(N,N,N,3*m,order='F')
    X=cpx.fft.fftn(X,axes=(0,1,2))
    X=cpx.fft.ifftn(X,axes=(0,1,2))
    X=X.reshape(3*n,m,order='F')
    t_o=time.time()
    
    t_cpx=t_o-t_h

    
    print(3*n,"*",m,"datatype=",data_type,".\nFFT: np_time=",'%6.3f'%t_np,\
          "s, sc_time=",'%6.3f'%t_sc,"s, cp_time=",'%6.3f'%t_cp,"s, cpx_time=",'%6.3e'%t_cpx,\
              "s.\nAcceleration ratio: sc/cpx=x",'%6.3f'%(t_sc/t_cpx))

def norm_test(N,m,is_complex=True):
    
    n=N**3
    
    if is_complex:
        X=rand(3*n,m)+1j*rand(3*n,m)
        data_type="complex"
    else:
        X=rand(3*n,m)
        data_type="real"
    
    # numpy-@

    t_h=time.time()
    _=np.sqrt(np.real(np.trace(X.T.conj()@X)))
    t_o=time.time()

    t_np=t_o-t_h
    
    # scipy-dot
        
    t_h=time.time()
    _=sc.sqrt(sc.real(sc.trace(sc.dot(X.T.conj(),X))))
    t_o=time.time()

    t_sc=t_o-t_h
    
    # cupy-@
    X=cp.asarray(X)
    
    t_h=time.time()
    _=cp.sqrt(cp.real(cp.trace(cp.matmul(X.T.conj(),X))))
    t_o=time.time()

    t_cp=t_o-t_h

    # cupy-dot
    X=cp.asarray(X)
    
    t_h=time.time()
    _=cp.sqrt(cp.real(cp.trace(cp.dot(X.T.conj(),X))))
    t_o=time.time()

    t_cpx=t_o-t_h
    
    print(3*n,"*",m,"datatype=",data_type,".\nNorm: np-@=",'%6.3f'%t_np,\
          "s, sc-dot=",'%6.3f'%t_sc,"s, cp-@=",'%6.3f'%t_cp,"s, cp-dot=",\
          '%6.3e'%t_cpx,"s.\nAcceleration ratio: cp-@/cp-dot=x",'%6.3f'%(t_cp/t_cpx))
    
def matmul_test(N,m,is_complex=True):
    
    n=N**3
    
    if is_complex:
        X=rand(3*n,m)+1j*rand(3*n,m)
        Y=rand(3*n,m)+1j*rand(3*n,m)
        data_type="complex"
    else:
        X=rand(3*n,m)
        Y=rand(3*n,m)
        data_type="real"
    
    # numpy-@

    t_h=time.time()
    _=X.T.conj()@Y
    t_o=time.time()

    t_np=t_o-t_h
    
    # scipy-dot
        
    t_h=time.time()
    _=sc.dot(X.T.conj(),Y)
    t_o=time.time()

    t_sc=t_o-t_h
    
    # cupy-@
    X=cp.asarray(X)
    Y=cp.asarray(Y)
    
    t_h=time.time()
    _=X.T.conj()@Y
    t_o=time.time()

    t_cp=t_o-t_h

    # cupy-dot
    X=cp.asarray(X)
    
    t_h=time.time()
    _=cp.dot(X.T.conj(),Y)
    t_o=time.time()

    t_cpx=t_o-t_h
    
    print(3*n,"*",m,"datatype=",data_type,".\nNorm: np-@=",'%6.3f'%t_np,\
          "s, sc-dot=",'%6.3f'%t_sc,"s, cp-@=",'%6.3f'%t_cp,"s, cp-dot=",\
          '%6.3e'%t_cpx,"s.\nAcceleration ratio: cp-@/cp-dot=x",'%6.3f'%(t_cp/t_cpx))

def mfd_blocks_test(N,k):
    
    a=2*pi*N**(2/3)
    Alphas=np.array([[pi,pi,0],[pi,0,0],[pi,pi,pi]])/a
    pnt=2*N
    CT=np.identity(3)
    
    print("Size:",3*N**3,".\n")
    
    for i in range(np.size(Alphas,0)):
        alpha=Alphas[i,:]
        # Numpy
        t_h=time.time()
        D,Di=mfd.mfd_fft_blocks(a, N, k, CT,"np")
        _=mfd.full_fft_blocks(D, Di, alpha, pnt,"np")
        t_o=time.time()
        t_np=t_o-t_h
        
        # Scipy
        t_h=time.time()
        D,Di=mfd.mfd_fft_blocks(a, N, k, CT,"sc")
        _=mfd.full_fft_blocks(D, Di, alpha, pnt,"sc")
        t_o=time.time()
        t_sc=t_o-t_h
        
        # Cupy
        t_h=time.time()
        D,Di=mfd.mfd_fft_blocks(a, N, k, CT,"cp")
        _=mfd.full_fft_blocks(D, Di, alpha, pnt,"cp")
        t_o=time.time()
        t_cp=t_o-t_h
        
        # Print
        print("Mat: t_np=",'%6.3f'%t_np,\
              "s, t_sc=",'%6.3f'%t_sc,"s, t_cp=",'%6.3f'%t_cp,"s.")
            
def element_wise_mul_test(N,m):
    
    n=N**3
    co=rand(3*n)
    x=rand(3*n,m)
    
    # numpy

    t_h=time.time()
    _=(x.T*co).T
    t_o=time.time()

    t_np=t_o-t_h
    
    # cupy
    x=cp.asarray(x)
    co=cp.asarray(co)
    
    t_h=time.time()
    _=(x.T*co).T
    t_o=time.time()

    t_cp=t_o-t_h
    
    print("Size: ",3*n,"*",m,".\nt_np=",'%6.3f'%t_np,\
          "s, t_cp=",'%6.3f'%t_cp,"s, ratio=x",\
          '%6.3f'%(t_np/t_cp),".")
    
def cufft_test(N,m,pack_name="np"):
    
    NP=eval(pack_name)
    n=N**3
    
    x=eval(pack_name[0:2]).random.rand(3*n,m)
    
    x=x.reshape(N,N,N,3*m,order="F")
    t_h=time.time()
    
    _=NP.fft.fftn(x,axes=(0,1,2))
    _=NP.fft.ifftn(x,axes=(0,1,2))
    
    t_o=time.time()
    x=x.reshape(3*n,m,order="F")
    
    print("Size:",3*n,m,",time=",'%6.3f'%(t_o-t_h))
    
    
def prec_mul_test(N,m):
    
    n=N**3
    x=cp.random.rand(3*n,m)
    
    options_mat={"a":2*pi*N**(2/3),\
                 "k":1,"eps":13,\
                 "alpha":np.array([pi,pi,pi]),\
                 "d_flag_name":"sc_curv"}
        
    t_h=time.time()
    D,B,M0,INV,relax_opt=mfd.mfd_final_mat(N,options_mat)
    t_o=time.time()
    print("Matrix blocks done, ",'%6.3f'%(t_o-t_h),"s elapsed.")  
    
    D,B,M0,INV=cp.asarray(D),cp.asarray(B),cp.asarray(M0),cp.asarray(INV)
    
    Hx=H_fft(A_fft(mfd.x_mul_y(M0,A_fft(x,-D.conj(),"cpx")),D,"cpx")\
             +H_fft(x,B,"cpx"),INV,"cpx")
        
    print(cp.diag(cp.dot(x.T.conj(),Hx)))
    
    
def mfd_pnt_shift_test(N,omega,pack_name="cpx"):
     
    NP=eval(pack_name[0:2])
    a=1
    n=N**3
    alpha=np.array([pi,pi,0])/a
    pnt=2*N
    x=NP.random.rand(3*n,1)
    x_norm=NP.linalg.norm(x)
    x=x/x_norm
     
    k=1
    d_flag_name="sc_curv"
    
    d_func= lambda w: 5.8+18.6*(10.45**2-1.2*(w/a)**2-2*(w/a)**3+3.64*(w/a)**4)\
                        /(9.89**2-2.6*(w/a)-1.2*(w/a)**2-(w/a)**3)

    shift=omega**2
    eps=d_func(omega)
    
    # Diagonal matrix of dielectric coefficients    
    t_diel_h=time.time()
    M0=NP.ones(3*n)
    ind_d,CT,__=diel.dielectric_save_and_load(N,d_flag_name)
    M0[ind_d]=1/eps
    t_diel_o=time.time()
    print("Dielectric matrix done, ",'%6.3f'%(t_diel_o-t_diel_h),"s elapsed.")
    
    D,Di=mfd.mfd_fft_blocks(a, N, k, CT)
    D,B,INV=mfd.full_fft_blocks(D,Di,alpha,pnt,-shift)
    
    if pack_name[0:2]=="cp":
        D,B,M0,INV=cp.asarray(D),cp.asarray(B),\
                   cp.asarray(M0),cp.asarray(INV)
    
    Hx=A_fft(mfd.x_mul_y(M0,A_fft(x,-D.conj(),pack_name)),D,pack_name)\
        +H_fft(x,B,pack_name)-shift*x
    PHx=H_fft(Hx,INV,pack_name)
    
    d11=(D[0:n]*D[0:n].conj()).real
    d22=(D[n:2*n]*D[n:2*n].conj()).real
    d33=(D[2*n:]*D[2*n:].conj()).real
    d12=D[0:n].conj()*D[n:2*n]
    d13=D[0:n].conj()*D[2*n:]
    d23=D[n:2*n].conj()*D[2*n:]
    INV0=mfd.inverse_3_times_3_block(\
       pnt*d11+d22+d33,d11+pnt*d22+d33,\
       d11+d22+pnt*d33,(pnt-1)*d12,(pnt-1)*d13,(pnt-1)*d23)
    
    P2Hx=H_fft(Hx,INV0,pack_name)
    x1=A_fft(A_fft(x,-D.conj(),pack_name),D,pack_name)\
        +H_fft(x,B,pack_name)-shift*x
    x1=H_fft(x1,INV,pack_name)
        
    print("|Hx|=",'%6.3f'%(NP.linalg.norm(Hx)))
    print("|PHx|=",'%6.3f'%(NP.linalg.norm(PHx)))
    print("|P2Hx|=",'%6.3f'%(NP.linalg.norm(P2Hx)))
    print("|x-x1|=",'%6.3e'%(NP.linalg.norm(x-x1)))
    