# -*- coding: utf-8 -*-
#
# Created on 2024-03-14 (Thursday) at 21:07:27
#
# Author: Epsilon-79th
#
# Usage: Procedures in Mimetic Finite Difference Discretization.
#

# Packages.

import numpy as np
import scipy as sc
import cupy as cp
from cupyx.scipy.fft import fftn,ifftn

from numpy import pi
from gpu_opts import arrtype
import dielectric as diel

# FFT options. (cpu/gpu)
#fft3d_cpu=lambda x: np.fft.fftn(x,axes=(0,1,2),out=x)
#ifft3d_cpu=lambda x: np.fft.ifftn(x,axes=(0,1,2),out=x)

fft3d_cpu=lambda x: sc.fft.fftn(x,axes=(0,1,2))
ifft3d_cpu=lambda x: sc.fft.ifftn(x,axes=(0,1,2))

#fft3d_gpu=lambda x: cp.fft.fftn(x,axes=(0,1,2))
#ifft3d_gpu=lambda x: cp.fft.ifftn(x,axes=(0,1,2))

fft3d_gpu=lambda x: fftn(x,axes=(0,1,2))
ifft3d_gpu=lambda x: ifftn(x,axes=(0,1,2))

#fft3d_gpu=lambda x: fftn(x,axes=(0,1,2),out=x,overwrite_x=True)
#ifft3d_gpu=lambda x: ifftn(x,axes=(0,1,2),out=x,overwrite_x=True)

"""
Auxiliary procedures
"""

def x_mul_y(x,y):
    
    # Input: x: n*1, y: n*m.
    # Output: diag(x)y.
    
    return (y.T*x).T

def scalar_prod(x,ind,c=None):
    
    # ind={indexes, coefficient}
    # x[ind]=x[ind]*c
    
    if c is None:
        x[ind[0]]*=ind[1]
    else:
        x[ind]*=c
    
    return x

def hermitize(M):
    
    # Make a 2D square matrix hermitian.
    
    return (M+M.T.conj())/2

    
"""
Basic operations.

kron_diag: return the diagonal of kronecker product of two diagonal matrices.
mfd_stencil: compute the symmetric difference stencil on a line.
mfd_fft_blocks: compute the fft blocks of discrete curl, grad operators. 
full_fft_blocks: successive operations of mfd_fft_blocks.
diag_circulant_complex: return the FFT diagonal blocks of a circulant matrix.
diag_circulant_adjoint: return the FFT diagonal blocks of AA', where A is circulant.
inverse_3_times_3_block: return the inverse of a 3*3 matrix block, each block is diagonal.

FFT3d, IFFT3d: a standard FFT,IFFT along three dimensions.
H_fft, A_fft: implement sparse matrix multiplication via FFT.

"""

"""
Scaling: a=2*pi.
"""

def kron_diag(A,B):

    # Input: array A, B.
    # Output: kronecker product of diag(A)\otimes diag(B).

    n,m=len(A),len(B)
    A=A.reshape(1,n)
    B=B.reshape(m,1)
    
    if isinstance(A,cp.ndarray):
        C=cp.dot(B,A)
    else:
        C=np.dot(B,A)

    return C.reshape(m*n,order='F')

def mfd_stencil(k,ind):
    
    # Input: k, order; ind, index.
    # Output: 1D finite difference stencil (symmetric)
    
    mfd_mat=np.zeros([2*k,2*k])
    mfd_coef=np.zeros([2*k])
    mfd_coef[ind]=ind+1
    
    for j in range(2*k):
        w=2*(j-k)+1
        mfd_mat[0][j]=1
        for i in range(1,2*k):
            mfd_mat[i][j]=mfd_mat[i-1][j]*w
    
    return np.linalg.solve(mfd_mat,mfd_coef)

def diag_circulant_complex(sten,L,ind,N):
    
    # Diagonals of A, A is a circulant matrix.
    
    # Input: sten, FD stencil; L, length; ind, index; N, size of mat.
    # Output: diagonal of a circulant matrix.
    
    d0=np.zeros(N,dtype=complex)
    for i in range(N):
        for j in range(ind-1,L):
            d0[i]=d0[i]+sten[j]*np.exp(i*1j*(j-ind+1)*2.0*pi/N)
        
        for j in range(0,ind-1):
            d0[i]=d0[i]+sten[j]*np.exp(i*1j*(N-1+j-ind+2)*2.0*pi/N)
        
    return d0

def diag_circulant_adjoint(sten,L,ind,N):
    
    # Diagonals of AA', A is a  circulant matrix.
    
    # Input: sten, FD stencil; L, length; ind, index; N, size of mat.
    # Output: diagonal of a circulant matrix.
    
    d0=np.zeros(N)
    for i in range(N):
        tmp=0+0*1j
        for j in range(ind-1,L):
            tmp=tmp+sten[j]*np.exp(i*1j*(j-ind+1)*2.0*pi/N)
        
        for j in range(0,ind-1):
            tmp=tmp+sten[j]*np.exp(i*1j*(N-1+j-ind+2)*2.0*pi/N)
        
        d0[i]=np.abs(tmp)**2
        
    return d0

def inverse_3_times_3_block(D11,D22,D33,D12,D13,D23):
    
    # Compute the inverse of 3*3 hermitian matrix block.
    # Input: Blocks of the upper part,
    #        [[D11,D12,D13],[D12',D22,D23],[D13' D23' D33]]
    # Output: Upper blocks of the inverse. 

    DET=(D11*D22*D33-(D11*(D23*D23.conj())+D22*(D13*D13.conj())\
        +D33*(D12*D12.conj())))+2*(D12*D23*D13.conj())
    
    F_diag=np.hstack(((D22*D33-D23*D23.conj())/DET,\
                      (D11*D33-D13*D13.conj())/DET,\
                      (D11*D22-D12*D12.conj())/DET))
    
    F_sdiag=np.hstack(((D13*D23.conj()-D12*D33)/DET,\
                       (D12*D23-D13*D22)/DET,\
                       (D13*D12.conj()-D11*D23)/DET))
    
    if not str(D11.dtype)[0]=='c':
        F_diag=F_diag.real
    return (F_diag,F_sdiag)

def inverse_3_times_3_B(B,pnt,shift=0):
    
    # Compute the inverse of 3*3 hermitian matrix block
    # using the upper FFT diagonal of B.
    
    n=round(len(B[0])/3)
    
    return inverse_3_times_3_block\
        (pnt*B[0][0:n]+B[0][n:2*n]+B[0][2*n:]+shift,\
         B[0][0:n]+pnt*B[0][n:2*n]+B[0][2*n:]+shift,\
         B[0][0:n]+B[0][n:2*n]+pnt*B[0][2*n:]+shift,\
         (pnt-1)*B[1][0:n],(pnt-1)*B[1][n:2*n],(pnt-1)*B[1][2*n:])

"""
Matrix blocks
"""

def fft_blocks(N,k,CT,alpha=None):
    
    # Input: a,lattice constant; N, grid size; k, accuracy;
    #        alpha, shift in Brillouin zone.
    
    # Output: diagonal blocks after FFT's diagonalization
    
    # OPTION: If alpha appears in the input then output returns a complete FFT
    #         block A,B. Otherwise D,Di are returned.
    
    h=2*pi/N    
    n=N**3
    
    fd_stencil_0=mfd_stencil(k,0)
    fd_stencil_1=mfd_stencil(k,1)
    
    D1=diag_circulant_complex(fd_stencil_1/h, 2*k, k, N)
    D0=diag_circulant_complex(fd_stencil_0, 2*k, k, N)
    
    D01=kron_diag(np.ones(N*N), D1)
    D02=kron_diag(np.ones(N),kron_diag(D1,np.ones(N)))
    D03=kron_diag(D1,np.ones(N*N))
    
    if alpha is None:
        Di=np.hstack((kron_diag(np.ones(N*N),D0),\
                  kron_diag(np.ones(N),kron_diag(D0,np.ones(N))),\
                  kron_diag(D0,np.ones(N*N))))
    
        D=np.hstack((CT[0][0]*D01+CT[0][1]*D02+CT[0][2]*D03,\
                 CT[1][0]*D01+CT[1][1]*D02+CT[1][2]*D03,\
                 CT[2][0]*D01+CT[2][1]*D02+CT[2][2]*D03))
            
        return D,Di
    else:
        alpha=alpha/(2*pi)
        A=np.hstack((CT[0][0]*D01+CT[0][1]*D02+CT[0][2]*D03+\
                        1j*alpha[0]*kron_diag(np.ones(N*N),D0),\
                     CT[1][0]*D01+CT[1][1]*D02+CT[1][2]*D03+\
                        1j*alpha[1]*kron_diag(np.ones(N),kron_diag(D0,np.ones(N))),\
                     CT[2][0]*D01+CT[2][1]*D02+CT[2][2]*D03+\
                        1j*alpha[2]*kron_diag(D0,np.ones(N*N))))
    
        B=(np.hstack(((A[0:n]*A[0:n].conj()).real,\
                      (A[n:2*n]*A[n:2*n].conj()).real,\
                      (A[2*n:]*A[2*n:].conj()).real)),\
           np.hstack((A[0:n].conj()*A[n:2*n],A[0:n].conj()*A[2*n:],A[n:2*n].conj()*A[2*n:])))    
        return A,B

# Relaxation parameters: shift, penalty and deflation.
def set_relaxation(N,alpha):
    
    nrm_alpha=np.linalg.norm(alpha)
    if nrm_alpha>pi/5:
        opt=(0,0.3)
        pnt=2*N
    elif nrm_alpha==0:
        opt=(1.0/(4.0*pi)/N,0.35)
        pnt=2*N
    else:
        opt=(nrm_alpha,0.4)
        pnt=2*N/nrm_alpha**2
    
    return opt,pnt
    

"""
FFT implementation.

Reminder: If input is an 1-D array or 2-D array with a single column,
          the output of H_fft and A_fft would automatically be flattened.

"""

def H_fft(X,DIAG):
    
    # Hermitian matrix multiplication done by FFT.
    # Input: X (multicolumn vectors to be manipulated)
    #        DIAG is a cell that stores the diagonals of 3*3 blocks.
    
    # Output (in-place): X:=H*X.
    
    NP=arrtype(X)
    if NP==np:
        FFT,IFFT=fft3d_cpu,ifft3d_cpu
    else:
        FFT,IFFT=fft3d_gpu,ifft3d_gpu
    
    if X.ndim==1:
        nn,m=len(X),1
    else:
        nn,m=X.shape
    nn=round(nn/3)
    n=round(nn**(1/3))
    
    X=FFT(X.reshape(n,n,n,3*m,order='F')).reshape(3*nn,m,order='F')
    
    VM=NP.vstack
    X=VM((x_mul_y(DIAG[0][0:nn],X[0:nn,:])+x_mul_y(DIAG[1][0:nn],X[nn:2*nn,:])\
                  +x_mul_y(DIAG[1][nn:2*nn],X[2*nn:,:]),\
          x_mul_y(DIAG[1][0:nn].conj(),X[0:nn,:])+x_mul_y(DIAG[0][nn:2*nn],X[nn:2*nn,:])\
                  +x_mul_y(DIAG[1][2*nn:],X[2*nn:,:]),\
          x_mul_y(DIAG[1][nn:2*nn].conj(),X[0:nn,:])+x_mul_y(DIAG[0][2*nn:],X[2*nn:,:])\
                  +x_mul_y(DIAG[1][2*nn:].conj(),X[nn:2*nn,:])))
    
    X=IFFT(X.reshape(n,n,n,3*m,order='F')).reshape(3*nn,m,order='F')
    
    if m==1:
        X=X.flatten()

    return X

def A_fft(X,D):
    
    # Matrix multiplication of A done by FFT.
    # Input: X (multicolumn vectors to be manipulated)
    #        A=[[0,-D3,D2],[D3,0,-D1],[-D2,D1,0]]
    
    # pack_name: support only "np","sc","cp","cpx"
    
    # Output (in-place): X:=A*X.
    
    NP=arrtype(X)
    if NP==np:
        FFT,IFFT=fft3d_cpu,ifft3d_cpu
    else:
        FFT,IFFT=fft3d_gpu,ifft3d_gpu
    
    if X.ndim==1:
        nn,m=len(X),1
    else:
        nn,m=X.shape
    nn=round(nn/3)
    n=round(nn**(1/3))
    
    X=FFT(X.reshape(n,n,n,3*m,order='F')).reshape(3*nn,m,order='F')
    
    VM=NP.vstack
    X[:]=VM((-x_mul_y(D[2*nn:],X[nn:2*nn,:])+x_mul_y(D[nn:2*nn],X[2*nn:,:]),\
             x_mul_y(D[2*nn:],X[0:nn,:])-x_mul_y(D[0:nn],X[2*nn:,:]),\
             -x_mul_y(D[nn:2*nn],X[0:nn,:])+x_mul_y(D[0:nn],X[nn:2*nn,:])))
    
    X=IFFT(X.reshape(n,n,n,3*m,order='F')).reshape(3*nn,m,order='F')

    if m==1:
        X=X.flatten()
        
    return X
        


"""

Function handle of AM_0A'+pnt BB' and inverse of its preconditioner (AA'+pnt BB').

"""

def PCs_mfd_handle(n,**kwargs):
    
    # Warning: DoFs = 3n^3, please be careful with the input n.
    
    """
    Input:
        n:   grid size.
    
    **kwargs: 
        alpha, eps, shift, lattice type and gpu switch.
    
    Output:
        a_func:   function handle of AM0A'+pnt B'B.
        p_func:   function handle of inv(AA'+pnt B'B).
    """
    
    opts={'alpha':np.array([pi,pi,pi]),
          'gpu_opt':True,
          'eps':15.0,
          'shift':0.0,
          'd_flag_name': None
          }
    opts.update(kwargs)
    
    alpha,gpu_opt,eps,shift=opts['alpha'],opts['gpu_opt'],opts['eps'],opts['shift']
    d_flag_name=opts['d_flag_name']
    pnt=2.0*n
    
    if d_flag_name is None:
        den=(3.0-5**0.5)/2
        diel_ind=np.random.randint(3*n**3-1,size=(int(den*3*n**3)))
        ct=np.eye(3)
    else:
        diel_ind,ct,__=diel.dielectric_save_and_load(d_flag_name)
    
    a_fft,b_fft=fft_blocks(n,1,ct,alpha)
    inv_fft=inverse_3_times_3_B(b_fft,pnt,shift=shift)
    if gpu_opt:
        a_fft,b_fft,inv_fft=cp.asarray(a_fft),cp.asarray(b_fft),cp.asarray(inv_fft)
    
    a_func=lambda x: A_fft(scalar_prod(A_fft(x,-a_fft.conj()),diel_ind,1/eps),\
                     a_fft)+pnt*H_fft(x,b_fft)
    p_func=lambda x: H_fft(x,inv_fft)
    
    return a_func,p_func
    
