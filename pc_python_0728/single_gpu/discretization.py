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
import cupy as cp

from numpy import pi
from cmath import exp

import scipy as sc
from cupyx.scipy.fft import fftn,ifftn

from time import time
owari = lambda: (cp.cuda.Device().synchronize(),time())[1]

fft3d_cpu=lambda x: sc.fft.fftn(x,axes=(0,1,2))
ifft3d_cpu=lambda x: sc.fft.ifftn(x,axes=(0,1,2))

fft3d_gpu=lambda x: fftn(x,axes=(0,1,2))
ifft3d_gpu=lambda x: ifftn(x,axes=(0,1,2))

def x_mul_y(x,y):
    return (y.T*x).T

"""
Auxiliary procedures
"""

# Relaxation parameters: shift, penalty and deflation.
def set_relaxation(N, alpha):
    
    nrm_alpha = np.linalg.norm(alpha)
    if nrm_alpha > pi / 5:
        opt = (0, 0.3)
        pnt = 2 * N
    elif nrm_alpha == 0:
        opt = (1.0 / (pi), 0.35)
        pnt = 2*N
    else:
        opt = (nrm_alpha, 0.4)
        pnt = 2 * N / (nrm_alpha**0.5)
    
    return opt,pnt

def scalar_prod(x,ind,c=None):
    
    # ind={indexes, coefficient}
    # x[ind]=x[ind]*c
    
    if c is None:
        x[ind[0]]*=ind[1]
    else:
        x[ind]*=c
    
    return x

    
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
    
    # Input:  Two 1D array A,B.
    # Output: Entries of diag(A) \otimes diag(B).

    n, m = len(A), len(B)
    return (B.reshape(m, 1) @ A.reshape(1, n)).reshape(m * n,order='F')

def mfd_stencil(k,ind):
    
    # Input:  Order k and index ind.
    # Output: 1D symmetric finite difference stencil.
    
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
            d0[i]=d0[i]+sten[j]*exp(i*1j*(j-ind+1)*2.0*pi/N)
        
        for j in range(0,ind-1):
            d0[i]=d0[i]+sten[j]*exp(i*1j*(N-1+j-ind+2)*2.0*pi/N)
        
    return d0

def diag_circulant_adjoint(sten,L,ind,N):
    
    # Diagonals of AA', A is a  circulant matrix.
    
    # Input: sten, FD stencil; L, length; ind, index; N, size of mat.
    # Output: diagonal of a circulant matrix.
    
    d0=np.zeros(N)
    for i in range(N):
        tmp=0+0*1j
        for j in range(ind-1,L):
            tmp=tmp+sten[j]*exp(i*1j*(j-ind+1)*2.0*pi/N)
        
        for j in range(0,ind-1):
            tmp=tmp+sten[j]*exp(i*1j*(N-1+j-ind+2)*2.0*pi/N)
        
        d0[i]=np.abs(tmp)**2
        
    return d0

def inverse_3_times_3_block(D11,D22,D33,D12,D13,D23):
    
    # Compute the inverse of 3*3 hermitian matrix block.
    # Input: Blocks of the upper part,
    #        [[D11,D12,D13],[D12',D22,D23],[D13' D23' D33]]
    # Output: Upper blocks of the inverse. 

    DET=(D11*D22*D33-(D11*(D23*D23.conj())+D22*(D13*D13.conj())\
        +D33*(D12*D12.conj())))+2*(D12*D23*D13.conj())
    
    NP = cp if isinstance(D11,cp.ndarray) else np
    F_diag = NP.concatenate((
                (D22*D33-D23*D23.conj())/DET, (D11*D33-D13*D13.conj())/DET, (D11*D22-D12*D12.conj())/DET))
    
    F_sdiag = NP.concatenate((
                (D13*D23.conj()-D12*D33)/DET, (D12*D23-D13*D22)/DET, (D13*D12.conj()-D11*D23)/DET))
    
    if not str(D11.dtype)[0]=='c':
        F_diag=F_diag.real
    return (F_diag,F_sdiag)

def inverse_3_times_3_B(B, pnt, shift = 0):
    
    # Compute the inverse of 3*3 hermitian matrix block
    # using the upper FFT diagonal of B.
    
    n = round(len(B[0]) / 3)
    
    return inverse_3_times_3_block\
        (pnt*B[0][0:n]+B[0][n:2*n]+B[0][2*n:]+shift,\
         B[0][0:n]+pnt*B[0][n:2*n]+B[0][2*n:]+shift,\
         B[0][0:n]+B[0][n:2*n]+pnt*B[0][2*n:]+shift,\
         (pnt-1)*B[1][0:n],(pnt-1)*B[1][n:2*n],(pnt-1)*B[1][2*n:])

"""
Matrix blocks
"""

def fft_blocks(N, k, CT, alpha = None, gpu = False):
    
    """
    Input: 
        N:     Grid size.
        k:     Accuracy.
        CT:    Coordinate transform matrix.
    
    Output:
        diagonal blocks after FFT's diagonalization
    
    **kwargs: 
        alpha: Translation vector. (If alpha is not None then output returns a complete FFT
               block A,B. Otherwise D,Di are returned)
        gpu:   Using numpy(false)/cupy(true).
    """
    
    t1 = time()
    NP = cp if gpu else np
    
    h = 2 * pi / N    
    n = N**3
    
    fd_stencil_0 = mfd_stencil(k, 0)
    fd_stencil_1 = mfd_stencil(k, 1)
    
    D1 = NP.asarray(diag_circulant_complex(fd_stencil_1 / h, 2 * k, k, N))
    D0 = NP.asarray(diag_circulant_complex(fd_stencil_0, 2*k, k, N))
    o1 = NP.ones(N)
    o2 = NP.ones(N*N)

    D01 = kron_diag(o2, D1)
    D02 = kron_diag(o1, kron_diag(D1, o1))
    D03 = kron_diag(D1, o2)
    t1 = owari()-t1
    
    t2 = time()
    if alpha is None:
        Di = NP.concatenate((kron_diag(o2,D0), kron_diag(o1,kron_diag(D0,o1)), kron_diag(D0,o2)))
    
        D = NP.concatenate((CT[0][0]*D01+CT[0][1]*D02+CT[0][2]*D03,\
                            CT[1][0]*D01+CT[1][1]*D02+CT[1][2]*D03,\
                            CT[2][0]*D01+CT[2][1]*D02+CT[2][2]*D03))
            
        return D, Di
    else:
        alpha=alpha/(2*pi)
        A = NP.concatenate((
                CT[0][0]*D01+CT[0][1]*D02+CT[0][2]*D03+1j*alpha[0]*kron_diag(o2, D0),\
                CT[1][0]*D01+CT[1][1]*D02+CT[1][2]*D03+1j*alpha[1]*kron_diag(o1,kron_diag(D0,o1)),\
                CT[2][0]*D01+CT[2][1]*D02+CT[2][2]*D03+1j*alpha[2]*kron_diag(D0,o2)))
    
        B = (NP.concatenate(((A[0:n]*A[0:n].conj()).real,(A[n:2*n]*A[n:2*n].conj()).real,(A[2*n:]*A[2*n:].conj()).real)).astype(float),\
             NP.concatenate((A[0:n].conj()*A[n:2*n],A[0:n].conj()*A[2*n:],A[n:2*n].conj()*A[2*n:])))
        t2 = owari()-t2

        print()
        return A, B
    
