# -*- coding: utf-8 -*-
#
# Created on 2025-03-09 (Sunday) at 15:57:27
#
# Author: Epsilon-79th
#
# Usage: FFT in PC model problem.
#

import numpy as np
import scipy as sc
import cupy as cp
from cupyx.scipy.fft import fftn, ifftn

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

# GPU environment.
FFT, IFFT = fft3d_gpu, ifft3d_gpu

def x_mul_y(x,y):
    return y*x[:,np.newaxis]
    # return (y.T*x).T

"""
Block matrix operations.
"""

def H_block(X, DIAG):

    nn = X.shape[0]
    nn //= 3

    """
    HX = cp.concatenate((x_mul_y(DIAG[0][:nn],X[:nn,:])+x_mul_y(DIAG[1][:nn],X[nn:2*nn,:])\
                         +x_mul_y(DIAG[1][nn:2*nn],X[2*nn:,:]),\
                         x_mul_y(DIAG[1][:nn].conj(),X[:nn,:])+x_mul_y(DIAG[0][nn:2*nn],X[nn:2*nn,:])\
                         +x_mul_y(DIAG[1][2*nn:],X[2*nn:,:]),\
                         x_mul_y(DIAG[1][nn:2*nn].conj(),X[:nn,:])+x_mul_y(DIAG[0][2*nn:],X[2*nn:,:])\
                         +x_mul_y(DIAG[1][2*nn:].conj(),X[nn:2*nn,:])), axis = 0)
    """

    HX1 = cp.einsum('i,ij->ij', DIAG[0][:nn], X[:nn, :]) + \
          cp.einsum('i,ij->ij', DIAG[1][:nn], X[nn:2*nn, :]) + \
          cp.einsum('i,ij->ij', DIAG[1][nn:2*nn], X[2*nn:, :])

    HX2 = cp.einsum('i,ij->ij', DIAG[1][:nn].conj(), X[:nn, :]) + \
          cp.einsum('i,ij->ij', DIAG[0][nn:2*nn], X[nn:2*nn, :]) + \
          cp.einsum('i,ij->ij', DIAG[1][2*nn:], X[2*nn:, :])

    HX3 = cp.einsum('i,ij->ij', DIAG[1][nn:2*nn].conj(), X[:nn, :]) + \
          cp.einsum('i,ij->ij', DIAG[0][2*nn:], X[2*nn:, :]) + \
          cp.einsum('i,ij->ij', DIAG[1][2*nn:].conj(), X[nn:2*nn, :])

    return cp.concatenate((HX1, HX2, HX3), axis=0)

def A_block(X, D):

    nn = X.shape[0]
    nn //= 3

    """
    AX = cp.concatenate((-x_mul_y(D[2*nn:],X[nn:2*nn,:])+x_mul_y(D[nn:2*nn],X[2*nn:,:]),\
                         x_mul_y(D[2*nn:],X[:nn,:])-x_mul_y(D[:nn],X[2*nn:,:]),\
                         -x_mul_y(D[nn:2*nn],X[:nn,:])+x_mul_y(D[:nn],X[nn:2*nn,:])), axis = 0)
    """

    AX1 = -cp.einsum('i,ij->ij', D[2*nn:], X[nn:2*nn, :]) + \
           cp.einsum('i,ij->ij', D[nn:2*nn], X[2*nn:, :])

    AX2 = cp.einsum('i,ij->ij', D[2*nn:], X[:nn, :]) - \
          cp.einsum('i,ij->ij', D[:nn], X[2*nn:, :])

    AX3 = -cp.einsum('i,ij->ij', D[nn:2*nn], X[:nn, :]) + \
           cp.einsum('i,ij->ij', D[:nn], X[nn:2*nn, :])

    return cp.concatenate((AX1, AX2, AX3), axis=0)

def AMA_BB(X_in, D_A, D_B, diel, shift = 0):

    if X_in.ndim == 1:
        nn, m = len(X_in), 1
    else:
        nn, m = X_in.shape
    
    nn //= 3
    n  = round(nn ** (1/3))

    HX = diel(IFFT(A_block(X_in, -D_A.conj()).reshape(n,n,n,3*m, order = 'F')).reshape(3*nn,m, order = 'F'))
    HX = A_block(FFT(HX.reshape(n,n,n,3*m, order = 'F')).reshape(3*nn,m, order = 'F'), D_A)
    HX += H_block(X_in, D_B)

    if shift != 0:
        HX += shift * X_in

    return HX