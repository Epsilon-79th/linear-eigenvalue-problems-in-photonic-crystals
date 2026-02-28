# -*- coding: utf-8 -*-
#
# Created on 2025-03-09 (Sunday) at 15:57:27
#
# Author: Epsilon-79th
#
# Usage: FFT in PC model problem.
#

import cupy as cp
import cupyx as cpx
from _kernels import *

"""
Block matrix operations via cupy.ElementwiseKernel
"""

def H_block_kernel(X, DIAG):

    if X.ndim == 1:
        nn = X.size // 3
        m = 1
    else:
        nn = X.shape[0] // 3
        m = X.shape[1]

    Y = cp.empty_like(X)
    h_block_kernel(X, DIAG[0], DIAG[1], nn, m, Y)
    
    return Y

def A_block_kernel(X, D):
    if X.ndim == 1:
        nn = X.size // 3
        m = 1
    else:
        nn = X.shape[0] // 3
        m = X.shape[1]
    
    Y = cp.empty_like(X)
    a_block_kernel(X, D, nn, m, Y)
    
    return Y

"""
Block matrix operations via cupy.einsum.
Explicit but not opimally efficient.
"""

def H_block(X, DIAG):

    if X.ndim == 1:
        return H_block_dim1(X, DIAG)

    nn = X.shape[0]
    nn //= 3

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

def H_block_dim1(X, DIAG):

    nn = X.size
    nn //= 3

    HX1 = cp.einsum('i,i->i', DIAG[0][:nn], X[:nn]) + \
          cp.einsum('i,i->i', DIAG[1][:nn], X[nn:2*nn]) + \
          cp.einsum('i,i->i', DIAG[1][nn:2*nn], X[2*nn:])

    HX2 = cp.einsum('i,i->i', DIAG[1][:nn].conj(), X[:nn]) + \
          cp.einsum('i,i->i', DIAG[0][nn:2*nn], X[nn:2*nn]) + \
          cp.einsum('i,i->i', DIAG[1][2*nn:], X[2*nn:])

    HX3 = cp.einsum('i,i->i', DIAG[1][nn:2*nn].conj(), X[:nn]) + \
          cp.einsum('i,i->i', DIAG[0][2*nn:], X[2*nn:]) + \
          cp.einsum('i,i->i', DIAG[1][2*nn:].conj(), X[nn:2*nn])
    
    return cp.concatenate((HX1, HX2, HX3))

def A_block(X, D):

    if X.ndim == 1:
        return A_block_dim1(X, D)

    nn = X.shape[0]
    nn //= 3

    AX1 = -cp.einsum('i,ij->ij', D[2*nn:], X[nn:2*nn, :]) + \
           cp.einsum('i,ij->ij', D[nn:2*nn], X[2*nn:, :])

    AX2 = cp.einsum('i,ij->ij', D[2*nn:], X[:nn, :]) - \
          cp.einsum('i,ij->ij', D[:nn], X[2*nn:, :])

    AX3 = -cp.einsum('i,ij->ij', D[nn:2*nn], X[:nn, :]) + \
           cp.einsum('i,ij->ij', D[:nn], X[nn:2*nn, :])

    return cp.concatenate((AX1, AX2, AX3), axis=0)

def A_block_dim1(X, D):

    nn = X.size
    nn //= 3

    AX1 = -cp.einsum('i,i->i', D[2*nn:], X[nn:2*nn]) + \
           cp.einsum('i,i->i', D[nn:2*nn], X[2*nn:])

    AX2 = cp.einsum('i,i->i', D[2*nn:], X[:nn]) - \
          cp.einsum('i,i->i', D[:nn], X[2*nn:])

    AX3 = -cp.einsum('i,i->i', D[nn:2*nn], X[:nn]) + \
           cp.einsum('i,i->i', D[:nn], X[nn:2*nn])

    return cp.concatenate((AX1, AX2, AX3))

"""
Assemble global matrix-free function handle.
"""

def AMA(X, D_A, diel):

    """
    Usage:
        Implement AMA' via DFT.
    """

    a_handle = A_block_kernel

    if X.ndim == 1:
        nn, m = len(X), 1
    else:
        nn, m = X.shape
    
    nn //= 3
    n  = round(nn ** (1/3))

    # In-place FFT.
    AX = a_handle(X, -D_A.conj()).reshape(n,n,n,3*m, order = 'F')
    AX = cpx.scipy.fft.fftn(AX, axes = (0,1,2), overwrite_x = True)
    AX = diel(AX.reshape(3*nn,m, order = 'F')).reshape(n,n,n,3*m, order = 'F')
    AX = cpx.scipy.fft.ifftn(AX, axes = (0,1,2), overwrite_x = True)
    AX = a_handle(AX.reshape(3*nn,m, order = 'F'), D_A)

    # Previous operation increases the dimension.
    if X.ndim == 1:
        AX = AX.ravel()

    return AX

def AMA_BB(X, D_A, D_B, diel, shift = 0):
    
    r"""
    Usage:
        Implement AMA'+\gamma B'B via DFT.
        shift is optional, default 0.
    """

    h_handle = H_block_kernel
    HX = AMA(X, D_A, diel)

    # Previous operation increases the dimension.
    if X.ndim == 1:
        HX = HX.ravel()

    #HX += H_block(X, D_B)
    HX += h_handle(X, D_B)

    if shift != 0:
        HX += shift * X

    return HX
