# -*- coding: utf-8 -*-
#
# Created on 2025-12-26 (Friday) at 22:50:00
#
# Author: Epsilon-79th
#
# Usage: GPU kernels.
#

import cupy as cp

# H_block, A_block elementwise operations.
h_block_kernel = cp.ElementwiseKernel(
    # T -> X, U -> DIAG[0], V -> DIAG[1]
    'raw T X, raw U D0, raw V D1, int32 nn, int32 m', 
    'T Y',
    '''
    int row = i / m;
    
    T val;
    if (row < nn) {
        val = (T)D0[row] * X[i] 
            + (T)D1[row] * X[i + nn * m] 
            + (T)D1[row + nn] * X[i + 2 * nn * m];
    } 
    else if (row < 2 * nn) {
        int r = row - nn;
        val = conj((T)D1[r]) * X[i - nn * m] 
            + (T)D0[row] * X[i] 
            + (T)D1[row + nn] * X[i + nn * m];
    } 
    else {
        int r = row - 2 * nn;
        val = conj((T)D1[r + nn]) * X[i - 2 * nn * m] 
            + (T)D0[row] * X[i] 
            + conj((T)D1[row]) * X[i - nn * m];
    }
    Y = val;
    ''',
    'h_block_complex_kernel'
)

a_block_kernel = cp.ElementwiseKernel(
    # T -> X (complex128), U -> D (complex128)
    'raw T X, raw U D, int32 nn, int32 m',
    'T Y',
    '''
    int row = i / m;
    
    T val;
    if (row < nn) {
        // AX1: -D[2nn:] * X[nn:2nn] + D[nn:2nn] * X[2nn:]
        val = -(T)D[row + 2 * nn] * X[i + nn * m] 
              + (T)D[row + nn] * X[i + 2 * nn * m];
    } 
    else if (row < 2 * nn) {
        // AX2: D[2nn:] * X[:nn] - D[:nn] * X[2nn:]
        int r = row - nn;
        val = (T)D[r + 2 * nn] * X[i - nn * m] 
              - (T)D[r] * X[i + nn * m];
    } 
    else {
        // AX3: -D[nn:2nn] * X[:nn] + D[:nn] * X[nn:2nn]
        int r = row - 2 * nn;
        val = -(T)D[r + nn] * X[i - 2 * nn * m] 
              + (T)D[r] * X[i - nn * m];
    }
    Y = val;
    ''',
    'a_block_complex_kernel'
)


