# -*- coding: utf-8 -*-
#
# Created on 2024-06-10 (月) at 17:44:54
#
# Author: Epsilon-79th
#
# Usage: Self-defined GPU options including norm, timing.

import numpy as np
import cupy as cp
from time import time

"""
Norms.
"""

def norm(X):
    
    # Input: matrix X.
    # Output: Frobenius norm.

    NP=cp if isinstance(X,cp.ndarray) else np
    
    if X.ndim<=1:
        return NP.linalg.norm(X)
    else:
        return NP.sqrt((NP.trace(NP.dot(X.T.conj(),X))).real)

def norms(X):
    
    # Input: multicolumn vector X.
    # Output: an array containing norm of each column.
    
    NP=cp if isinstance(X,cp.ndarray) else np
    
    if X.ndim<=1:
        return NP.linalg.norm(X)
    else:
        return NP.sqrt((NP.diag(NP.dot(X.T.conj(),X))).real)
    
def dots(X,Y):
    NP=cp if isinstance(X,cp.ndarray) else np

    if X.ndim<=1:
        return NP.dot(X.conj(),Y)
    else:
        return NP.diag(NP.dot(X.T.conj(),Y))
    

"""
Package and timing.
"""

def arrtype(x):
    
    if "numpy" in str(type(x)):
        return np
    elif "cupy" in str(type(x)):
        return cp
    else:
        ValueError("The input should either be a numpy.ndarray or cupy.ndarray.")
        
def owari_opt(T=None):
    
    return lambda: (cp.cuda.Device().synchronize(),time())[1]\
        if not T is None and (T=="cp" or T=="gpu" or "cupy" in T)\
        else time

def owari_cuda():
    return (cp.cuda.Device().synchronize(),time())[1]

