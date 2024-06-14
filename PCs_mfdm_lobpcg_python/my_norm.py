# -*- coding: utf-8 -*-
"""
Created on Thu Jun 6 01:05:04 2024

@author: Epsilon-79th.
"""

import numpy as np
import cupy as cp

def norm(X):
    
    # Input: matrix X.
    # Output: Frobenius norm.
    
    if isinstance(X,cp.ndarray):
        NP=eval("cp")
    else:
        NP=eval("np")
    
    if X.ndim==1 or NP.size(X,1)==1:
        return NP.linalg.norm(X)
    else:
        return NP.sqrt((NP.trace(NP.dot(X.T.conj(),X))).real)

def norms(X):
    
    # Input: multicolumn vector X.
    # Output: an array containing norm of each column.
    
    if isinstance(X,cp.ndarray):
        NP=eval("cp")
    else:
        NP=eval("np")
    
    if X.ndim==1 or NP.size(X,1)==1:
        return NP.linalg.norm(X)
    else:
        return NP.sqrt((NP.diag(NP.dot(X.T.conj(),X))).real)
    
def dots(X,Y):
    if isinstance(X,cp.ndarray):
        return cp.diag(cp.dot(X.T.conj(),Y))
    else:
        return np.diag(np.dot(X.T.conj(),Y))