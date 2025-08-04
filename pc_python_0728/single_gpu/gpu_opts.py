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
from contextlib import contextmanager
from functools import wraps

RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"
RESET = "\033[0m"

@contextmanager
def timing(process_name=None, time_array=None, index=None, runtime_dict=None, print_time=False, accumulate=False):
    """
    通用计时 contextmanager。
    - process_name: 过程名（可选）
    - time_array: 存储时间的数组（可选，需配合 index）
    - index: time_array 的索引
    - runtime_dict: 存储时间的 dict（可选）
    - print_time: 是否打印时间
    - accumulate: 是否累加
    """
    t_h = time()
    yield
    t_o = time()
    elapsed = t_o - t_h

    if time_array is not None and index is not None:
        if accumulate:
            time_array[index] += elapsed
        else:
            time_array[index] = elapsed
    if runtime_dict is not None and process_name is not None:
        runtime_dict[process_name] = runtime_dict.get(process_name, 0) + elapsed
    if print_time and process_name is not None:
        print(f"Runtime of {process_name} is {elapsed:<6.3f} s.")
    

def general_wrapper():
    
    def decorator(my_func):
        @wraps(my_func)
        def wrapper(*args, **kwargs):
            t_h=time.time()
            output=my_func()
            t_o=time.time()
            
            print(f"Function {my_func.__name__} takes {t_o-t_h:<6.3f} s to ellapse.")

"""
Norms.
"""

def norm(X):
    
    """
    Input:  matrix X.
    Output: Frobenius norm.
    """    

    NP=cp if isinstance(X,cp.ndarray) else np
    
    if X.ndim<=1:
        return NP.linalg.norm(X)
    else:
        return NP.sqrt((NP.trace(NP.dot(X.T.conj(),X))).real)

def norms(X):
    
    """
    Input:  multicolumn vector X.
    Output: an array containing norm of each column.
    """  
    
    NP=cp if isinstance(X,cp.ndarray) else np
    
    if X.ndim<=1:
        return NP.linalg.norm(X)
    else:
        return NP.sqrt((NP.diag(NP.dot(X.T.conj(),X))).real)
    
def dots(X,Y):
    
    """
    Input:  multicolumn vector X,Y (same column number).
    Output: an array containing inner product of each column of X,Y, diag(X^H*Y).
    """
    
    NP = cp if isinstance(X,cp.ndarray) else np

    if X.ndim <= 1:
        return NP.dot(X.conj(), Y)
    else:
        return NP.diag(NP.dot(X.T.conj(), Y))
    

"""
Package and GPU timing.
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
        else time()

def owari_cuda():
    return (cp.cuda.Device().synchronize(),time())[1]


    