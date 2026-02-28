# -*- coding: utf-8 -*-
#
# Created on 2025-09-27 (Saturday) at 16:31:50
#
# Author: Epsilon-79th.
#
# Usage: Global environment for computing photonic crystals.
#

import numpy as np
import cupy as cp

from numpy import pi
import time
from contextlib import contextmanager
from functools import wraps

# Path of output.
OUTPUT_PATH = "output/"
DIEL_PATH = "dielectric_examples/"

# Global parameters.
K = 1                         # Stencil length.
NEV = 10                      # Number of desired eigenpairs.
SCAL = 1                      # Lattice scaling constant.
TOL = 4*pi*pi*1e-5            # Tolerance.
GAP = 20                      # Segmentation of the Brillouin zone.

# Chiral constants (positive real).
CHIRAL_EPS_EG = {"sc_curv":13.0, "bcc_sg":16.0, "bcc_dg":16.0, "fcc":13.0}

# Pseudochiral constants (Hermitian postive definite, 3*3). 
PSEUDOCHIRAL_EPS_LOC = [np.array([(1+0.875**2)**0.5, (1+0.875**2)**0.5, 1.0, -1j*0.875, 0.0, 0.0]),\
                        np.array([(1+0.875**2)**0.5, 1.0, (1+0.875**2)**0.5, 0.0, 1j*0.875, 0.0]),\
                        np.array([1.0346,0.5059,0.2595, -0.0163-0.2319j, 0.027+0.0827j, -0.2743-0.0076j]),\
                        np.array([3.0, 3.0, 3.0, np.sqrt(3)+1j, 1j, np.sqrt(2)*(1+1j)])/5.0]

# Linear/Eigensolver settings.
MAXITER = 500
RESTART_MAX = 100
N_SUBSPACE = 40

# Lattice type.
SC_F1  = 'sc_flat1'
SC_F2  = 'sc_flat2'
SC_C   = 'sc_curv'
BCC_SG = 'bcc_sg'
BCC_DG = 'bcc_dg'
FCC    = 'fcc'

# Chiroptical properties.
TYPE0 = "chiral"
TYPE1 = "pseudochiral_trivial"
TYPE2 = "pseudochiral_crossdof"
TYPE3 = "pseudochiral_crossdof2"

# Robust square root.
sqrt_robust = lambda x: 0 if x<1e-10 else x**0.5

# Color.
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"
RESET   = "\033[0m"

# Dielectric domain, symmetric points.
DIEL_LIB={'CT_sc':[[1,0,0],[0,1,0],[0,0,1]],\
          'CT_bcc':[[0,1,1],[1,0,1],[1,1,0]],\
          'CT_fcc':[[-1,1,1],[1,-1,1],[1,1,-1]],\
          'sym_sc':[[0,0,0],[pi,0,0],[pi,pi,0],\
                    [pi,pi,pi],[0,0,0]],\
          'sym_bcc':[[0,0,2*pi],[0,0,0],[pi,pi,pi],\
                     [0,0,2*pi],[pi,0,pi],[0,0,0],\
                     [0,2*pi,0],[pi,pi,pi],[pi,0,pi]],\
          'sym_fcc':[[0,2*pi,0],[pi/2,2*pi,pi/2],[pi,pi,pi],\
                     [0,0,0],[0,2*pi,0],[pi,2*pi,0],\
                     [3*pi/2,3*pi/2,0]]}

@contextmanager
def timing(process_name=None, time_array=None, index=None, runtime_dict=None, print_time=False, accumulate=False):
    """
    Usage:
        General TIMING contextmanager.
    Input:
        process_name: Name of process.
        time_array:   Storing time.
        index:        Index of time array.
        runtime_dict: Dict storing time.
        print_time:   Whether print time.
        accumulate:   Whether accumulate time.
    """
    
    t_h = time.time()
    yield
    t_o = owari_cuda()
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
            t_h = time.time()
            output = my_func()
            t_o = owari_cuda()
            
            print(f"Function {my_func.__name__} takes {t_o-t_h:<6.3f} s to ellapse.")

def solver_wrapper():
    
    """    
    Wrapping the linear solver function, output convergence info.
    Warning if maximum iterations reached.

    Parameters:
        iter_max (int): Maxmimum iterations allowed.
        
    """
    def decorator(solver_func):
        @wraps(solver_func)
        def wrapper(*args, **kwargs):
            
            iter_max=kwargs.get('iter_max',1000)
            t_h = time.time()
            
            # Solver.
            solution, it = solver_func(*args, **kwargs)
            t_o = time.time()
            runtime = t_o - t_h       
            
            solver_name=solver_func.__name__     
            print(f"Solver {solver_name} takes {it} iterations with runtime {runtime:<6.3f} s.")
            
            # Output warning if maximum iteration is reached.
            if it + 2 >= iter_max:
                print(f"{RED}Warning: {solver_name} reaches maxmimum iterations, solution might be inaccurate.{RESET}")
            else:
                print(f"{GREEN}Convergence of {solver_name} is reached.{RESET}")
            
            return solution, [it, runtime]
        return wrapper
    return decorator

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
    
    NP = arrtype(X)

    if X.ndim <= 1:
        return X.conj() @ Y
    else:
        return NP.diag(X.T.conj() @ Y)
    

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
    
    return lambda: (cp.cuda.Device().synchronize(),time.time())[1]\
        if not T is None and (T=="cp" or T=="gpu" or "cupy" in T)\
        else time.time()

def owari_cuda():
    return (cp.cuda.Device().synchronize(),time.time())[1]