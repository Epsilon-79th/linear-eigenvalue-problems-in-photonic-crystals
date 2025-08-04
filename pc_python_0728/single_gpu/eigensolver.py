# -*- coding: utf-8 -*-
#
# Created on 2024-06-10 (Monday) at 18:05:57
#
# Author: Epsilon-79th
#
# Usage: Self-defined eigensolvers.
#

from time import time

import numpy as np
import scipy as sc
import cupy as cp

from gpu_opts import *

#from slepc4py import SLEPc
#from petsc4py import PETSc

TOL = 1e-5
ITER_MAX = 1000

# Hermitization of a square matrix.  
def hermitize(M, out = None):
    if out is None:
        return (M+M.T.conj())/2
    else:
        arrtype(M).copyto(out, M)
        out += M.T.conj()
        out /= 2
        return None
    
"""
Generalized eigenvalue problem (GEP).
If on cpu, the scipy.linalg.eigh allows two inputs thus GEP is directly solved.
If on gpu, the scipy.linalg.eigh ONLY allows single argin, which means GEP can't be explicitly solved
           on gpudevice. Now we give two solutions: the 1st is to use '.get()' to drag the arrays 
           back to cpu then scipy.linalg.eigh is available. The 2nd is to transform GEP into SEP via
           Cholesky factorization. 

When the relatively 'small' GEP is in a large scale, the communication costs of gpuarray to cpuarray
might be more expensive than chol. 
"""

def GEP_get(T,G):
    # GEP on gpu 1: cupy to scipy, eigh(), scipy to cupy.
    
    t_h = time()
    T,G = T.get(), G.get() 
    lambdas, eigvec = sc.linalg.eigh(T,G)
    lambdas, eigvec = cp.asarray(lambdas), cp.asarray(eigvec)
    t_o = owari_cuda()

    return lambdas, eigvec, t_o - t_h

def GEP_chol(T,G, herm = True, slice = None):
    # GEP on gpu 2: GEP to SEP via cholesky.
    
    t_h = time()
    L = cp.linalg.inv(cp.linalg.cholesky(G))
    T = cp.dot(cp.dot(L,T),L.T.conj())
    if herm:        
        lambdas,eigvec = cp.linalg.eigh(T)
    else:
        lambdas,eigvec = cp.linalg.eig(T)
    eigvec = cp.dot(L.T.conj(),eigvec)        
    t_o = owari_cuda()

    if slice is None:
        return lambdas, eigvec, t_o - t_h
    else:
        return lambdas[:slice], eigvec[:,:slice], t_o - t_h

def GEP_cpu(T_in, G_in, herm = True):
    
    # GEP on cpu: call scipy.linalg.eigh().
    T, G = T_in.get(), G_in.get()
    
    t_h = time()
    if herm:
        lambdas, eigvec = sc.linalg.eigh(T,G)
    else:
        lambdas, eigvec = sc.linalg.eig(T,G)
    t_o = time()
    
    if slice is None:
        return cp.asarray(lambdas), cp.asarray(eigvec), t_o - t_h
    else:
        return cp.asarray(lambdas[:slice]), cp.asarray(eigvec[:,:slice]), t_o - t_h


"""
    Slepc. 
"""

"""

def to_petsc(mat_in):
    if isinstance(mat_in, cp.ndarray):
        mat = mat_in.get()
    else:
        mat = mat_in.copy()
    mat_petsc = PETSc.Mat().create()
    mat_petsc.setType("dense")
    mat_petsc.setSizes(mat.shape)
    mat_petsc.setUp()
    mat_petsc[:, :] = mat
    mat_petsc.assemble()
    return mat_petsc

def slepc_gnhep(A, B):
    nev = A.shape[0]    # Solve all eigenpairs.
    A_petsc = to_petsc(A)
    B_petsc = to_petsc(B)

    eps = SLEPc.EPS().create()
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setOperators(A_petsc, B_petsc)
    eps.setDimensions(nev=nev)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.ALL)
    eps.setType(SLEPc.EPS.Type.LAPACK)
    eps.setTolerances(tol=1e-8)
    eps.setFromOptions()

    eps.solve()
    
    assert eps.getConverged() == nev, f"SLEPc convergence failed"
    
    vals = np.empty(nev, dtype = complex)
    eigv = np.empty((nev, nev), dtype = complex)
    for i in range(nev):
        l = eps.getEigenvalue(i)
        v = A_petsc.getVecLeft()
        eps.getEigenvector(i, v)
        vals[i] = l
        eigv[:,i] = np.array(v.getArray())
        
        r = A@eigv[:,i] - vals[i]*B@eigv[:,i]
        res_norm = np.linalg.norm(r)
        print(f"i = {i+1:<4d}, res = {res_norm:<6.3e}.")

    if isinstance(A, cp.ndarray):
        return cp.asarray(vals), cp.asarray(eigv)
    else:
        return vals, eigv

def slepc_nep():

    return
"""


def main():
    
    N = 100
    A = cp.random.rand(N,N) + 1j * cp.random.rand(N,N)
    B = cp.random.rand(N,N) + 1j * cp.random.rand(N,N)
    B = B+B.T+ cp.sqrt(N) * cp.eye(N)
    
    l, v, _ = GEP_cpu(A,B, herm=False)
    print(f"N = {N}, residual = {cp.linalg.norm(A@v - (B@v)*l):<6.3e}.")        
    print(f"ortho = {cp.diag(v.T.conj()@v)}")
    return

if __name__ == '__main__':
    main()
    
    