# -*- coding: utf-8 -*-
#
# Created on 2024-06-10 (Monday) at 18:05:57
#
# Author: Epsilon-79th
#
# Usage: Self-defined LOCAL eigensolvers.
#

from time import time

import numpy as np
import scipy as sc
import cupy as cp
from cupy.cublas import gemm

from environment import *

#from slepc4py import SLEPc
#from petsc4py import PETSc

TOL = 1e-5
maxiter = 1000

# Hermitization of a square matrix.  
def hermitize(M, out = None):
    if out is None:
        return (M+M.T.conj())/2
    else:
        arrtype(M).copyto(out, M)
        out += M.T.conj()
        out /= 2
        return None
    
# Short QR done by chol.
def short_qr(x):

    """
    Usage:
        Short QR via Cholesky factorization.
        Supports both cpu and gpu arrays.
    """

    NP = arrtype(x)
    l = NP.linalg.cholesky(hermitize(x.T.conj()@x))
    return x @ NP.linalg.inv(l.T.conj())

def rr_svd(xx):

    NP = arrtype(xx)
    l = NP.linalg.cholesky(xx)
    _ , sigmas, v = NP.linalg.svd(l.T.conj())

    return sigmas, v

# Basic power method to find max eigenvalue.
def power_method(A, x0, maxiter = maxiter, tol = TOL, xi_max = True):

    """
    Find max eigenvalue by power method.
    A should be a function handle, initial x0 provided.
    """
    
    print(f"\n{GREEN}Power method for max eigenvalue.{RESET}\n")

    x = x0.copy()
    if xi_max:
        get_l = lambda x: x[cp.argmax(cp.max(x))]
    else:
        get_l = lambda x: norm(x)
    
    for i in range(maxiter):
        Ax = A(x)
        l = get_l(Ax)
        x = Ax / l
        r = cp.linalg.norm(Ax - l*x, ord = cp.inf) / cp.abs(l)
        print(f"Iter = {i+1}, l = {l:<8.3e}, res = {r:<6.3e}.")
        if r < tol:
            break

    if i == maxiter - 1:
        print(f"\n{RED}Tolerance {tol:<6.3e} isn't reached after {maxiter} iterations.{RESET}\n")
    else:
        print(f"\n{GREEN}Convergence reached after {i+1} iterations.{RESET}\n")
    return l
    
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

def GEP_chol(T,G, herm = True, slice = None):
    # GEP on gpu 2: GEP to SEP via cholesky.
    
    t_h = time.time()
    L = cp.linalg.inv(cp.linalg.cholesky(G))
    T = cp.dot(cp.dot(L,T),L.T.conj())
    if herm:        
        lambdas,eigvec = cp.linalg.eigh(T)
    else:
        lambdas,eigvec = cp.linalg.eig(T)
    eigvec = cp.dot(L.T.conj(),eigvec)        
    t_chol = owari_cuda() - t_h

    if slice is None:
        return lambdas, eigvec, t_chol
    else:
        return lambdas[:slice], eigvec[:,:slice], t_chol

def GEP_cpu(T_in, G_in, herm = True, slice = None):
    
    # GEP on cpu: call scipy.linalg.eigh().
    T, G = T_in.get(), G_in.get()
    
    t_h = time.time()
    if herm:
        lambdas, eigvec = sc.linalg.eigh(T,G)
    else:
        lambdas, eigvec = sc.linalg.eig(T,G)
        idx = np.argsort(lambdas.real)
        lambdas = lambdas[idx]
        eigvec = eigvec[:,idx]
    t_eigh = owari_cuda() - t_h
    
    if slice is None:
        return cp.asarray(lambdas), cp.asarray(eigvec), t_eigh
    else:
        return cp.asarray(lambdas[:slice]), cp.asarray(eigvec[:,:slice]), t_eigh

"""
Rayleigh-Ritz Procedure.
"""
def rayleigh_ritz_chol_sep(s, hs):

    t_h = time.time()
    ss = hermitize(gemm('H', 'N', s, s))
    shs = hermitize(gemm('H', 'N', s, hs))
    t_mul = owari_cuda() - t_h

    t_h = time.time()
    L = cp.linalg.inv(cp.linalg.cholesky(ss))
    shs = cp.dot(cp.dot(L,shs),L.T.conj())
    lambdas, eigvec = cp.linalg.eigh(shs)
    eigvec = cp.dot(L.T.conj(),eigvec)        
    t_chol = owari_cuda() - t_h

    return lambdas, eigvec, t_mul + t_chol

def _inplace_mgs_qr(s):
   
    n, k = s.shape
    r = cp.zeros((k, k), dtype=s.dtype)
    
    for i in range(k):
        norm = cp.linalg.norm(s[:, i])
        if norm < 1e-14:
            r[i, i] = 0
            continue
            
        r[i, i] = norm
        s[:, i] /= norm
        
        if i + 1 < k:
            projections = s[:, i].T.conj() @ s[:, i+1:]
            r[i, i+1:] = projections
            s[:, i+1:] -= s[:, i:i+1] @ projections[None, :]
            
    return s, r

def rayleigh_ritz_qr_sep(s, hs):

    t_h = time.time()
    #_ , r = cp.linalg.qr(s, mode='reduced')
    _ , r = _inplace_mgs_qr(s)
    r = cp.linalg.inv(r)
    shs = hermitize(r.T.conj() @ gemm('H','N',s,hs) @ r)
    l, v = cp.linalg.eigh(shs)
    v = r @ v
    t_rr = owari_cuda() - t_h
    return l, v, t_rr

def rayleigh_ritz_svd(hs):

    t_h = time.time()
    _ , r = cp.linalg.qr(hs)
    _ , l, g = cp.linalg.svd(r)
    t_rr = owari_cuda() - t_h

    return l, g, t_rr

def rayleigh_ritz_chol_gep(s, ms, hs):

    t_h = time.time()
    sms = hermitize(gemm('H', 'N', s, ms))
    shs = hermitize(gemm('H', 'N', s, hs))
    t_mul = owari_cuda() - t_h
    l, v, t_gep = GEP_chol(shs, sms)

    return l, v, t_mul + t_gep

def rayleigh_ritz_qr_gep(s, ms, hs):

    t_h = time.time()
    _ , r = cp.linalg.qr(s)
    r = cp.linalg.inv(r)

    # Warning: cupy.linalg.eigh ONLY solves SEP.
    sms = hermitize(r.T.conj() @ gemm('H','N',s,ms) @ r).get()
    shs = hermitize(r.T.conj() @ gemm('H','N',s,hs) @ r).get()
    l, v = sc.linalg.eigh(shs, sms)
    v_out = r @ cp.asarray(v)
    t_rr = owari_cuda() - t_h
    return cp.asarray(l), v_out, t_rr

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
    
    