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
import sympy

from numpy import pi

import dielectric as diel
from environment import *

from time import time
owari = lambda: (cp.cuda.Device().synchronize(),time())[1]

def x_mul_y(x,y):
    return (y.T*x).T

"""
Auxiliary procedures
"""

def set_relaxation(alpha, scal=SCAL):
    """
    Usage:
        Relaxation parameters: shift, penalty and deflation.
        In previous version, penalty was dependent on N.
    """
    
    nrm_alpha = np.linalg.norm(alpha/scal)
    if nrm_alpha > 1:
        opt = (0, 0.6)
        pnt = 4*pi*pi
    elif nrm_alpha == 0:
        opt = (1.0 / (pi), 0.6)
        pnt = 4*pi*pi
    else:
        opt = (nrm_alpha, 0.6)
        pnt = (2*pi / nrm_alpha)
    
    return opt,pnt

def scalar_prod(x,ind,c=None):
    
    # ind={indexes, coefficient}
    # x[ind]=x[ind]*c
    
    if c is None:
        x[ind[0]]*=ind[1]
    else:
        x[ind]*=c
    
    return x

def sparse_kron(A, B):

    """
    Usage:
        Kronecker product of two sparse matrices on GPU.
        WHY NOT USE SPARSE.KRON?
        - SCIPY~ is slow, CUPYX.SCIPY~ has some problems.
    Input:
        A, B: tuple / coo matrix / integer.
        If A or B then A or B = (r, c, v).
        (Warning: in such case neither last row nor column of B can be empty.)
        If A or B is an integer, then it's an identity matrix.
    Output:
        COO format 3-tuple (r,c,v).
    """

    if isinstance(A, int):
        r_A, c_A, v_A = cp.arange(A), cp.arange(A), cp.ones(A)
        nnz_A = A
    elif isinstance(A, tuple):
        r_A, c_A, v_A = A
        nnz_A = len(v_A)
    else:
        r_A, c_A, v_A = A.row, A.col, A.data
        nnz_A = A.nnz

    if isinstance(B, int):
        r_B, c_B, v_B = cp.arange(B), cp.arange(B), cp.ones(B)
        n_B, m_B, nnz_B = B,B,B
    elif isinstance(B, tuple):
        r_B, c_B, v_B = B
        n_B, m_B = max(r_B)+1, max(c_B)+1   # We ensure that last row/column isn't empty.
        nnz_B = len(v_B)
    else:
        r_B, c_B, v_B = B.row, B.col, B.data
        n_B, m_B = B.shape
        nnz_B = B.nnz

    return cp.repeat(r_A, nnz_B) * n_B + cp.tile(r_B, nnz_A),\
           cp.repeat(c_A, nnz_B) * m_B + cp.tile(c_B, nnz_A),\
           cp.repeat(v_A, nnz_B) * cp.tile(v_B, nnz_A)

def P01(A, ind_in, flag_left = True):

    """
    Usage:
        Extract certain rows or columns from a COO matrix.
    Input:
        A: coo sparse/ 3-tuple.

    """

    if isinstance(A, tuple):
        r, c, v = A
    else:
        r, c, v = A.row, A.col, A.data
    
    ind_in = cp.asarray(ind_in)
    if flag_left:
        ind = cp.where(cp.isin(r, ind_in))[0]
    else:
        ind = cp.where(cp.isin(c, ind_in))[0]

    return (r[ind], c[ind], v[ind]) if isinstance(A, tuple) \
        else cp.sparse.coo_matrix((v[ind],(r[ind],c[ind])), shape=A.shape)
    
"""
Basic operations.

kron_diag: return the diagonal of kronecker product of two diagonal matrices.
mfd_stencil: compute the symmetric difference stencil on a line.
mfd_fft_blocks: compute the fft blocks of discrete curl, grad operators. 
full_fft_blocks: successive operations of mfd_fft_blocks.
diag_circulant_complex: return the FFT diagonal blocks of a circulant matrix.
inverse_3_times_3_block: return the inverse of a 3*3 matrix block, each block is diagonal.

FFT3d, IFFT3d: a standard FFT,IFFT along three dimensions.
H_fft, A_fft: implement sparse matrix multiplication via FFT.

"""

def kron_diag(A,B):
    
    # Input:  Two 1D array A,B.
    # Output: Entries of diag(A) \otimes diag(B).

    n, m = len(A), len(B)
    return (B.reshape(m, 1) @ A.reshape(1, n)).reshape(m * n,order='F')

def mfd_stencil(k, deriv_order):
    """
    Usage:
        Calculates a 1D symmetric finite difference stencil with high numerical stability
        for any given derivative order.
    
        This function uses symbolic computation (SymPy) to avoid the ill-conditioned
        Vandermonde matrix problem and uses the correct factorial term for generality.

    Input:
        k: int, half-width of the stencil. The total stencil will have 2*k points.
        deriv_order: int, the order of the derivative to approximate (e.g., 0, 1, 2...).

    Output:
        1D numpy array (float64): The finite difference stencil coefficients.
    """
    num_points = 2 * k
    if deriv_order >= num_points:
        raise ValueError(f"Derivative order ({deriv_order}) must be less than "
                         f"the number of stencil points ({num_points}).")

    # Define notations and stencil points.
    points = [sympy.Integer(2 * (j - k) + 1) for j in range(num_points)]
    coeffs = sympy.symbols(f'c0:{num_points}')
    
    # Assemble linear system.
    equations = []
    for i in range(num_points):
        # sum(c_j * (p_j**i))
        expr = sum(c * (p**i) for c, p in zip(coeffs, points))
        if i == deriv_order:
            rhs = deriv_order + 1
        else:
            rhs = 0
            
        equations.append(sympy.Eq(expr, rhs))
        
    # Sympy solve.
    solution = sympy.linsolve(equations, coeffs)
    stencil = np.array([s.evalf() for s in solution.args[0]], dtype=float)
    
    return stencil

def diag_circulant_complex(sten, L, ind, N):
    """
    Usage:
        Diagonals of A, where A is a circulant matrix. This function calculates the 
        eigenvalues of the circulant matrix constructed from the stencil.

        This stable version uses the Fast Fourier Transform (FFT), which is more
        numerically stable and efficient than direct summation.

    Input:
        sten: 1D array, the FD stencil coefficients.
        L: int, length of the stencil (can be derived from sten).
        ind: int, 1-based index of the stencil's center element (the one on the main diagonal).
        N: int, size of the circulant matrix.

    Output:
        d0: 1D complex array, the diagonal of the diagonalized matrix (i.e., the eigenvalues).
    """
    # Assemble first row.
    c = np.zeros(N, dtype=complex)
    end_part_len = L - (ind - 1)
    c[0:end_part_len] = sten[ind-1 : L]
    start_part_len = ind - 1
    if start_part_len > 0:
        c[N - start_part_len : N] = sten[0 : start_part_len]

    # Apply FFT to compute eigenvalues.
    return np.fft.ifft(c) * N

def inverse_3_times_3_block(*args, shift = 0.0, hermitian = True):
    
    """
    Usage:
        Inverse of 3*3 matrix block with general diagonal and symmetric off-diagonal.
        Each block is diagonal.
    Input:
        diags: Diagonal entries.
        sdiags: Symmetric off-diagonal entries.
        shift:       Shift added to diagonal entries.
        hermitian:   Whether the matrix is hermitian.
    """

    if len(args) == 6:
        D11, D22, D33, D12, D13, D23 = args
        nn = len(D11)
        diags = cp.concatenate((D11,D22,D33))
        sdiags = cp.concatenate((D12,D13,D23))
        del D11, D22, D33, D12, D13, D23
    elif len(args) == 2:
        diags, sdiags = args
        nn = len(diags) // 3
    elif len(args) == 1:
        diags, sdiags = args[0]
        nn = len(diags) // 3

    if shift != 0.0:
        diags[:nn], diags[nn:2*nn], diags[2*nn:] = diags[:nn] + shift, diags[nn:2*nn] + shift, diags[2*nn:] + shift

    DET = (diags[:nn] * diags[nn:2*nn] * diags[2*nn:] \
          - (diags[:nn] * (sdiags[2*nn:] * sdiags[2*nn:].conj()) + diags[nn:2*nn] * (sdiags[nn:2*nn] * sdiags[nn:2*nn].conj())\
          + diags[2*nn:] * (sdiags[:nn] * sdiags[:nn].conj()))) + 2 * (sdiags[:nn] * sdiags[2*nn:] * sdiags[nn:2*nn].conj()).real
    
    F_diag = cp.concatenate((
                (diags[nn:2*nn]*diags[2*nn:]-sdiags[2*nn:]*sdiags[2*nn:].conj())/DET, \
                (diags[:nn]*diags[2*nn:]-sdiags[nn:2*nn]*sdiags[nn:2*nn].conj())/DET, \
                (diags[:nn]*diags[nn:2*nn]-sdiags[:nn]*sdiags[:nn].conj())/DET))
    
    if hermitian:
        DET, F_diag = DET.real, F_diag.real
    
    F_sdiag = cp.concatenate((
                (sdiags[nn:2*nn]*sdiags[2*nn:].conj()-sdiags[:nn]*diags[2*nn:])/DET, \
                (sdiags[:nn]*sdiags[2*nn:]-sdiags[nn:2*nn]*diags[nn:2*nn])/DET, \
                (sdiags[nn:2*nn]*sdiags[:nn].conj()-diags[:nn]*sdiags[2*nn:])/DET))
    
    return (F_diag,F_sdiag)

def inverse_3_times_3_A(A, shift = 1.0):
    
    # Establish the inverse of AA'+shift*I.
    
    n = round(len(A) / 3)
    ds = (A.conj() * A).real
    
    return inverse_3_times_3_block(\
        ds[n:2*n]+ds[2*n:], ds[:n]+ds[2*n:], ds[:n]+ds[n:2*n],\
        -A[:n].conj()*A[n:2*n], -A[:n].conj()*A[2*n:], -A[n:2*n].conj()*A[2*n:],\
        shift = shift)

def inverse_3_times_3_B(B, pnt, shift = 0.0):
    
    # Establish the inverse of 3*3 matrix block with identical diagonal.
    # B = (diag, sdiag) of the matrix block.
    
    n = round(len(B[0]) / 3)
    
    return inverse_3_times_3_block\
        (pnt*B[0][:n]+B[0][n:2*n]+B[0][2*n:],\
         B[0][:n]+pnt*B[0][n:2*n]+B[0][2*n:],\
         B[0][:n]+B[0][n:2*n]+pnt*B[0][2*n:],\
         (pnt-1)*B[1][:n],(pnt-1)*B[1][n:2*n],(pnt-1)*B[1][2*n:], shift = shift)

"""
Matrix blocks
"""

def fft_blocks(N, k, CT, alpha = None, scal = SCAL):
    
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
    """

    h = scal / N    
    n = N*N*N
    
    fd_stencil_0 = mfd_stencil(k, 0)
    fd_stencil_1 = mfd_stencil(k, 1)
    
    D1 = cp.asarray(diag_circulant_complex(fd_stencil_1 / h, 2 * k, k, N))
    D0 = cp.asarray(diag_circulant_complex(fd_stencil_0, 2 * k, k, N))

    D01 = cp.tile(D1, N*N)
    D02 = cp.tile(cp.repeat(D1, N), N)
    D03 = cp.repeat(D1, N*N)
    if alpha is None:
        Di = cp.concatenate((cp.tile(D0, N*N), cp.tile(cp.repeat(D0, N), N), cp.repeat(D0, N*N)))
        D = cp.concatenate((CT[0][0]*D01+CT[0][1]*D02+CT[0][2]*D03,\
                            CT[1][0]*D01+CT[1][1]*D02+CT[1][2]*D03,\
                            CT[2][0]*D01+CT[2][1]*D02+CT[2][2]*D03))
            
        return D, Di
    else:
        alpha_ = alpha/scal
        A = cp.concatenate((
                CT[0][0]*D01+CT[0][1]*D02+CT[0][2]*D03+1j*alpha_[0]*cp.tile(D0, N*N),\
                CT[1][0]*D01+CT[1][1]*D02+CT[1][2]*D03+1j*alpha_[1]*cp.tile(cp.repeat(D0, N), N),\
                CT[2][0]*D01+CT[2][1]*D02+CT[2][2]*D03+1j*alpha_[2]*cp.repeat(D0, N*N)))

        B = (cp.concatenate(((A[0:n]*A[0:n].conj()).real,(A[n:2*n]*A[n:2*n].conj()).real,(A[2*n:]*A[2*n:].conj()).real)).astype(float),\
             cp.concatenate((A[0:n].conj()*A[n:2*n],A[0:n].conj()*A[2*n:],A[n:2*n].conj()*A[2*n:])))
    
        return A, B
    
"""
Dielectric Matrix Function Handle.
"""

def chiral_handle(n, d_flag, eps_opt = 0, k = None):

    ind_e = diel.diel_io_index(n, d_flag, dofs = "edge")

    if eps_opt is None or eps_opt == 0:
        eps1 = diel.diel_chiral_const(d_flag)
    else:
        eps1 = eps_opt

    def Diel0(x_in):
        x = x_in.copy()
        x[ind_e] /= eps1
        return x 
    
    return Diel0

def pseudochiral_trivial_handle(n, d_flag = SC_C, eps_opt = 0, eps_mat = None, k = None, flag_mat = False):

    """
    Usage:
        Pseudochiral D-matrix with trivial cross-dof coupling.
        Return either function handle or vsr matrix.
    """

    nn = n * n * n
    if eps_mat is None:
        eps_loc = diel.diel_pseudochiral_const(eps_opt) / diel.diel_chiral_const(d_flag)
    else:
        eps_loc = eps_mat # No need to copy.

    t_h = time()

    # Load volume and edge dofs.
    ind_e = cp.asarray(diel.diel_io_index(n, d_flag, dofs = "edge"))
    ind_v = cp.asarray(diel.diel_io_index(n, d_flag, dofs = "volume"))
    ones_v = cp.ones(len(ind_v))

    diags = cp.ones(3*nn)
    diags[ind_e[ind_e < nn]] = eps_loc[0].real
    diags[ind_e[(ind_e >= nn) & (ind_e < 2*nn)]] = eps_loc[1].real
    diags[ind_e[ind_e >= 2*nn]] = eps_loc[2].real

    dmat = cp.sparse.coo_matrix((cp.concatenate((diags, ones_v*eps_loc[3], ones_v*eps_loc[4], ones_v*eps_loc[5], \
                                                 ones_v*eps_loc[3].conj(), ones_v*eps_loc[4].conj(), ones_v*eps_loc[5].conj())),\
                (cp.concatenate((cp.arange(3*nn), ind_v, ind_v, ind_v+nn, ind_v+nn, ind_v+2*nn, ind_v+2*nn)),\
                 cp.concatenate((cp.arange(3*nn), ind_v+nn, ind_v+2*nn, ind_v+2*nn, ind_v, ind_v, ind_v+nn)))), shape=(3*nn, 3*nn)).tocsr()
    
    t_o = owari()
    print(f"D-matrix is generated, runtime = {t_o - t_h:<6.3f}s.")
    return dmat if flag_mat else lambda x: dmat @ x 

def pseudochiral_crossdof_handle(n, d_flag = SC_C, eps_opt = 0, eps_mat = None, k = 1, flag_mat = False):
    """
    Usage:
        Pseudochiral D-matrix with cross-dof coupling.
        Return either function handle or vsr matrix.
    """

    nn = n*n*n
    if eps_mat is None:
        eps_loc = diel.diel_pseudochiral_const(eps_opt) / diel.diel_chiral_const(d_flag)
    else:
        eps_loc = eps_mat # No need to copy.

    t_h = time()
    ind_e = diel.diel_io_index(n, d_flag, dofs = "edge")

    diags = cp.ones(3*nn)
    diags[ind_e[ind_e < nn]] = eps_loc[0].real
    diags[ind_e[(ind_e >= nn) & (ind_e < 2*nn)]] = eps_loc[1].real
    diags[ind_e[ind_e >= 2*nn]] = eps_loc[2].real

    dx, dy, dz = ind_e[ind_e < nn], ind_e[(ind_e >= nn) & (ind_e < 2*nn)] - nn, ind_e[ind_e >= 2*nn] - 2*nn

    # COO of circulant matrix c.
    r_tmp = cp.repeat(cp.arange(n), 2*k)
    c = (r_tmp, (cp.tile(cp.arange(1-k,1+k),n) + r_tmp)%n, cp.tile(mfd_stencil(k,0), n))
    c_T = (c[1], c[0], c[2])
    del r_tmp

    t12_tuple = sparse_kron(sparse_kron(c, c_T), n)
    t13_tuple = sparse_kron(sparse_kron(c, n), c_T)
    t23_tuple = sparse_kron(sparse_kron(n, c), c_T)

    S12_l, S_12_r = P01(t12_tuple, dx), P01(t12_tuple, dy, flag_left=False)
    S13_l, S_13_r = P01(t13_tuple, dx), P01(t13_tuple, dz, flag_left=False)
    S23_l, S_23_r = P01(t23_tuple, dy), P01(t23_tuple, dz, flag_left=False)

    r12, r13, r23 = cp.concatenate((S12_l[0], S_12_r[0])), cp.concatenate((S13_l[0], S_13_r[0])), cp.concatenate((S23_l[0], S_23_r[0]))
    c12, c13, c23 = cp.concatenate((S12_l[1], S_12_r[1])), cp.concatenate((S13_l[1], S_13_r[1])), cp.concatenate((S23_l[1], S_23_r[1]))
    v12, v13, v23 = cp.concatenate((S12_l[2], S_12_r[2])) / 2,\
                    cp.concatenate((S13_l[2], S_13_r[2])) / 2,\
                    cp.concatenate((S23_l[2], S_23_r[2])) / 2
    
    dmat = cp.sparse.coo_matrix((cp.concatenate((diags, v12*eps_loc[3], v13*eps_loc[4], v23*eps_loc[5], \
                                 v12*eps_loc[3].conj(), v13*eps_loc[4].conj(), v23*eps_loc[5].conj())),
                                (cp.concatenate((cp.arange(3*nn), r12, r13, r23+nn, c12+nn, c13+2*nn, c23+2*nn)), 
                                 cp.concatenate((cp.arange(3*nn), c12+nn, c13+2*nn, c23+2*nn, r12, r13, r23+nn)))), shape=(3*nn, 3*nn)).tocsr()
    
    t_o = owari()
    print(f"D-matrix (her) is generated, runtime = {t_o - t_h:<6.3f}s.")
    return dmat if flag_mat else lambda x: dmat @ x 


# Unused.
def pseudochiral_crossdof_malposition_handle(n, d_flag = SC_C, eps_opt = 0, eps_mat = None, k = 1, flag_mat = False):
    """
    Usage:
        Pseudochiral D-matrix with cross-dof coupling (malposition form).
        Return either function handle or vsr matrix.
    """

    nn = n * n * n
    if eps_mat is None:
        eps_loc = diel.diel_pseudochiral_const(eps_opt) / diel.diel_chiral_const(d_flag)
    else:
        eps_loc = eps_mat # No need to copy.

    t_h = time()
    ind_e = diel.diel_io_index(n, d_flag, dofs = "edge")
    ind_v = diel.diel_io_index(n, d_flag, dofs = "volume")

    diags = cp.ones(3*nn)
    diags[ind_e[ind_e < nn]] = eps_loc[0].real
    diags[ind_e[(ind_e >= nn) & (ind_e < 2*nn)]] = eps_loc[1].real
    diags[ind_e[ind_e >= 2*nn]] = eps_loc[2].real

    # COO of circulant matrix c.
    r_tmp = cp.repeat(cp.arange(n), 2*k)
    c = (r_tmp, (cp.tile(cp.arange(1-k,1+k),n) + r_tmp)%n, cp.tile(mfd_stencil(k,0), n))
    c_T = (c[1], c[0], c[2])
    del r_tmp
    
    r12, c12, v12 = P01(sparse_kron(sparse_kron(c, c_T), n), ind_v)
    r13, c13, v13 = P01(sparse_kron(sparse_kron(c, n), c_T), ind_v)
    r23, c23, v23 = P01(sparse_kron(sparse_kron(n, c), c_T), ind_v)

    dmat = cp.sparse.coo_matrix((cp.concatenate((diags, v12*eps_loc[3], v13*eps_loc[4], v23*eps_loc[5], \
                                                 v12*eps_loc[3].conj(), v13*eps_loc[4].conj(), v23*eps_loc[5].conj())),\
                (cp.concatenate((cp.arange(3*nn), r12, r13, r23+nn, c12+nn, c13+2*nn, c23+2*nn)),\
                 cp.concatenate((cp.arange(3*nn), c12+nn, c13+2*nn, c23+2*nn, r12, r13, r23+nn)))), shape=(3*nn, 3*nn)).tocsr()

    t_o = owari()
    print(f"D-matrix (mal) is generated, runtime = {t_o - t_h:<6.3f}s.")
    return dmat if flag_mat else lambda x: dmat @ x 



