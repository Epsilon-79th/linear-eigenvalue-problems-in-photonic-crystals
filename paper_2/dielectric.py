# -*- coding: utf-8 -*-
"""

Dielectric coefficient.

"""

import os
import numpy as np
from numpy import pi
import cupy as cp

from environment import *

"""
    Part 1: File I/O.
"""

# Get dielectric info.
def diel_info(d_flag, option = None):
    """
    Usage:
        Get coordinate transform matrix (ct) 
        or symmetric points in Brillouin zone (sym)
        or both (default).
    """
    lattice_name = d_flag.split('_')[0]
    ct = np.array(DIEL_LIB["CT_"+lattice_name])
    sym = np.array(DIEL_LIB["sym_"+lattice_name])
    if option == 'ct':
        return ct
    elif option == 'sym':
        return sym

    return ct, sym

def diel_alpha(d_flag, no, gap = GAP):

    """
    Usage:
        Get alpha vector.
        Index no starts from 1 (no=0 is also allowed).
    """
    sym = np.array(DIEL_LIB["sym_"+d_flag.split('_')[0]])
    i0, j0 = no // gap, no % gap
    if j0 == 0:
        return sym[i0, :]
    else:
        return (j0 * sym[i0 + 1, :] + (gap - j0) * sym[i0, :]) / gap

def diel_chiral_const(d_flag = "sc_curv"):
    return CHIRAL_EPS_EG[d_flag]

def diel_pseudochiral_const(no = 0):
    return PSEUDOCHIRAL_EPS_LOC[no]

# Load dielectric indices.
def diel_io_index(N, d_flag, dofs = "edge", gpu = True):

    """
    Usage:
        Load the indices of dielectric edge/volume dofs. 
        If not exists, compute and save indices to folder "dielectric_examples".
    Inputs:     
        N (int): Number of points along each axis.
        d_flag (str): Name of the dielectric flag function.
        dofs (str): Type of dofs, either "edge" or "volume". Default is "edge".
        gpu (bool): Whether output array is on gpu.
    Outputs:  
        ind (np.ndarray): Indices of the dielectric dofs.
    """

    # Randomly generate indices if d_flag is not defined.
    if d_flag is None:
        print("Flag function is empty or not defined, dielectric indices will be randomly generated.")
        ind = cp.random.randint(3*N**3-1,size=(int(0.372*3*N**3)))
        return ind if gpu else ind.get()
    
    t_h = time.time()

    ct= diel_info(d_flag, option='ct')
    path = DIEL_PATH + dofs + "_dofs/" + d_flag + "_" + str(N) + ".bin"
    if not os.path.exists(path):
        # No previous record.
        print(f"{RED}New lattice type {d_flag} or size {N} isn't computed.{RESET}")
        ind = eval("FLAG_" + d_flag)(eval("mesh3d_" + dofs + "_dofs(N)") @ np.linalg.inv(ct.T))
        ind.tofile(path)
    else:
        print(f"{GREEN}Index file already exists.{RESET}")
        if gpu:
            ind = cp.fromfile(path, dtype=int)
        else:
            ind = np.fromfile(path, dtype=int)

    t_o = time.time()
    print(f"Dielectric {dofs} indices for {d_flag} with N = {N} loaded, {t_o-t_h:<6.3f}s elapsed.")
    return ind


"""
    Part 2: Mesh generation of Edge/Volume DoFs.
"""

def mesh3d_edge_dofs(N):
    """
    Generate indices for the edge dofs of a 3D mesh with N points along each axis.
    DoFs = 3N^3.
    """
    
    t_h = time.time()
    I,J,K = np.tile(np.arange(N), N * N), np.tile(np.repeat(np.arange(N), N),N), np.repeat(np.arange(N), N * N)
    dofs = np.vstack((np.column_stack(((I+0.5)/N, J/N, K/N)), 
                      np.column_stack((I/N, (J+0.5)/N, K/N)), 
                      np.column_stack((I/N, J/N, (K+0.5)/N))))
    t_o = time.time()
    print(f"Dielectric matrix done, {t_o-t_h:<6.3f}s elapsed.")
    return dofs
    
def mesh3d_volume_dofs(N):
    """
    Generate indices for the volume dofs of a 3D mesh with N points along each axis.
    DoFs = N^3.
    """
    
    t_h = time.time()
    I,J,K = np.tile(np.arange(N), N * N), np.tile(np.repeat(np.arange(N), N),N), np.repeat(np.arange(N), N * N)
    dofs = np.column_stack(((I+0.5)/N, (J+0.5)/N, (K+0.5)/N))
    t_o = time.time()
    print(f"Volume dofs done, {t_o-t_h:<6.3f}s elapsed.")
    return dofs

def mesh3d_offdiagonal_dofs(N, d_flag):

    # Get volume indices.
    v_dofs = diel_io_index(N, d_flag, dofs="volume")[0]

    # Indices to (I,J,K).
    K = v_dofs // (N*N)
    J = v_dofs % (N*N) // N
    I = v_dofs % N

    # (I,J,K) to indices handle.
    ijk2ind = lambda i,j,k: i%N + (j%N)*N + (k%N)*N*N

    # Related edge DoFs.
    x_dofs = np.concatenate((v_dofs, ijk2ind(I, J+1, K), ijk2ind(I, J, K+1), ijk2ind(I, J+1, K+1)))
    y_dofs = np.concatenate((v_dofs, ijk2ind(I+1, J, K), ijk2ind(I, J, K+1), ijk2ind(I+1, J, K+1)))# + N*N*N
    z_dofs = np.concatenate((v_dofs, ijk2ind(I+1, J, K), ijk2ind(I, J+1, K), ijk2ind(I+1, J+1, K)))# + 2*N*N*N

    return x_dofs, y_dofs, z_dofs
    

"""
    Part 3: Flag functions.
"""

def FLAG_sc_flat1(coo):
    ind = np.where((coo[:, 0] <= 0.25) & (coo[:, 1] <= 0.25) |
                   (coo[:, 0] <= 0.25) & (coo[:, 2] <= 0.25) |
                   (coo[:, 1] <= 0.25) & (coo[:, 2] <= 0.25))[0]

    return ind

def FLAG_sc_flat2(coo):
    ind = np.where((coo[:, 0] <= 0.25) & (coo[:, 1] <= 0.25) |
                   (coo[:, 0] <= 0.25) & (coo[:, 2] >= 0.25) & (coo[:, 2] <= 0.5) |
                   (coo[:, 1] >= 0.5) & (coo[:, 1] <= 0.75) & (coo[:, 2] >= 0.5) & (coo[:, 2] <= 0.75) |
                   (coo[:, 0] >= 0.5) & (coo[:, 0] <= 0.75) & (coo[:, 2] >= 0.75))[0]

    return ind


def FLAG_sc_curv(coo_in):
    r1, R1 = 0.11, 0.345
    coo = coo_in - 0.5
    ind = np.where(( (coo[:, 0]**2 + coo[:, 1]**2 + coo[:, 2]**2 <= R1**2) |
                     (coo[:, 0]**2 + coo[:, 1]**2 <= r1**2) |
                     (coo[:, 0]**2 + coo[:, 2]**2 <= r1**2) |
                     (coo[:, 1]**2 + coo[:, 2]**2 <= r1**2)))[0]

    return ind

FLAG_bcc_sg = lambda coo: FLAG_bcc_gyroid(coo, double_flag=False)
FLAG_bcc_dg = lambda coo: FLAG_bcc_gyroid(coo, double_flag=True)

def FLAG_bcc_gyroid(coo, double_flag=False):
    def g(r):
        return np.sin(2 * pi * r[0]) * np.cos(2 * pi * r[1]) + \
               np.sin(2 * pi * r[1]) * np.cos(2 * pi * r[2]) + \
               np.sin(2 * pi * r[2]) * np.cos(2 * pi * r[0])

    if double_flag:
        # Double gyroid.
        ind = np.where(np.abs(g(coo.T)) > 1.1)[0]
    else:
        # Single gyroid.
        ind = np.where(g(coo.T) > 1.1)[0]

    return ind

def FLAG_fcc(coo_in, lib=np):
    # Parameters.
    r = 0.12
    b_val = 0.11

    # Type check for library.
    if lib.__name__ == 'cupy':
        if not isinstance(coo_in, cp.ndarray):
            coo = cp.array(coo_in.copy())
        else:
            coo = coo_in.copy()
    else:
        if not isinstance(coo_in, np.ndarray):
            coo = np.array(coo_in.copy())
        else:
            coo = coo_in.copy()

    if coo.ndim == 1:
        coo = coo.reshape(3, 1)
    elif coo.shape[0] != 3:
        coo = coo.T  # shape = (3, N).

    a = lib.array([[0,0,0.5,0.5],[0,0.5,0,0.5],[0,0.5,0.5,0]], dtype=coo.dtype)
    cnt = lib.ones(3, dtype=coo.dtype) * 0.25

    tran1 = lib.hstack((lib.array([
        [0, 0, 0],[1, 0, 0],[0, 1, 0],
        [0, 0, 1],[0, 1, 1],[1, 0, 1],
        [1, 1, 0],[1, 1, 1],[0, 0.5, 0.5],
        [0.5, 0, 0.5],[0.5, 0.5, 0],
        [1, 0.5, 0.5],[0.5, 1, 0.5],
        [0.5, 0.5, 1]], dtype=coo.dtype).T,
        cnt[:,lib.newaxis] + a)
    )

    # Compute o1-d4
    o = lib.zeros((3, 4), dtype=coo.dtype)
    d = lib.zeros((3, 4), dtype=coo.dtype)
    c = lib.zeros(4, dtype=coo.dtype)
    for i in range(4):
        o[:, i] = (a[:, i] + cnt) / 2
        d[:, i] = (a[:, i] - cnt) / 2
        c[i] = lib.linalg.norm(d[:, i])
        d[:, i] /= c[i]

    # Sphere condition.
    condition_sphere = lib.any(lib.sum((coo[:,:,lib.newaxis] - tran1[:,lib.newaxis,:])**2, axis=0) < r*r, axis=1)

    def ell(x, cnt, b, c, d, tran):
        X = x[:,lib.newaxis,:] - (cnt[:,lib.newaxis] + tran)[:,:,lib.newaxis] # (3,4,N)
        a_val = lib.hypot(b, c)
        L1 = lib.tensordot(d, X, axes=([0],[0])) ** 2
        L2 = lib.sum(X**2, axis=0) - L1
        condition = (L1 / a_val**2) + (L2 / b**2) < 1
        return lib.any(condition, axis=0)

    ells = lib.empty((coo.shape[1], 4), dtype=lib.bool_)
    for i in range(4):
        ells[:, i] = ell(coo, o[:, i], b_val, c[i], d[:, i], a)

    return lib.where(condition_sphere | lib.any(ells, axis=1))[0]


"""
    Test and Main Function.
"""

def test():
    ind_sc = diel_io_index(100, "sc_curv", dofs = "edge")
    print(len(ind_sc))

def main():

    test()

    return

if __name__ == "__main__":
    main()