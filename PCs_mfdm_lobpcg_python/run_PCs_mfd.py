# -*- coding: utf-8 -*-
#
# Created on 2024-04-28 (Sunday) at 16:52:44
#
# Author: Epsilon-79th
#
# Usage: Basic operations and debugging for calculations of photonic crystal bandgaps.
#
# eigen_1p: compute frequencies of a_fft single lattice point.
# bandgap:  compute certain band info (default: all).
# tol_cmp:  variable tolerance.
# scal_cmp: variable scalings.
# pnt_cmp:  variable penalty coefficients.
# rela_cmp: variable relaxation.
# pack_cmp: variable packages (numpy,scipy,cupy,cupyx).
# grid_cmp: variable grid sizes.
# eps_cmp:  variable epsilons, dielectric coefficients.


# Packages.
from time import time
import os,json

import numpy as np
import cupy as cp

import discretization as mfd
import dielectric as diel

from eigen_solver import lobpcg_PCs_mfd

from numpy import pi
from numpy.random import rand
from gpu_opts import norm,owari_cuda

owari=owari_cuda

# Global parameters.
EPS = 13                      # Dielectric constant.
K = 1                         # Stencil length.
M_CONV = 10                   # Desired eigenpairs.
SCAL = 1                      # SCALing coefficient.
TOL = 1e-4                    # Tolerance.

"""
Part 0: Initialization, print.
"""


def uniform_initialization(n,d_flag_name,alpha,gpu_opt=True):
    
    """
    Usage:
        Trivial operations in matrix assembling.
    
    Input:
        n:           grid size.
        d_flag_name: name of lattice type.
        alpha:       lattice vector.
        gpu_opt:     gpu option (default==True).
    
    Output:
        a_fft:   fft diagonal blocks of A.
        b_fft:   fft diagonal blocks of B'B.
        diels:   dielectric indexes and constant.
        inv_fft: fft diagonal blocks of inv(AA'+pnt B'B).
        x0:      initial guess.
        shift:   ensuring the insingularity of the system.
        
    """
    
    t_h = time()

    relax_opt,pnt = mfd.set_relaxation(n,alpha)
    diel_ind,ct, __ = diel.dielectric_save_and_load(n,d_flag_name)
    a_fft,b_fft = mfd.fft_blocks(n,K,ct,alpha)
    inv_fft = mfd.inverse_3_times_3_B(b_fft,pnt,relax_opt[0])

    a_fft /= SCAL
    b_fft = (pnt * b_fft[0] / SCAL / SCAL,pnt * b_fft[1] / SCAL / SCAL)
    inv_fft = (inv_fft[0] * SCAL * SCAL,inv_fft[1] * SCAL * SCAL)
    m = round(M_CONV * relax_opt[1]) + M_CONV
    x0 = rand(3 * n ** 3,m)
    
    if gpu_opt:
        a_fft,b_fft,inv_fft,x0 = cp.asarray(a_fft),cp.asarray(b_fft),\
                                 cp.asarray(inv_fft),cp.asarray(x0)
    
    t_o = owari()
    print("Matrix blocks done, ", '%6.3f' % (t_o - t_h), "s elapsed.")
    
    return a_fft, b_fft, [diel_ind, 1 / EPS], inv_fft, x0, relax_opt[0]


def print_and_normalize(lambda_pnt,lambda_re,scal=SCAL):
    
    """
    Print and return normalized frequencies.
    """
    
    n_lambdas=len(lambda_pnt)
    lambda_pnt=scal*np.sqrt(lambda_pnt)
    
    if len(lambda_re)!=n_lambdas:
        ValueError("Number of penalized and recomputed eigenvalues doesn't match")
    lambda_re=scal*np.sqrt(lambda_re)
    for i in range(n_lambdas):
        print("i=",i+1,", lambda_pnt=",'%10.6f'%lambda_pnt[i],\
              ", lambda_re=",'%10.6f'%lambda_re[i],\
              ", deviation=",'%6.3e'%(abs(lambda_pnt[i]-lambda_re[i])))
    return lambda_pnt,lambda_re


"""
Part I: Compute a_fft single lattice point, certain lattice points (bandgap).

ALL functions return None.
"""


def eigen_1p(n,d_flag_name,alpha):
    
    """
    Usage:
        Compute eigenvalues at one SINGLE lattice vector.
    
    Input:
        n:           grid size.
        d_flag_name: name of diel type.
        alpha:       lattice vector.
    """

    a_fft,b_fft,diels,inv_fft,x0,shift=uniform_initialization(n,d_flag_name,alpha)

    lambda_pnt,lambda_re,__,eigvec=lobpcg_PCs_mfd(a_fft,b_fft,diels,\
            inv_fft,x0,M_CONV,tol=TOL/SCAL/SCAL,shift=shift)
    del x0
    
    print("n=",n,", lattice type:",d_flag_name,", alpha=[",\
          '%5.2f'%(alpha[0]/pi),'%5.2f'%(alpha[1]/pi),'%5.2f'%(alpha[2]/pi),\
          "] pi.")
    
    # Print, residual.
    lambdas=cp.asarray(lambda_re)
    R=mfd.A_fft(mfd.scalar_prod(mfd.A_fft(eigvec[:,:M_CONV],-a_fft.conj()),diels),a_fft)-eigvec[:,:M_CONV]*lambdas[:M_CONV]
    for i in range(M_CONV):
        l1=SCAL*np.sqrt(lambda_pnt[i])
        l2=SCAL*np.sqrt(lambda_re[i])
        
        print(f"i={i + 1}, {l1:<10.6f}, {l2:<10.6f}, {abs(l1-l2):<6.3e}, res={norm(R[:,i])}.")
    return
  

def bandgap(n,d_flag_name,gap,indices=None):
    
    
    """
    Compute the complete bandgap w.r.t to a_fft given lattice material with 
    certain grid size. (uniformly tensor division).
    """
    
    nn=n**3
    diel_ind,ct,sym_points=diel.dielectric_save_and_load(n,d_flag_name)
    diels=[diel_ind,1/diel.eps_eg[d_flag_name]]
    del diel_ind
    
    d_fft,di_fft=mfd.fft_blocks(n,K,ct)
    n_pt,__=sym_points.shape
    n_pt-=1
    
    if not indices:
        # Default: compute the complete bandgap.
        indices=list(range(n_pt*gap))
    if max(indices)>=n_pt*gap or min(indices)<0:
        ValueError("Index is non-positive or is incompatible with gap size.")
    
    # Discrete lattice points.
    alphas=np.zeros((n_pt*gap,3))
    for i in range(n_pt):
        alphas[(i+1)*gap-1,:]=sym_points[i+1,:]
        for j in range(gap-1):
            alphas[i*gap+j,:]=((j+1)*sym_points[i+1,:]+(gap-j-1)*sym_points[i,:])/gap
            
    # Preloading.
   
    if not os.path.exists("output"):
        os.mkdir("output")
    
    # Filename:
    bandgap_name="output/bandgap_"+d_flag_name+".json"
    
    # Keywords:
    var_name_it=d_flag_name+"_"+str(n)+"_iterations"
    var_name_fq=d_flag_name+"_"+str(n)+"_frequencies"
    
    if not os.path.exists(bandgap_name):
        # New lattice bandgap plot.
        
        print("The bandgap of type ",d_flag_name," has no previous record.")
        
        gap_rec_it,gap_rec_fq=[[0]*2]*(n_pt*gap),[[0]*M_CONV]*(n_pt*gap)
        gap_lib={var_name_it:gap_rec_it,\
                 var_name_fq:gap_rec_fq}
        
        with open(bandgap_name,'w') as file:
            json.dump(gap_lib,file,indent=4)
    else:
        with open(bandgap_name,'r') as file:
            gap_lib=json.load(file)
        if var_name_it in gap_lib.keys():
            # Previous record exists.
            
            print("Lattice type ",d_flag_name," with grid size n=",n,\
                  " has a_fft previous record.")     
                   
            gap_rec_it,gap_rec_fq=gap_lib[var_name_it],gap_lib[var_name_fq]
        else:
            # New grid size.
            
            print("Lattice type ",d_flag_name,\
                  " will be computed with a_fft new grid size n=",n,".")
            
            gap_rec_it,gap_rec_fq=[[0]*2]*(n_pt*gap),[[0]*M_CONV]*(n_pt*gap)
            gap_lib[var_name_it],gap_lib[var_name_fq]=gap_rec_it,gap_rec_fq
            
            with open(bandgap_name,'w') as file:
                json.dump(gap_lib,file,indent=4)
    
    
    """
    Calculation of bandgap.
    """
    
    x0=cp.random.rand(3*nn,M_CONV+max(10,M_CONV))+\
        1j*cp.random.rand(3*nn,M_CONV+max(10,M_CONV))
    
    # Main Loop: compute each lattice point.    
        
    for i in range(len(indices)):
        t_h=time()
        alpha=alphas[indices[i]]/(2*pi)
       
        relax_opt,pnt=mfd.set_relaxation(n,alpha)
        m=M_CONV+round(M_CONV*relax_opt[1])
        
        a_fft=np.hstack((d_fft[:nn]+1j*alpha[0]*di_fft[:nn],\
                         d_fft[nn:2*nn]+1j*alpha[1]*di_fft[nn:2*nn],\
                         d_fft[2*nn:]+1j*alpha[2]*di_fft[2*nn:] ))
        b_fft=(np.hstack(((a_fft[0:nn]*a_fft[0:nn].conj()).real,\
                          (a_fft[nn:2*nn]*a_fft[nn:2*nn].conj()).real,\
                          (a_fft[2*nn:]*a_fft[2*nn:].conj()).real)),\
           np.hstack((a_fft[0:nn].conj()*a_fft[nn:2*nn],\
                      a_fft[0:nn].conj()*a_fft[2*nn:],a_fft[nn:2*nn].conj()*a_fft[2*nn:])))
        inv_fft=mfd.inverse_3_times_3_B(b_fft,pnt,relax_opt[0])
        
        a_fft/=SCAL
        b_fft=(pnt*b_fft[0]/SCAL/SCAL,pnt*b_fft[1]/SCAL/SCAL)
        inv_fft=(inv_fft[0]*SCAL*SCAL,inv_fft[1]*SCAL*SCAL)
        
        a_fft,b_fft,inv_fft=cp.asarray(a_fft),cp.asarray(b_fft),cp.asarray(inv_fft)
        t_o=time()
        print("Matrix blocks done, ",'%6.3f'%(t_o-t_h),"s elapsed.")
        
        lambda_pnt,lambda_re,iters,x0=lobpcg_PCs_mfd\
            (a_fft,b_fft,diels,inv_fft,x0[:,:m],M_CONV,shift=relax_opt[0],tol=TOL/SCAL/SCAL)        
        
        print("Gap index ",indices[i]+1," out of ",n_pt*gap," (",d_flag_name,") is computed.")
        print("Iterations=",iters[0],", runtime=",'%6.3f'%iters[1],"s.\n")
        __,lambda_re=print_and_normalize(lambda_pnt,lambda_re)    

        gap_rec_it[indices[i]]=iters.tolist()
        gap_rec_fq[indices[i]]=lambda_re.tolist()                
        
        t_h=time()
        gap_lib[var_name_it],gap_lib[var_name_fq]=gap_rec_it,gap_rec_fq
        
        with open(bandgap_name,'w') as file:
            json.dump(gap_lib,file,indent=4)
            
        t_o=owari()
        print("Gap info library (",d_flag_name,") is updated, time=",\
              '%6.3f'%(t_o-t_h),"s.")
    return

     
"""
Part II: Control experiment.
"""
 
# Control experiment.
# Factor: tolerance (tol), SCALing coefficient (SCAL), grid size (n),
#         linear algebra package (pack), dielectric coefficient (eps).
#         penalty number (pnt), relaxation test (rela)
    
def tol_cmp(n,d_flag_name,tols,alpha=np.array([pi,pi,pi])):
    
    """
    Usage:
        The effect of different thresholds on convergence.
    
    Input:
        n:           grid size.
        d_flag_name: name of diel type.
        tols:        various tolerances.
        alpha:       lattice vector.
    
    """

    a_fft,b_fft,diels,inv_fft,x0,shift=uniform_initialization(n,d_flag_name,alpha)

    n_tols=len(tols)
    lambda_pnt=np.zeros((n_tols,M_CONV))
    lambda_re=np.zeros((n_tols,M_CONV))
    iters=np.zeros((n_tols,2))
    
    for i in range(n_tols):
        lambda_pnt[i,:],lambda_re[i,:],iters[i,:],__=lobpcg_PCs_mfd\
            (a_fft,b_fft,diels,inv_fft,x0,M_CONV,tol=tols[i]/SCAL/SCAL,shift=shift)
            
        print("\ntol=",'%.2e'%tols[i]," is done computing.\n")
        lambda_pnt[i,:],lambda_re[i,:]=print_and_normalize(lambda_pnt[i,:],lambda_re[i,:])        
        
    std_pnt=np.std(lambda_pnt,axis=0)
    std_re=np.std(lambda_re,axis=0)
    print("Tolerance:")
    for i in range(n_tols):
        print("tol=",'%5.2e'%tols[i],", iterations=",iters[i,0],", time=",\
              '%5.2f'%iters[i,1],"s.")
    
    print("\nStandard deviation of each eigenvalue:")
    for i in range(M_CONV):
        print("i=",'%4d'%(i+1),"std_pnt=",'%7.3e'%std_pnt[i],",std_re=",'%7.3e'%std_re[i])
     
    return

def pnt_cmp(n,d_flag_name,pnts,alpha=np.array([pi,pi,pi])):
    # Increase default penalty by ratio n^pnts[i].
    
    a_fft,b_fft,diels,__,x0,shift=uniform_initialization(n,d_flag_name,alpha)
    __,pnt0=mfd.set_relaxation(n,alpha)
    b_fft=(b_fft[0]/pnt0,b_fft[1]/pnt0)

    n_pnts=len(pnts)
    lambda_pnt=np.zeros((n_pnts,M_CONV))
    lambda_re=np.zeros((n_pnts,M_CONV))
    iters=np.zeros((n_pnts,2))
    
    for i in range(n_pnts):
        pnt=2*n**(pnts[i]+1.0)
        inv_fft=mfd.inverse_3_times_3_B(b_fft,pnt,shift)

        lambda_pnt[i,:],lambda_re[i,:],iters[i,:],__=lobpcg_PCs_mfd\
            (a_fft,(b_fft[0]*pnt,b_fft[1]*pnt),diels,inv_fft,x0,M_CONV,shift=shift)
        
        print("\npnt=n^",'%.2f'%pnts[i]," is done computing.\n")
        lambda_pnt[i,:],lambda_re[i,:]=print_and_normalize(lambda_pnt[i,:],lambda_re[i,:])
        
    std_pnt=np.std(lambda_pnt,axis=0)
    std_re=np.std(lambda_re,axis=0)
    print("Penalties:")
    for i in range(n_pnts):
        print("pnt=",'%5.2e'%pnts[i],", iterations=",iters[i,0],", time=",\
              '%5.2f'%iters[i,1],"s.")
    
    print("\nStandard deviation of each eigenvalue:")
    for i in range(M_CONV):
        print("i=",'%4d'%(i+1),"std_pnt=",'%7.3e'%std_pnt[i],",std_re=",'%7.3e'%std_re[i])
    return


def rela_cmp(n,d_flag_name,relas,alpha=np.array([pi,pi,pi])):
    
    """
    Usage:
        The effect of different relaxation ratios on convergence.
    
    Input:
        n:           grid size.
        d_flag_name: name of diel type.
        relas:       various tolerances and relaxation ratio.
        alpha:       lattice vector.
    
    """
    
    a_fft,b_fft,diels,inv_fft,__=uniform_initialization(n,d_flag_name,alpha)
    
    nn = n ** 3
    tols=relas[0]
    rels=relas[1]
    n_relas=len(tols)
    lambda_pnt=np.zeros((n_relas,M_CONV))
    lambda_re=np.zeros((n_relas,M_CONV))
    iters=np.zeros((n_relas,2))
    
    if n_relas!=len(rels):
        ValueError("Tuple relas should contain two arrays with same length.")
    
    for i in range(n_relas):
        m=M_CONV+round(M_CONV*rels[i])
        x0=cp.random.rand(3*nn,m)
        lambda_pnt[i,:],lambda_re[i,:],iters[i,:],__=lobpcg_PCs_mfd\
            (a_fft,b_fft,diels,inv_fft,x0,M_CONV,tol=tols[i])
        lambda_pnt[i,:],lambda_re[i,:]=print_and_normalize(lambda_pnt[i,:],lambda_re[i,:])
            
    return

def scal_cmp(n,d_flag_name,scals,alpha=np.array([pi,pi,pi])):
    
    """
    Usage:
        The effect of different saclings on convergence.
    
    Input:
        n:           grid size.
        d_flag_name: name of diel type.
        scals:       various scalings.
        alpha:       lattice vector.
        
    """

    a_fft,b_fft,diels,inv_fft,x0,options_lobpcg=uniform_initialization(n,d_flag_name,alpha)

    n_scals=len(scals)
    lambda_pnt=np.zeros((n_scals,M_CONV))
    lambda_re=np.zeros((n_scals,M_CONV))
    iters=np.zeros((n_scals,2))
    
    for i in range(n_scals):
        scal0=(n**scals[i])**2

        lambda_pnt[i,:],lambda_re[i,:],iters[i,:],__=lobpcg_PCs_mfd\
            (a_fft/np.sqrt(scal0),(b_fft[0]/scal0,b_fft[1]/scal0),diels,(inv_fft[0]*scal0,inv_fft[1]*scal0),\
             x0,M_CONV,tol=TOL/scal0)

        print("\nscal=n^",'%.2f'%scals[i]," is done computing.\n")
        lambda_pnt[i,:],lambda_re[i,:]=print_and_normalize(lambda_pnt[i,:],lambda_re[i,:],np.sqrt(scal0))
     
    std_pnt=np.std(lambda_pnt,axis=0)
    std_re=np.std(lambda_re,axis=0)
    print("Scaling power:")
    for i in range(n_scals):
        print("SCAL=",'%5.2f'%scals[i],", iterations=",int(iters[i,0]),", time=",\
              '%5.2f'%iters[i,1],"s.")
    
    print("\nStandard deviation of each eigenvalue:")
    for i in range(M_CONV):
        print("i=",'%4d'%(i+1),"std_pnt=",'%7.3e'%std_pnt[i],",std_re=",'%7.3e'%std_re[i])
    
    return

def eps_cmp(n,d_flag_name,epss,alpha=np.array([pi,pi,pi])):
    
    """
    Usage:
        The effect of different diel constants on convergence.
    
    Input:
        n:           grid size.
        d_flag_name: name of diel type.
        eps:         various diel constants.
        alpha:       lattice vector.
    
    """

    a_fft,b_fft,diels,inv_fft,x0,shift=uniform_initialization(n,d_flag_name,alpha)
    
    n_epss=len(epss)
    lambda_pnt=np.zeros((n_epss,M_CONV))
    lambda_re=np.zeros((n_epss,M_CONV))
    iters=np.zeros((n_epss,2))
    
    for i in range(n_epss):
        diels=(diels[0],1/epss[i])
        lambda_pnt[i,:],lambda_re[i,:],iters[i,:],__=lobpcg_PCs_mfd\
            (a_fft,b_fft,diels,inv_fft,x0,M_CONV,shift=shift)
        
        print("\neps=",'%.2f'%epss[i]," is done computing.\n")
        lambda_pnt[i,:],lambda_re[i,:]=print_and_normalize(lambda_pnt[i,:],lambda_re[i,:])
    return
    
def grid_cmp(Ns,d_flag_name,alpha=np.array([pi,pi,pi])):
    
    """
    Usage:
        The effect of different grid sizes on convergence.
    
    Input:
        n:           grid size.
        d_flag_name: name of diel type.
        relas:       various grid sizes.
        alpha:       lattice vector.
    
    """
    
    Ns.sort()

    n_Ns=len(Ns)
    lambda_pnt=np.zeros((n_Ns,M_CONV))
    lambda_re=np.zeros((n_Ns,M_CONV))
    iters=np.zeros((n_Ns,2))
    
    for i in range(n_Ns):
        a_fft,b_fft,diels,inv_fft,x0,shift=uniform_initialization(Ns[i],d_flag_name,alpha)
        lambda_pnt[i,:],lambda_re[i,:],iters[i,:],__=lobpcg_PCs_mfd\
            (a_fft,b_fft,diels,inv_fft,x0,M_CONV,shift=shift)

        print("\nGrid size=",Ns[i]," is done computing.\n")
        lambda_pnt[i,:],lambda_re[i,:]=print_and_normalize(lambda_pnt[i,:],lambda_re[i,:])

    print("Grid size:")
    for i in range(n_Ns):
        print("n=",Ns[i],", iterations=",int(iters[i,0]),", time=",\
              '%5.2f'%iters[i,1],"s.")
    
    print("\nDeviation (relative error):")
    
    for i in range(M_CONV):
        print("i=",i+1,end=":\t")
        for j in range(1,n_Ns):
            print('%7.3e'%abs(lambda_pnt[j,i]-lambda_pnt[j-1,i]),end=" ")
        print()
        
    return


def speedup(Ns,d_flag_name,alpha=np.array([pi,pi,pi])):
    
    """
    Cuda speedup test.
    """
    
    # Packs={gpu_pack, cpu_pack}. 
    # cpu: numpy,scipy. gpu: cupy,cupyx.
    
    runtime_name="output/speedup_"+d_flag_name+".json"
    
    if not os.path.exists(runtime_name):
        runtime_pack_lib={}
    else:
        with open(runtime_name,'r') as file:
            runtime_pack_lib=json.load(file)

    n_Ns=len(Ns)
    iters=np.zeros((n_Ns,2,2))
    
    for i in range(n_Ns):
        a_fft,b_fft,diels,inv_fft,x0,shift=uniform_initialization(Ns[i],d_flag_name,alpha)

        lambda_pnt,lambda_re,iters[i,0,:],__=lobpcg_PCs_mfd\
            (a_fft,b_fft,diels,inv_fft,x0,M_CONV,shift=shift)
        print("\nN=",Ns[i],", gpu (cupy/cupyx.scipy) is done computing, runtime=",\
              '%6.3f'%iters[i,0,1],"s.\n")
        __,__=print_and_normalize(lambda_pnt,lambda_re)
        
        a_fft,b_fft,inv_fft,x0=a_fft.get(),b_fft.get(),inv_fft.get(),x0.get()
        lambda_pnt,lambda_re,iters[i,1,:],__=lobpcg_PCs_mfd\
            (a_fft,b_fft,diels,inv_fft,x0,M_CONV,shift=shift)
        print("\nN=",Ns[i],", cpu (numpy/scipy) is done computing, runtime=",\
              '%6.3f'%iters[i,1,1],"s.\n")
        __,__=print_and_normalize(lambda_pnt,lambda_re)
        
        runtime_pack_lib["pack_cmp_"+str(Ns[i])]=\
            [iters[i,0,0],iters[i,1,1],iters[i,0,1],iters[i,1,1]/iters[i,0,1]]
            
        with open(runtime_name,'w') as file:
            json.dump(runtime_pack_lib,file,indent=4)

    print("\nTesting lattice type: ",d_flag_name)
    print("Runtime comparison using different linear algebra packages:")
    for i in range(n_Ns):
        print("n=",Ns[i],", iterations=",iters[i,0,0],", cputime=",\
              '%5.2f'%iters[i,1,1],"s, gputime=",'%5.2f'%iters[i,0,1],\
              "s, ratio=",'%5.2f'%(iters[i,1,1]/iters[i,0,1]),".")
      
    return


def main():
    
    """
    Main function.
    """
    
    eigen_1p(100,"sc_curv",np.array([pi,pi/2,0]))
    #tol_cmp(100,"sc_curv",[5e-4,1e-4,1e-5,1e-6])
    #pnt_cmp(100,"sc_curv",[0,1.2,1.5,2])        
    #scal_cmp(100,"sc_curv",[0,0.2,0.5,0.8,1])
    #grid_cmp([100,120,150],"sc_curv")
    #speedup([100,120,150],"sc_curv")
    #bandgap(100,'bcc_single_gyroid',20)
    
if __name__=="__main__":
    main()

    
