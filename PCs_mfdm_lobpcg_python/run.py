# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:52:44 2024

@author: 11034
"""

# Packages.
import numpy as np
import cupy as cp

import os
import json

import discretization as mfd
import dielectric as diel
from time import time
import eigen_lobpcg

from numpy import pi
from numpy.random import rand

"""
    Usage.
"""
    
# eigen_1p: compute frequencies of a single lattice point.
# bandgap:  compute certain band info (default: all).
# tol_cmp:  variable tolerance.
# scal_cmp: variable scalings.
# pnt_cmp:  variable penalty coefficients.
# rela_cmp: variable relaxation.
# pack_cmp: variable packages (numpy,scipy,cupy,cupyx).
# grid_cmp: variable grid sizes.    


# Information of each lattice point: eigenvalues and iterations.
class bandgap_info:
    def __init__(self,eigen,iters):
        self.eigen=eigen
        self.iters=iters        
    
# Information of runtime and acceleration.
class runtime_info:
    def __init__(self,iters,cputime,gputime):
        self.iters=iters
        self.cputime=cputime
        self.gputime=gputime
        self.ratio=cputime/gputime

# Global parameters.
eps=13                      # Dielectric constant.
k=1                         # Stencil length.
m_conv=15                   # Desired eigenpairs.
scal=1                      # Scaling coefficient.
tol=1e-4                    # Tolerance.
pack_name="cpx"             # Linear algebra package.
NP=eval(pack_name[0:2])

"""

Part 0: Initialization, print.

"""

# Trivial operations in matrix assembling.
def uniform_initialization(N,d_flag_name,alpha):
    
    t_h=time()    
    
    relax_opt,pnt=mfd.set_relaxation(N,alpha)
    M_ind,CT,__=diel.dielectric_save_and_load(N,d_flag_name)
    #print("Number of d_indexes=",len(M_ind))
        
    A,B=mfd.fft_blocks(N,k,CT,alpha)
    INV=mfd.inverse_3_times_3_B(B,pnt)
    
    A/=scal
    B=(pnt*B[0]/scal/scal,pnt*B[1]/scal/scal)
    INV=(INV[0]*scal*scal,INV[1]*scal*scal)
    
    options_lobpcg={"shift":relax_opt[0],\
                    "pack_name":pack_name,\
                    "iter_max":1000,\
                    "tol":tol/scal/scal}
    
    m=round(m_conv*relax_opt[1])+m_conv
    X0=rand(3*N**3,m)
    t_o=time()
    print("Matrix blocks done, ",'%6.3f'%(t_o-t_h),"s elapsed.")
    
    return A,B,(M_ind,1/eps),INV,X0,options_lobpcg

# Print and return normalized frequencies.
def print_and_normalize(lambda_pnt,lambda_re=None):
    
    n_lambdas=len(lambda_pnt)
    lambda_pnt=scal*np.sqrt(lambda_pnt)
    
    if lambda_re is None:
        # No recomputing process involved.
        for i in range(n_lambdas):
            print("i=",i+1,", frequency=",'%10.6f'%lambda_pnt[i])
        return lambda_pnt
    else:
        if len(lambda_re)!=n_lambdas:
            ValueError("Number of penalized and recomputed eigenvalues doesn't match")
        lambda_re=scal*np.sqrt(lambda_re)
        for i in range(n_lambdas):
            print("i=",i+1,", lambda_pnt=",'%10.6f'%lambda_pnt[i],\
                  ", lambda_re=",'%10.6f'%lambda_re[i],\
                  ", deviation=",'%6.3e'%(abs(lambda_pnt[i]-lambda_re[i])))
        return lambda_pnt,lambda_re

"""

Part I: Compute a single lattice point, certain lattice points (bandgap).

"""

# Compute eigenvalues at one SINGLE lattice vector.
def eigen_1p(N,d_flag_name,alpha):

    A,B,Ms,INV,X0,options_lobpcg=uniform_initialization(N,d_flag_name,alpha)

    lambda_pnt,lambda_re,__,X=eigen_lobpcg.PCs_mfd_lobpcg(A,B,Ms,\
            INV,X0,m_conv,options_lobpcg)
    del X0
    
    print("N=",N,", lattice type:",d_flag_name,", alpha=[",\
          '%5.2f'%(alpha[0]/pi),'%5.2f'%(alpha[1]/pi),'%5.2f'%(alpha[2]/pi),\
          "] pi.")
    
    # Print, residual.
    for i in range(m_conv):
        l1=scal*np.sqrt(lambda_pnt[i])
        l2=scal*np.sqrt(lambda_re[i])
        r=mfd.res_comp(A,Ms,lambda_re[i],X[:,i],options_lobpcg["pack_name"])
        print("i=",i+1,", ",'%10.6f'%l1,'%10.6f'%l2,'%6.3e'%(abs(l1-l2)),", res=",\
              '%6.3e'%(mfd.norm_gpu(r)))
            
    return
  
# Compute the complete bandgap w.r.t to a given lattice material with 
# certain grid size. (uniformly tensor division)
def bandgap(N,d_flag_name,gap,indices=None):
    
    n=N**3
    M_ind,CT,sym_points=diel.dielectric_save_and_load(N,d_flag_name)
    D,Di=mfd.fft_blocks(N,k,CT)
    __,n_pt=np.size(sym_points)
    n_pt-=1
    
    if not indices:
        # Default: compute the complete bandgap.
        indices=list(range(n_pt*gap))
    if max(indices)>n_pt*gap or min(indices)<1:
        ValueError("Index is non-positive or is incompatible with gap size.")
    
    # Discrete lattice points.
    alphas=np.zeros((n_pt*gap,3))
    for i in range(n_pt):
        alphas[(i+1)*gap-1,:]=sym_points[i+1,:]
        for j in range(gap-1):
            alphas[i*gap+j,:]=((j+1)*sym_points[i+1,:]+(gap-j-1)*sym_points[i,:])/gap
            
    # Preload. 
    if not os.path.exists("output"):
        os.mkdir("output")
    
    NP=eval(pack_name[0:2])
    bandgap_name="output/bandgap_"+d_flag_name+".json"
    var_name=d_flag_name+"_"+str(N)
    if not os.path.exists(bandgap_name):
        # New lattice bandgap plot.
        print("The bandgap of type ",d_flag_name," has no previous record.")
        gap_rec=bandgap_info(np.zeros((n_pt*gap,m_conv)),np.zeros((n_pt*gap,2)))        
        gap_lib={var_name:gap_rec}
        
        with open(bandgap_name,'w') as file:
            json.dump(gap_lib,file,indent=4)
    else:
        with open(bandgap_name,'r') as file:
            gap_lib=json.load(bandgap_name)
        if var_name in gap_lib.keys():
            # Previous record exists.
            print("Lattice type ",d_flag_name," with grid size N=",N,\
                  " has a previous record.")
            gap_rec=gap_lib[var_name]
        else:
            # New grid size.
            print("Lattice type ",d_flag_name,\
                  " will be computed with a new grid size N=",N,".")
            gap_rec=bandgap_info()
            gap_rec.eigen=NP.zeros((n_pt*gap,m_conv)).tolist()
            gap_rec.iters=NP.zeros((n_pt*gap,2)).tolist()
            gap_lib[var_name]=gap_rec
            
            with open(bandgap_name,'w') as file:
                json.dump(gap_lib,file,indent=4)
    
    X0=rand(3*n,m_conv+max(10,m_conv))
    options_lobpcg={"pack_name":pack_name,\
                    "iter_max":1000,\
                    "tol":tol/scal/scal}
        
    for i in range(len(indices)):
        t_h=time()
        alpha=alphas[indices[i]]
        relax_opt,pnt=mfd.set_relaxation(N,alpha)
        options_lobpcg["shift"]=relax_opt[0]
        m=m_conv+round(m_conv*relax_opt[1])
        
        A=D+1j*alpha*Di/(2*pi)
        B=(np.hstack(((A[0:n]*A[0:n].conj()).real,\
                      (A[n:2*n]*A[n:2*n].conj()).real,\
                      (A[2*n:]*A[2*n:].conj()).real)),\
           np.hstack((A[0:n].conj()*A[n:2*n],A[0:n].conj()*A[2*n:],A[n:2*n].conj()*A[2*n:])))
        INV=mfd.inverse_3_times_3_B(B,pnt)
        
        A/=scal
        B=(pnt*B[0]/scal/scal,pnt*B[1]/scal/scal)
        INV=(INV[0]*scal*scal,INV[1]*scal*scal)
        t_o=time()
        print("Matrix blocks done, ",'%6.3f'%(t_o-t_h),"s elapsed.")
        
        lambda_pnt,lambda_re,iters,X=eigen_lobpcg.PCs_mfd_lobpcg\
            (A,B,(M_ind,1/eps),INV,X0[:,:m],m_conv,options_lobpcg)        
        
        print("Gap index ",indices[i]," out of ",n_pt*gap," (",d_flag_name,") is computed.")
        print("Iterations=",iters[0],", runtime=",'%6.3f'%iters[1],"s.\n")
        lamdba_pnt,lambda_re=print_and_normalize(lambda_pnt,lambda_re)
        
        gap_rec.eigen[indices[i],:]=lambda_re.tolist()
        gap_rec.iters[indices[i],:]=iters.tolist()
        
        t_h=time()
        gap_lib[var_name]=gap_rec
        
        with open(bandgap_name,'w') as file:
            json.dump(gap_lib,file,indent=4)
        t_o=time()
        print("Gap info library (",d_flag_name,") is updated, file optime=",\
              '%6.3f'%(t_o-t_h),"s.")
    return
        
"""

Part II: Control experiment.

"""
 
# Control experiment. 
# Factor: tolerance (tol), scaling coefficient (scal), grid size (N),
#         linear algebra package (pack), dielectric coefficient (eps).
#         penalty number (pnt), relaxation test (rela)
    
def tol_cmp(N,d_flag_name,tols,alpha=np.array([pi,pi,pi])):

    A,B,Ms,INV,X0,options_lobpcg=uniform_initialization(N,d_flag_name,alpha)

    n_tols=len(tols)
    lambda_pnt=NP.zeros((n_tols,m_conv))
    lambda_re=NP.zeros((n_tols,m_conv))
    iters=NP.zeros((n_tols,2))
    
    for i in range(n_tols):
        options_lobpcg["tol"]=tols[i]/scal/scal
        lambda_pnt[i,:],lambda_re[i,:],iters[i,:],__=eigen_lobpcg.PCs_mfd_lobpcg\
            (A,B,Ms,INV,X0,m_conv,options_lobpcg)
            
        print("\ntol=",'%.2e'%tols[i]," is done computing.\n")
        lambda_pnt[i,:],lambda_re[i,:]=print_and_normalize(lambda_pnt[i,:],lambda_re[i,:])        
        
    std_pnt=NP.std(lambda_pnt,axis=0)
    std_re=NP.std(lambda_re,axis=0)
    print("Tolerance:")
    for i in range(n_tols):
        print("tol=",'%5.2e'%tols[i],", iterations=",iters[i,0],", time=",\
              '%5.2f'%iters[i,1],"s.")
    
    print("\nStandard deviation of each eigenvalue:")
    for i in range(m_conv):
        print("i=",'%4d'%(i+1),"std_pnt=",'%7.3e'%std_pnt[i],",std_re=",'%7.3e'%std_re[i])
     
    return

def pnt_cmp(N,d_flag_name,pnts,alpha=np.array([pi,pi,pi])):
    # Increase default penalty by ratio N^pnts[i].
    
    A,B,Ms,__,X0,options_lobpcg=uniform_initialization(N,d_flag_name,alpha)
    __,pnt0=mfd.set_relaxation(N,alpha)
    B=(B[0]/pnt0,B[1]/pnt0)

    n_pnts=len(pnts)
    lambda_pnt=NP.zeros((n_pnts,m_conv))
    lambda_re=NP.zeros((n_pnts,m_conv))
    iters=NP.zeros((n_pnts,2))
    
    for i in range(n_pnts):
        pnt=2*N**(pnts[i]+1.0)
        INV=mfd.inverse_3_times_3_B(B,pnt)

        lambda_pnt[i,:],lambda_re[i,:],iters[i,:],__=eigen_lobpcg.PCs_mfd_lobpcg\
            (A,(B[0]*pnt,B[1]*pnt),Ms,INV,X0,m_conv,options_lobpcg)
        
        print("\npnt=N^",'%.2f'%pnts[i]," is done computing.\n")
        lambda_pnt[i,:],lambda_re[i,:]=print_and_normalize(lambda_pnt[i,:],lambda_re[i,:])
        
    std_pnt=NP.std(lambda_pnt,axis=0)
    std_re=NP.std(lambda_re,axis=0)
    print("Penalties:")
    for i in range(n_pnts):
        print("pnt=",'%5.2e'%pnts[i],", iterations=",iters[i,0],", time=",\
              '%5.2f'%iters[i,1],"s.")
    
    print("\nStandard deviation of each eigenvalue:")
    for i in range(m_conv):
        print("i=",'%4d'%(i+1),"std_pnt=",'%7.3e'%std_pnt[i],",std_re=",'%7.3e'%std_re[i])
    return

# Pairs: tol & rela.
def rela_cmp(N,d_flag_name,relas,alpha=np.array([pi,pi,pi])):
    
    A,B,Ms,INV,__,options_lobpcg=uniform_initialization(N,d_flag_name,alpha)
    
    n=N**3
    tols=relas[0]
    rels=relas[1]
    n_relas=len(tols)
    lambda_pnt=NP.zeros((n_relas,m_conv))
    lambda_re=NP.zeros((n_relas,m_conv))
    iters=NP.zeros((n_relas,2))
    
    if n_relas!=len(rels):
        ValueError("Tuple relas should contain two arrays with same length.")
    
    for i in range(n_relas):
        options_lobpcg["tol"]=tols[i]
        m=m_conv+round(m_conv*rels[i])
        X0=rand(3*n,m)
        lambda_pnt[i,:],lambda_re[i,:],iters[i,:],__=eigen_lobpcg.PCs_mfd_lobpcg\
            (A,B,Ms,INV,X0,m_conv,options_lobpcg)
        lambda_pnt[i,:],lambda_re[i,:]=print_and_normalize(lambda_pnt[i,:],lambda_re[i,:])
            
    return

def scal_cmp(N,d_flag_name,scals,alpha=np.array([pi,pi,pi])):

    A,B,Ms,INV,X0,options_lobpcg=uniform_initialization(N,d_flag_name,alpha)

    n_scals=len(scals)
    lambda_pnt=NP.zeros((n_scals,m_conv))
    lambda_re=NP.zeros((n_scals,m_conv))
    iters=NP.zeros((n_scals,2))
    
    for i in range(n_scals):
        scal0=(N**scals[i])**2
        options_lobpcg["tol"]/=scal0

        lambda_pnt[i,:],lambda_re[i,:],iters[i,:],__=eigen_lobpcg.PCs_mfd_lobpcg\
            (A/np.sqrt(scal0),(B[0]/scal0,B[1]/scal0),Ms,(INV[0]*scal0,INV[1]*scal0),\
             X0,m_conv,options_lobpcg)

        print("\nscal=N^",'%.2f'%scals[i]," is done computing.\n")
        lambda_pnt[i,:],lambda_re[i,:]=print_and_normalize(lambda_pnt[i,:],lambda_re[i,:])
     
    std_pnt=NP.std(lambda_pnt,axis=0)
    std_re=NP.std(lambda_re,axis=0)
    print("Scaling power:")
    for i in range(n_scals):
        print("Scal=",'%5.2f'%scals[i],", iterations=",iters[i,0],", time=",\
              '%5.2f'%iters[i,1],"s.")
    
    print("\nStandard deviation of each eigenvalue:")
    for i in range(m_conv):
        print("i=",'%4d'%(i+1),"std_pnt=",'%7.3e'%std_pnt[i],",std_re=",'%7.3e'%std_re[i])
    
    return

def pack_cmp(Ns,d_flag_name,packs,alpha=np.array([pi,pi,pi])):
    
    # Packs={gpu_pack, cpu_pack}. 
    # cpu: numpy,scipy. gpu: cupy,cupyx.
    
    runtime_name="runtime_"+packs[0]+'_'+packs[1]    
    
    if not os.path.exists(runtime_name):
        runtime_pack_lib={}
    else:
        with open(runtime_name,'r') as file:
            runtime_pack_lib=json.load(file)

    n_Ns=len(Ns)
    iters=NP.zeros((n_Ns,2,2))
    
    for i in range(n_Ns):
        A,B,Ms,INV,X0,options_lobpcg=uniform_initialization(Ns[i],d_flag_name,alpha)
        
        options_lobpcg["pack_name"]=packs[0]
        lambda_pnt,lambda_re,iters[i,0,:],__=eigen_lobpcg.PCs_mfd_lobpcg\
            (A,B,Ms,INV,X0,m_conv,options_lobpcg)
        print("\nN=,",Ns[i],", Package=",packs[0]," is done computing.\n")
        __,__=print_and_normalize(lambda_pnt,lambda_re)
        
        options_lobpcg["pack_name"]=packs[1]
        lambda_pnt,lambda_re,iters[i,1,:],__=eigen_lobpcg.PCs_mfd_lobpcg\
            (A,B,Ms,INV,X0,m_conv,options_lobpcg)
        print("\nN=,",Ns[i],", Package=",packs[1]," is done computing.\n")
        __,__=print_and_normalize(lambda_pnt,lambda_re)
        
        runtime_pack_lib["pack_cmp_"+str(Ns[i])]=\
            runtime_info(iters[i,0,0],iters[i,1,1],iters[i,0,1])
            
        with open(runtime_name,'w') as file:
            json.dump(runtime_pack_lib,file,indent=4)

    print("Runtime comparison using different linear algebra packages:")
    for i in range(n_Ns):
        print("N=,",Ns[i],", iterations=",iters[i,0,0],", cputime=",\
              '%5.2f'%iters[i,1,1],"s, gputime=",'%5.2f'%iters[i,0,1],\
              "s, ratio=",'%5.2f'%(iters[i,1,1]/iters[i,0,1]),".")
      
    return

def eps_cmp():
    
    return
    
def grid_cmp(Ns,d_flag_name,alpha=np.array([pi,pi,pi])):
    
    Ns.sort()

    n_Ns=len(Ns)
    lambda_pnt=NP.zeros((n_Ns,m_conv))
    lambda_re=NP.zeros((n_Ns,m_conv))
    iters=NP.zeros((n_Ns,2))
    
    for i in range(n_Ns):
        A,B,Ms,INV,X0,options_lobpcg=uniform_initialization(Ns[i],d_flag_name,alpha)
        lambda_pnt[i,:],lambda_re[i,:],iters[i,:],__=eigen_lobpcg.PCs_mfd_lobpcg\
            (A,B,Ms,INV,X0,m_conv,options_lobpcg)

        print("\nGrid size=",Ns[i]," is done computing.\n")
        lambda_pnt[i,:],lambda_re[i,:]=print_and_normalize(lambda_pnt[i,:],lambda_re[i,:])

    print("Grid size:")
    for i in range(n_Ns):
        print("N=",Ns[i],", iterations=",iters[i,0],", time=",\
              '%5.2f'%iters[i,1],"s.")
    
    print("\nDeviation (relative error):")
    
    for i in range(m_conv):
        print("i=",i+1,end=":\t")
        for j in range(1,n_Ns):
            print('%7.3e'%abs(lambda_pnt[j,i]-lambda_pnt[j-1,i]),end=' ')
        print()
        
    return
#eigen_1p(100,"sc_curv",np.array([pi,pi,pi]))
#tol_cmp(100,"sc_curv",[5e-4,1e-4,1e-5,1e-6])
#pnt_cmp(100,"sc_curv",[0,1.2,1.5,2])        
#(100,"sc_curv",[0,0.2,0.5,0.6,0.8,1])
grid_cmp([80,100,120],"sc_curv")
     
    