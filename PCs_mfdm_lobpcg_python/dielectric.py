# -*- coding: utf-8 -*-
"""

Dielectric coefficient.

"""

import numpy as np
from numpy.linalg import norm
from numpy import pi
from time import time
from scipy.io import savemat,loadmat
from pathlib import Path

"""

Part 1: A universal procedure that generates index array corresponding
        with distribution of permittivity constant.
        Previous information is also saved in a certain directory.

"""

def dielectric_initialize():
    
    diel_lib={'CT_sc':np.identity(3),\
              'CT_bcc':np.array([[0,1,1],[1,0,1],[1,1,0]]),\
              'CT_fcc':np.array([[-1,1,1],[1,-1,1],[1,1,-1]]),\
              'sym_sc':np.array([[0,0,0],[pi,0,0],[pi,pi,0],\
                                 [pi,pi,pi],[0,0,0]]),\
              'sym_bcc':np.array([[0,0,2*pi],[0,0,0],[pi,pi,pi],\
                                  [0,0,2*pi],[pi,0,pi],[0,0,0],\
                                  [0,2*pi,0],[pi,pi,pi],[pi,0,pi]]),\
              'sym_fcc':np.array([[0,2*pi,0],[pi/2,2*pi,pi/2],[pi,pi,pi],\
                                  [0,0,0],[0,2*pi,0],[pi,2*pi,0],\
                                  [3*pi/2,3*pi/2,0]])}
    
    savemat("diel_info.mat",diel_lib)

def dielectric_save_and_load(N,d_flag_name):
    
    # Is function eval(d_flag_name) callable ?
    if not hasattr(eval("d_flag_"+d_flag_name),'__call__'):
        print("Flag function for lattice type ",d_flag_name," is NOT defined.")
        return
    
    t_h=time()

    # Check if the directory exists.
    dir_name="dielectric_examples"
    if not Path(dir_name).exists():
        Path(dir_name).mkdir()
    
    # Determine CT and symmetry points.
    file_name=dir_name+"/"+d_flag_name+".mat"
    if d_flag_name[2]=='_':
        # sc lattice.
        lattice_name=d_flag_name[0:2]
    else:
        # bcc or fcc lattice.
        lattice_name=d_flag_name[0:3]
        
    # Preload information.
    if not Path("diel_info.mat").exists():
        dielectric_initialize()
    diel_lib=loadmat("diel_info.mat")
    CT=diel_lib["CT_"+lattice_name]
    sym_points=diel_lib["sym_"+lattice_name]
    
    var_name=file_name+'_'+str(N)
    if not Path(file_name).exists():
        # No previous record.
        print("New lattice type ",d_flag_name)
        ind_d=dielectric_index(N,CT,eval("d_flag_"+d_flag_name))
        ind_lib={var_name:ind_d}
        savemat(file_name,ind_lib)
    else:
        ind_lib=loadmat(file_name)        
        if var_name in ind_lib.keys():
            # previous record exists.
            print("Lattice type ",d_flag_name," with grid size N=",N," already exists.")
            ind_d=ind_lib[var_name]
        else:
            # New grid size.
            print("New grid size=",N," for lattice type",d_flag_name)
            ind_d=dielectric_index(N,CT,eval("d_flag_"+d_flag_name))
            ind_lib[var_name]=ind_d
            savemat(file_name,ind_lib)

    t_o=time()
    print("Dielectric matrix done, ",'%6.3f'%(t_o-t_h),"s elapsed.")
    return ind_d,CT,sym_points

def dielectric_index(N,CT,d_flag):
    
    t_h=time()
    
    n=N**3
    h=1.0/N
    ind_d=np.zeros(3*n,dtype=int)
    
    ind=0
    L=0
    
    for z in range(N):
        for y in range(N):
            for x in range(N):

                e_x=np.dot(CT,np.array([(x+0.5)*h,y*h,z*h]))
                e_y=np.dot(CT,np.array([x*h,(y+0.5)*h,z*h]))
                e_z=np.dot(CT,np.array([x*h,y*h,(z+0.5)*h]))
                
                if d_flag(e_x):
                    ind_d[L]=ind
                    L=L+1
                    
                if d_flag(e_y):
                    ind_d[L]=n+ind     
                    L=L+1
                
                if d_flag(e_z):
                    ind_d[L]=2*n+ind
                    L=L+1
                
                ind=ind+1
    
    t_o=time()
    print("Time for assembling a new diel-diag matrix: ",'%.3f'%(t_o-t_h),"s.")
    
    return ind_d[:L]


"""

Part 2: Distance functions of simple cubic (SC) lattice.

"""

def d_flag_sc_curv(e):
    
    R1=0.345
    r1=0.11
    
    mid=np.array([0.5,0.5])
    
    if norm(e[[1,2]]-mid)<=r1 or \
       norm(e[[0,2]]-mid)<=r1 or \
       norm(e[[0,1]]-mid)<=r1 or \
       norm(e-np.array([0.5,0.5,0.5]))<=R1:
        return 1 
    else:
        return 0
    
def d_flag_sc_flat1(e):
    
    e=e%1 
    
    if ( e[1]<=0.25 and e[2]<=0.25 ) or \
       ( e[0]<=0.25 and e[2]<=0.25 ) or \
       ( e[0]<=0.25 and e[1]<=0.25 ):
        return 1
    else:
        return 0
    
def d_flag_sc_flat2(e):
    
    e=e%1 
    
    if ( e[1]<=0.25 and e[2]<=0.25 ) or \
       ( e[0]<=0.25 and e[2]>=0.25 and e[2]<=0.5 ) or \
       ( e[1]>=0.5 and e[1]<=0.75 and e[2]>=0.5 and e[2]<=0.75 ) or \
       ( e[0]>=0.5 and e[0]<=0.75 and e[2]>=0.75 ):
        return 1
    else:
        return 0
    
    
"""

Part 3: Distance functions of body centered cubic (BCC) lattice.

"""
       
def d_flag_bcc_single_gyroid(e):
    
    pi=np.pi
    g=lambda r: np.sin(2*pi*r[0])*np.cos(2*pi*r[1])+np.sin(2*pi*r[1])*np.cos(2*pi*r[2])\
                +np.sin(2*pi*r(3))*np.cos(2*pi*r(1));
    
    if g(e)>1.1:
        return 1
    else:
        return 0
    
def d_flag_bcc_double_gyroid(e):
    
    pi=np.pi
    g=lambda r: np.sin(2*pi*r[0])*np.cos(2*pi*r[1])+np.sin(2*pi*r[1])*np.cos(2*pi*r[2])\
                +np.sin(2*pi*r(3))*np.cos(2*pi*r(1));
    
    if np.abs(g(e))>1.1:
        return 1
    else:
        return 0
        
    
"""

Part 4: Distance functions of face centered cubic (FCC) lattice.

"""

"""
def d_flag_fcc(e):
    
    a1=np.array([0,1/2,1/2])
    a2=np.array([1/2,0,1/2])
    a3=np.array([1/2,1/2,0])

    cnt=(a1+a2+a3)/4
                
    tran1=[zeros(3,1),eye(3),[0;1;1],[1;0;1],[1;1;0],[1;1;1],...
       [0;1/2;1/2],[1/2;0;1/2],[1/2;1/2;0],[1;1/2;1/2],[1/2;1;1/2],[1/2;1/2;1],...
       cnt,cnt+a1,cnt+a2,cnt+a3];
    tran2=[np.zeros(3,1),a1,a2,a3]; 

    r=0.12;
    b=0.11;

    o1,d1,c1=cnt/2,cnt/2,norm(d1)        
    d1=d1/c1
    o2,d2=(a1+cnt)/2,(a1-cnt)/2
    c2=norm(d2)
    d2=d2/c2;
    
    o3,d3=(a2+cnt)/2,d3=(a2-cnt)/2
    c3=norm(d3)
    d3=d3/c3
    o4,d4=(a3+cnt)/2,(a3-cnt)/2
    c4=norm(d4)
    d4=d4/c4

    X=e-tran1
    
    if (diag(X'*X)-r^2)<0 || ...
   ell(e,o1,b,c1,d1,tran2) || ...
   ell(e,o2,b,c2,d2,tran2) || ...
   ell(e,o3,b,c3,d3,tran2) || ...
   ell(e,o4,b,c4,d4,tran2)
    flag=1;
else
    flag=0;
end

end

function val=ell(x,cnt,b,c,d,tran)

X=x-(cnt+tran);
a=sqrt(b^2+c^2);
L1=(X'*d).^2;
L2=diag(X'*X)-L1;

if min(L1/(a^2)+L2/(b^2))<1
   val=1;
else
    val=0;
end

end
"""
