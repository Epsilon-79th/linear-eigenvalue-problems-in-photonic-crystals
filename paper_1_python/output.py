# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:56:45 2024

@author: 11034
"""

import numpy as np
import matplotlib.pyplot as plt

import json

# Gap plot labels.
sc_labels=[r'$\Gamma$','L','M','N',r'$\Gamma$']
bcc_labels=["H'",r'$\Gamma$','P',"H'",'N',r'$\Gamma$','H','P','N']
fcc_labels=['X','U','L',r'$\Gamma$','X','W','K']


def compute_bandgap_ratio(frequencies,opts={"n_gap":1,"tol":1e-2}):
    
    F=np.sort(np.array(frequencies).flatten())
    F1=F[1:]-F[:-1]
    
    n_gap=opts["n_gap"]
    if n_gap==1:
        ind=np.argmax(F1)
        return np.array([F[ind],F[ind+1]])
    else:
        inds=(-F1).argsort()[:n_gap]
        if F1[inds[-1]]<opts["tol"]:
            ValueError("Not enough bandgaps.")
        
        omgs=np.zeros((len(inds),2))
        for i in range(omgs):
            omgs[i,0],omgs[i,1]=F[ind[i]],F[ind[i]+1]
        return omgs
    

def plot_bandgap(N,d_flag_name):
    
    file_name="output/bandgap_"+d_flag_name+".json"
    with open(file_name,'r') as file:
        gap_lib=json.load(file)
    
    var_name=d_flag_name+"_"+str(N)+"_"
    iters=np.array(gap_lib[var_name+"iterations"])
    frequencies=np.array(gap_lib[var_name+"frequencies"])
    
    omgs=compute_bandgap_ratio(frequencies)
    ratio=2*(omgs[1]-omgs[0])/(omgs[1]+omgs[0])
    
    labels=eval(d_flag_name.split('_')[0]+'_labels')
    n_pt=len(labels)-1
    n,m_conv=np.shape(frequencies)
    gap=round(n/n_pt)
    
    for i in range(n_pt*gap):
        plt.scatter((i+1)*np.ones((m_conv)),frequencies[i,:m_conv],s=3)
    plt.scatter(np.zeros((m_conv)),frequencies[(labels[1:].index(labels[0])+1)*gap-1,:m_conv],s=3)
    
    plt.xlabel('Wave Vector')
    plt.ylabel(r'$\omega / 2\pi$')
    plt.title('Figure of '+d_flag_name+' band structure, grid size N='+str(N)\
               +', gap ratio='+str(round(ratio,6)))
    
    plt.xticks(np.linspace(0,n_pt*gap,n_pt+1),labels)
    
    plt.show()
    
    # Iterations.
    
    iters=np.array(iters)
    print("Average iterations=",'%6.2f'%(sum(iters[:,0])/(n_pt*gap)),".")
    print("Average runtime=",'%6.2f'%(sum(iters[:,1])/(n_pt*gap)),"s.")
    print()
    print("Bandgap info from ",file_name," file.")
    return

def table_runtime(d_flag_name):
    
    file_name="output/runtime_"+d_flag_name+".json"
    with open(file_name,'r') as file:
        gap_lib=json.load(file)
        
    return

N=150
plot_bandgap(N,"sc_curv")
#plot_bandgap(N,"bcc_single_gyroid")
