# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:56:45 2024

@author: 11034
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import savemat,loadmat
from pathlib import Path

def load_gapplot(lattice_type,N,index):
    
    # Loading ...
    
    file_name="output/"+lattice_type+'_'+str(N)+".mat"
    ind_name=lattice_type+'_'+str(N)
    if Path(file_name).exists()==False:
        print("File '",file_name,"' doesn't exist.")
        return
    else:
        libs=loadmat(file_name)
        if ind_name in libs.keys():
            gap_info=libs[ind_name]
            del libs
        else:
            print("File '",file_name,"' exists but grid size N=",N," is never"\
                  " computed.")
            return
        
    # Plotting ...
    
    gap_info.eigen