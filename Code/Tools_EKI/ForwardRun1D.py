#!/usr/bin/env python
from __future__ import print_function
import h5py
import sys
import numpy as np
from FEnicsTools import Heat_predictions_oneD
import scipy.io as sio
import os
import time

from Miscellaneous import get_scalar, get_scalar_low






def ForwardRun1D(ID,task,flag):
    ti=time.time()
    with h5py.File('Ensemble_'+ID+'.hdf5', 'r') as f:
        g=f['FwdModel']
        if flag==0:
            num_steps=g.attrs['num_steps_inv']
        else:
            num_steps=g.attrs['num_steps_pred']
        
        k_en  = f['Unknown/Un_KC_en'][0][:,task-1]
        c_en = f['Unknown/Un_KC_en'][1][:,task-1]
        R_I_en = f['Unknown/Un_R_en'][0][0,task-1]
        R_E_en = f['Unknown/Un_R_en'][1][0,task-1]

    K_values=get_scalar(ID,'K',k_en)
    C_values=get_scalar(ID,'C',c_en)
    R_int=get_scalar_low(ID,'R_I',R_I_en)
    R_ext=get_scalar_low(ID,'R_E',R_E_en)
    alpha_int=1/R_int
    alpha_ext=1/R_ext

    ex=0
    while ex==0:
        try:
            pred=Heat_predictions_oneD(ID,K_values,C_values,alpha_int,alpha_ext,num_steps)
            elapsed = time.time() - ti
            ex=1
        except:
            

            dummy=0
    return pred,elapsed


if __name__ == "__main__":
    
    task=int(sys.argv[1])
    flag=int(sys.argv[2])
    ID=sys.argv[3]

    pred_all=[]
    time_all=[]
    
    foldername='Output_'+ID+'/'
    print(task)
    for n in range(1000):
        counter=n+1000*(task-1)+1
        pred,e=ForwardRun1D(ID,counter,flag)
        pred_all=np.append(pred_all,pred)
        time_all=np.append(time_all,e)
    
    sio.savemat(foldername+'pred_'+f'{task}'+ '.mat', {'pred':pred_all,'time':time_all})

