#!/usr/bin/env python
from __future__ import print_function
import h5py
import sys
import os
import numpy as np
import time

import scipy.io as sio
#from Tools_EKI.FEnicsTools import Heat_predictions
#from Tools_EKI.Miscellaneous import ExtractParameters
from FEnicsTools import Heat_predictions
from Miscellaneous import ExtractParameters
#from pyevtk.hl import imageToVTK

def ForwardRun3D(ID,task,flag):
    ti=time.time()
    k_B,k_A,c_B,c_A,LS, Error_int,Error_ext, R_int,R_ext, w,Deltaw,beta=ExtractParameters(ID,task)
    alpha_int=1/R_int
    alpha_ext=1/R_ext
#    foldername='Output_'+ID+'/'

#os.system('rm  '+foldername+'error_'+f'{task}'+ '.mat')
#pred,Flux_data=Heat_predictions(ID,LS,Error_int,Error_ext,k_A,k_B,c_A,c_B,beta,w,Deltaw, alpha_int,alpha_ext,flag)

#if flag==0 or flag==1:
    ex=0
    while ex==0:
        try:
            pred,Flux_data=Heat_predictions(ID,LS,Error_int,Error_ext,k_A,k_B,c_A,c_B,beta,w,Deltaw, alpha_int,alpha_ext,flag)
            elapsed = time.time() - ti
            ex=1
        #sio.savemat(foldername+'pred_'+f'{task}'+ '.mat', {'pred':pred,'Flux':Flux_data,'time':elapsed})
        except:
            #print("Error running fwd model")
            dummy=0
    return pred,Flux_data, elapsed

if __name__ == "__main__":
    
    task=int(sys.argv[1])
    flag=int(sys.argv[2])
    ID=sys.argv[3]
    pred_all=[]
    time_all=[]
    Flux_all=[]
    foldername='Output_'+ID+'/'
    print(task)
    counter=(task-1)+1
    pred,Flux,e=ForwardRun3D(ID,counter,flag)
    Flux_all=np.append(Flux_all,Flux)
    pred_all=np.append(pred_all,pred)
    time_all=np.append(time_all,e)

#    for n in range(5):
#        counter=n+5*(task-1)+1
#        print(counter)
#        pred,Flux,e=ForwardRun3D(counter,flag)
#        Flux_all=np.append(Flux_all,Flux)
#        pred_all=np.append(pred_all,pred)
#        time_all=np.append(time_all,e)

    sio.savemat(foldername+'pred_'+f'{task}'+ '.mat', {'pred':pred_all,'time':time_all,'Flux':Flux_all})
