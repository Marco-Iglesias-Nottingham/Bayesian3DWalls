from __future__ import print_function

import os
import numpy as np
import h5py
import sys
import scipy.io as sio
from Tools_EKI.Prior import writePrior_scalar,writePrior_scalar_transform
from Tools_EKI.Miscellaneous import SetFoldersFiles,SaveData,SavePriorForwardOneD
print('Genering Forward Model and Prior')


ID='1D'

N=300
Area=0.8*0.8
D=0.08

N_En=10000
Num_Ks=3

layers=[0.025,0.055]
num_steps_inv = 300
num_steps_pred = 510

dt =60#

#Seed for random numbers
seed=1026

np.random.seed(seed)

mat = sio.loadmat('./Real_Data/Temps.mat')
T_int =mat['T_int']
T_ext =mat['T_ext']
Ti=T_int[:,0]
Te=T_ext[:,0]


SetFoldersFiles(ID)
U_KC=[]
U_R=[]
U_KC.append(writePrior_scalar_transform(ID,Num_Ks,'K',N_En,[0.01 ,5],0.005,10))
U_KC.append(writePrior_scalar_transform(ID,Num_Ks,'C',N_En,[1e4 ,3e6],5e3,5e6))
U_R.append(writePrior_scalar(ID,1,'R_I',N_En,[0.05, 0.2],0.04,0.3))
U_R.append(writePrior_scalar(ID,1,'R_E',N_En,[0.05, 0.2],0.04,0.3))

SavePriorForwardOneD(ID,D,Area,layers,N,Num_Ks,num_steps_inv,num_steps_pred,dt,Ti,Te,N_En,U_KC,U_R)

mat = sio.loadmat('./FlowRates/FlowRates_real.mat')
Data =mat['data']
Gamma=mat['gamma']
sqrt_Gamma=np.sqrt(Gamma)
inv_sqrt_Gamma=np.reciprocal(sqrt_Gamma)
SaveData(ID,Data,inv_sqrt_Gamma)

    

print('Completed')
