from __future__ import print_function

import os
import numpy as np
import h5py
import sys
import scipy.io as sio
from Tools_EKI.Prior import writePrior_scalar,writePrior_scalar_transform
from Tools_EKI.Miscellaneous import SetFoldersFiles,SaveData,SavePriorForwardOneD
print('Genering Forward Model and Prior')


ID='1D_syn'

N=150
Area=2.0*2.0
D=0.3

N_En=10000
Num_Ks=5

layers=[0.05,0.12, 0.18, 0.25]
#layers=[0.07,0.15, 0.23]
num_steps_inv = 300
num_steps_pred = 751

dt =60*5#

#Seed for random numbers
seed=1026

np.random.seed(seed)

mat = sio.loadmat('./Synthetic_Data/Temps_syn.mat')
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
U_R.append(writePrior_scalar(ID,1,'R_E',N_En,[0.02, 0.2],0.01,0.3))

SavePriorForwardOneD(ID,D,Area,layers,N,Num_Ks,num_steps_inv,num_steps_pred,dt,Ti,Te,N_En,U_KC,U_R)

mat = sio.loadmat('./FlowRates/FlowRates_syn.mat')
Data =mat['data']
Gamma=mat['gamma']
sqrt_Gamma=np.sqrt(Gamma)
inv_sqrt_Gamma=np.reciprocal(sqrt_Gamma)
SaveData(ID,Data,inv_sqrt_Gamma)

    

print('Completed')
