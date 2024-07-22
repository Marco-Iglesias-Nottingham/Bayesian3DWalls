from __future__ import print_function

import os
import numpy as np
import h5py
import sys
import scipy.io as sio
from Tools_EKI.Prior import writePrior_scalar_joint, writePrior_scalar, writePrior_scalar_transform
from Tools_EKI.Miscellaneous import SetFoldersFiles,SaveData
from Tools_EKI.FEnicsTools import GetSyndata

from fenics import BoxMesh, Point

print('Genering Forward Model and Prior')

ID='3D_syn'
#Wall's domain (origin=(0,0,0))
D=[2.0,2.0,0.3]
#Reduced domain for anomaly
D_reduced=[[0.0,2.0, 0.0, 1.0],[0.0,2.0, 1.0, 2.0]]
layers=[0.1,0.2]
#Mesh for 3D Fenics Heat transfer solver
Fenics_N=[40,40,50]

#Mesh for 2D random fields (level-set and Model Errors)
RF_N=[88,88]
num_steps_inv = 250
num_steps_pred = 750

dt =60*5  # time step size

#Seed for random numbers
seed=10288

np.random.seed(seed)


#Speficy locations of sensors for temperature and flux

Po_temp=[[0.45,0.75,0],[0.85,0.75,0],[1.25,0.25,0],[0.45,1.6,0],[1.25,1.4,0],[1.65,1.4,0],[0.25,0.25,D[2]],[0.85,0.5,D[2]],[1.25,0.8,D[2]],[0.45,1.1,D[2]],[0.8,1.4,D[2]],[1.5,1.6,D[2]]   ]
Po_flux=[[0.3,0.2,0],[0.85,0.4,0],[1.25,0.7,0],[0.45,1.1,0],[0.85,1.25,0],[1.25,1.1,0],[0.85,1.6,0],[1.25,1.6,0],[0.4,0.85,D[2]],[0.6,0.5,D[2]],[1.25,0.25,D[2]],[0.45,1.6,D[2]],[1.2,1.3,D[2]],[1.5,1.1,D[2]]  ]



K_back=[0.84,0.0262,0.71]
C_back=[1.4e6,1.225e3,1.7e6]
K_defe=[45,120]
C_defe=[2.5e6,3.2e6]
R_int=0.13
R_ext=0.04


sigma_TI, sigma_temp,sigma_flux = 0.02,0.1,0.05
mat = sio.loadmat('./Synthetic_Data/Temps_syn.mat')
T_int =mat['T_int']
T_ext =mat['T_ext']
Ti=T_int[:,0]
Te=T_ext[:,0]



SetFoldersFiles(ID)

#Ensemble size
N_En=1000


mat = sio.loadmat('./PriorEnsembles/Lev_en_syn.mat')
U=[]
Lev_en1=mat['Lev1']
Lev_en2=mat['Lev2']

K_back=[0.84,0.0262,0.71]
C_back=[1.4e6,1.225e3,1.7e6]
K_defe=[45,120]
C_defe=[2.5e6,3.2e6]
R_int=0.13
R_ext=0.04



U.append(writePrior_scalar(ID,1,'c_B1',N_En,[1e6, 2.5e6],7e5,3e6))
U.append(writePrior_scalar(ID,1,'c_B2',N_En,[1.1e3, 2e3],0.8e3,2.5e3))
U.append(writePrior_scalar(ID,1,'c_B3',N_En,[1e6, 2.5e6],7e5,3e6))
U.append(writePrior_scalar(ID,1,'k_B1',N_En,[0.5, 1.2],0.1,2.0))
U.append(writePrior_scalar(ID,1,'k_B2',N_En,[0.01, 0.04],0.005,0.05))
U.append(writePrior_scalar(ID,1,'k_B3',N_En,[0.5, 1.2],0.1,2.0))
U.append(writePrior_scalar(ID,1,'c_A1',N_En,[7e5, 4e6],5e5,4.5e6))
U.append(writePrior_scalar(ID,1,'c_A2',N_En,[7e5, 4e6],5e5,4.5e6))
U.append(writePrior_scalar(ID,1,'k_A1',N_En,[20, 150],10,155))
U.append(writePrior_scalar(ID,1,'k_A2',N_En,[20, 150],10,155))
U.append(writePrior_scalar(ID,1,'R_I',N_En,[0.09,0.17],0.08,0.19))
U.append(writePrior_scalar(ID,1,'R_E',N_En,[0.025,0.055],0.015,0.06))
w_en1,Deltaw_en1= writePrior_scalar_joint(ID,1,'Geo1',N_En,[0.05,0.25],0.0,0.27,D[2]-0.02)
w_en2,Deltaw_en2= writePrior_scalar_joint(ID,1,'Geo2',N_En,[0.05,0.25],0.0,0.27,D[2]-0.02)

U.append(w_en1)
U.append(w_en2)
U.append(Deltaw_en1)
U.append(Deltaw_en2)
U.append(writePrior_scalar(ID,1,'beta1',N_En,[0.6,2.1],0.5,2.3))
U.append(writePrior_scalar(ID,1,'beta2',N_En,[0.6,2.1],0.5,2.3))

   

#Save everything in Ensemble_ID.hdf5 file
with h5py.File('Ensemble_'+ID+'.hdf5', 'a') as g:
    f=g.create_group('FwdModel')
    f.attrs['ID']=ID
    f.attrs['D']=D
    f.attrs['layers']=layers
    f.attrs['D_reduced']=D_reduced
    f.attrs['Fenics_N']=Fenics_N
    f.attrs['RF_N']=RF_N
    f.attrs['num_steps_inv']=num_steps_inv
    f.attrs['num_steps_pred']=num_steps_pred
    f.attrs['Po_temp']=Po_temp
    f.attrs['Po_flux']=Po_flux
    f.attrs['dt']=dt
    f.attrs['Ti']=Ti
    f.attrs['Te']=Te

Data,TI_data,Flux_Data, sqrt_Gamma= GetSyndata(ID,K_back,K_defe,C_back,C_defe, 1/R_int,1/R_ext,sigma_TI, sigma_temp,sigma_flux, flag=0,flag2=0)
sio.savemat('./Results_Syn/TI_synthetic.mat', {'TI_data':TI_data})

        
with h5py.File('Ensemble_'+ID+'.hdf5', 'a') as g:
    p=g.create_group('Unknown')
    p.create_dataset('Un_en', data=U)
    p.create_dataset('Level_en1', data=Lev_en1)
    p.create_dataset('Level_en2', data=Lev_en2)
    inv=g.create_group('Inversion')
    inv.attrs['N_En']=N_En


    
mesh = BoxMesh(Point(0,0,0), Point(D[0],D[1],D[2]),Fenics_N[0],Fenics_N[1],Fenics_N[2])
temp=mesh.coordinates()
print('mesh size',temp.shape)


    
    
    

inv_sqrt_Gamma=np.reciprocal(sqrt_Gamma)
sio.savemat('./Results_Syn/data.mat', {'data':Data, 'inv_sqrt_Gamma':inv_sqrt_Gamma})
SaveData(ID,Data,inv_sqrt_Gamma)

Data,TI_data,Flux_Data, sqrt_Gamma= GetSyndata(ID,K_back,K_defe,C_back,C_defe, 1/R_int,1/R_ext,sigma_TI, sigma_temp,sigma_flux, flag=1,flag2=0)
inv_sqrt_Gamma=np.reciprocal(sqrt_Gamma)
sio.savemat('./Results_Syn/data_all.mat', {'data':Data, 'inv_sqrt_Gamma':inv_sqrt_Gamma,'Flux_Data':Flux_Data})

Data,TI_data,Flux_Data, sqrt_Gamma= GetSyndata(ID,K_back,K_defe,C_back,C_defe, 1/R_int,1/R_ext,sigma_TI, sigma_temp,sigma_flux, flag=1,flag2=1)
inv_sqrt_Gamma=np.reciprocal(sqrt_Gamma)
sio.savemat('./Results_Syn/data_plain.mat', {'data':Data, 'inv_sqrt_Gamma':inv_sqrt_Gamma,'Flux_Data':Flux_Data})

        
        
        
        
        
print('done')

