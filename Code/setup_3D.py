from __future__ import print_function

import os
import numpy as np
import h5py
import sys
import scipy.io as sio
from Tools_EKI.Prior import writePrior_scalar_joint, writePrior_scalar
from Tools_EKI.Miscellaneous import SetFoldersFiles,SaveData
from Tools_EKI.FEnicsTools import GetPlain
from fenics import BoxMesh, Point

print('Genering Forward Model and Prior')

ID='3D'
#Wall's domain (origin=(0,0,0))
D=[0.8,0.8,0.08]
#Reduced domain for anomaly
D_reduced=[0.1859,0.6141, 0.1859, 0.6141]

#Mesh for 3D Fenics Heat transfer solver
Fenics_N=[50,50,60]

#Mesh for 2D random fields (level-set and Model Errors)
RF_N=[88,88]
num_steps_inv = 250
num_steps_pred = 500
layers=[D[2]]
dt =60  # time step size

#Seed for random numbers
seed=10286

np.random.seed(seed)


#Speficy locations of sensors for temperature and flux
Po_temp=[[0.4,0.375,0],[0.6,0.55,0],[0.2,0.25,0],[0.4,0.4,D[2]],[0.2,0.55,D[2]],[0.6,0.25,D[2]]]
Po_flux=[[0.4,0.4,0],[0.25,0.4,0],[0.4,0.25,0],[0.25,0.25,0] ]
std_temp=[0.1,0.1,0.1,0.1,0.1,0.1]
std_flux=[0.4,0.1,0.1,0.1]
std_data=std_temp+std_flux

mat = sio.loadmat('../Real_Data/Data_for_EKI.mat')
HFs =np.array(mat['HFs'])
T_surf_int =np.array(mat['T_surf_int'])
T_surf_ext =np.array(mat['T_surf_ext'])

Data=[]
sqrt_Gamma=[]
for n in range(num_steps_inv+1):
    Data=np.append(Data,[T_surf_int[n,1],T_surf_int[n,2],T_surf_int[n,0]])
    Data=np.append(Data,[T_surf_ext[n,1],T_surf_ext[n,2],T_surf_ext[n,0]])
    Data=np.append(Data,[HFs[n,1],HFs[n,0],HFs[n,2],HFs[n,3]])
    sqrt_Gamma=np.append(sqrt_Gamma,std_data)


mat = sio.loadmat('../Real_Data/Temps.mat')
T_int =mat['T_int']
T_ext =mat['T_ext']
Ti=T_int[:,0]
Te=T_ext[:,0]

SetFoldersFiles(ID)

#Ensemble size
N_En=1000

if N_En==1000:
    mat = sio.loadmat('../PriorEnsembles/Lev_en.mat')
else:
    mat = sio.loadmat('../PriorEnsembles/Lev_en_large.mat')
    
U=[]
Lev_en=mat['Lev']
if N_En==1000:
    mat = sio.loadmat('../PriorEnsembles/ModErr_en.mat')
else:
    mat = sio.loadmat('../PriorEnsembles/ModErr_en_large.mat')

Error_int_en=mat['Error_int']
print('sizeError', Error_int_en.shape)
Error_ext_en=mat['Error_ext']
U.append(writePrior_scalar(ID,1,'c_B',N_En,[4.0e4, 9.0e4],3.5e4,1.5e5))
U.append(writePrior_scalar(ID,1,'k_B',N_En,[0.019, 0.028],0.017,0.03))
U.append(writePrior_scalar(ID,1,'c_A',N_En,[4e5, 5e6],2e5,5.5e6))
U.append(writePrior_scalar(ID,1,'k_A',N_En,[2, 180],1.5,200))
U.append(writePrior_scalar(ID,1,'R_I',N_En,[0.019,0.24],0.018,0.28))
U.append(writePrior_scalar(ID,1,'R_E',N_En,[0.019,0.24],0.018,0.28))
w_en,Deltaw_en= writePrior_scalar_joint(ID,1,'Geo',N_En,[0.0,0.07],0,0.075,D[2])
U.append(w_en)
U.append(Deltaw_en)
#U.append(writePrior_scalar(ID,1,'beta',N_En,[-2.3,-1.4],-2.5,-1.3))
U.append(writePrior_scalar(ID,1,'beta',N_En,[0.5,1.9],0.3,2.0))
        
   

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
    f.attrs['std_data']=std_data
        
        
with h5py.File('Ensemble_'+ID+'.hdf5', 'a') as g:
    p=g.create_group('Unknown')
    p.create_dataset('Un_en', data=U)
    p.create_dataset('Level_en', data=Lev_en)
    p.create_dataset('Error_int_en', data=Error_int_en)
    p.create_dataset('Error_ext_en', data=Error_ext_en)
    inv=g.create_group('Inversion')
    inv.attrs['N_En']=N_En


    
mesh = BoxMesh(Point(0,0,0), Point(D[0],D[1],D[2]),Fenics_N[0],Fenics_N[1],Fenics_N[2])
temp=mesh.coordinates()
print('mesh size',temp.shape)
print('mesh size',mesh.num_cells())


    
    
    

inv_sqrt_Gamma=np.reciprocal(sqrt_Gamma)
#sio.savemat('data.mat', {'data':Data, 'inv_sqrt_Gamma':inv_sqrt_Gamma})
SaveData(ID,Data,inv_sqrt_Gamma)


#Data,TI_data,Flux_Data, sqrt_Gamma= GetPlain(ID,K_back,C_back,C_defe, 1/R_int,1/R_ext,sigma_TI, sigma_temp,sigma_flux, flag=1,flag2=1)
#inv_sqrt_Gamma=np.reciprocal(sqrt_Gamma)
#sio.savemat('data_plain.mat', {'data':Data, 'inv_sqrt_Gamma':inv_sqrt_Gamma,'Flux_Data':Flux_Data})

        
print('done')
