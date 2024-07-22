#!/usr/bin/env python
from __future__ import print_function
import h5py
import os
import sys
import numpy as np
import scipy.io as sio
#import matplotlib.pyplot as plt
import sys
from pyevtk.hl import imageToVTK
from Miscellaneous import get_scalar_box, get_scalar_low
ID='3D'


    
#with h5py.File('Ensemble_'+ID+'.hdf5', 'r') as f:
#    data = np.array(f['Data/data'])
mat = sio.loadmat('../Real_Data/Data_for_EKI.mat')
HFs =np.array(mat['HFs'])
T_surf_int =np.array(mat['T_surf_int'])
T_surf_ext =np.array(mat['T_surf_ext'])

std_temp=[0.1,0.1,0.1,0.1,0.1,0.1]
std_flux=[0.4,0.1,0.1,0.1]
std_data=std_temp+std_flux


with h5py.File('Ensemble_'+ID+'.hdf5', 'r') as f:
    inv=f['Inversion']
    N_En=inv.attrs['N_En']
    g=f['FwdModel']
    N=g.attrs['RF_N']
    D=g.attrs['D']
    D_reduced=g.attrs['D_reduced']
#    num_steps=g.attrs['num_steps_inv']
    num_steps=g.attrs['num_steps_pred']
    Level_en= f['Unknown/Level_en'][:]
    Error_int_en= f['Unknown/Error_int_en'][:]
    Error_ext_en= f['Unknown/Error_ext_en'][:]
    c_B=get_scalar_low(ID,'c_B',f['Unknown/Un_en'][0][:])
    k_B=get_scalar_low(ID,'k_B',f['Unknown/Un_en'][1][:])
    c_A=get_scalar_low(ID,'c_A',f['Unknown/Un_en'][2][:])
    k_A=get_scalar_low(ID,'k_A',f['Unknown/Un_en'][3][:])
    R_int=get_scalar_low(ID,'R_I',f['Unknown/Un_en'][4][:])
    R_ext=get_scalar_low(ID,'R_E',f['Unknown/Un_en'][5][:])
    w_en = f['Unknown/Un_en'][6][:]
    Deltaw_en = f['Unknown/Un_en'][7][:]
    beta =get_scalar_box(ID,'beta', f['Unknown/Un_en'][8][:])

Data=[]
sqrt_Gamma=[]
for n in range(num_steps+1):
    Data=np.append(Data,[T_surf_int[n,1],T_surf_int[n,2],T_surf_int[n,0]])
    Data=np.append(Data,[T_surf_ext[n,1],T_surf_ext[n,2],T_surf_ext[n,0]])
    Data=np.append(Data,[HFs[n,1],HFs[n,0],HFs[n,2],HFs[n,3]])
    sqrt_Gamma=np.append(sqrt_Gamma,std_data)
    
M=Data.shape[0]

with h5py.File('Ensemble_'+ID+'.hdf5', 'r') as f:
    g=f['Geo']
    a=g.attrs['min']
    b=g.attrs['max']

Deltaw=Deltaw_en
w=np.minimum(np.maximum(a, w_en) , b)


print(M)



pred_all=[]
flux_all=[]
for en in range(1000):
    mat=mat = sio.loadmat('Output_'+ID+'/pred_'+str(en+1)+'.mat')
    pred=mat['pred']
    pred_all=np.append(pred_all,pred)
    flux=mat['Flux']
    flux_all=np.append(flux_all,flux)

Fluxes=flux_all.reshape((int(flux_all.shape[0]/N_En),N_En),order='F').copy()
Pred=pred_all.reshape((int(pred_all.shape[0]/N_En),N_En),order='F').copy()

 
beta_mean=np.mean(beta, axis=1)
w_mean=np.mean(w,axis=1)
Deltaw_mean=np.mean(Deltaw,axis=1)
print('mean Geo Vales:',beta_mean,w_mean,Deltaw_mean)
LS_mean= np.mean(Level_en, axis=1)
temp=LS_mean.reshape(N[0],N[1],1,order='F').copy()

x = np.linspace(D_reduced[0], D_reduced[1], N[0])
y = np.linspace(D_reduced[2], D_reduced[3], N[1])
Anomaly=np.zeros((N[0]*N[1],1))

Ones=np.ones((N[0]*N[1],1))
ind=np.where (LS_mean>  beta_mean)
Anomaly[ind]=Ones[ind]

dx_dy_dz=[x[1]-x[0],y[1]-y[0],Deltaw_mean[0]]
origin=(D_reduced[0],D_reduced[2],w_mean[0])
print(dx_dy_dz)
print(origin)
imageToVTK('./Results/Region3D_real',spacing=(dx_dy_dz[0],dx_dy_dz[1],dx_dy_dz[2]),origin=(D_reduced[0],D_reduced[2],w_mean[0]),cellData={"AnomalyRegion": Anomaly.reshape(N[0],N[1],1,order='F').copy()})
imageToVTK('./Results/Level_set3D_real',spacing=(dx_dy_dz[0],dx_dy_dz[1],dx_dy_dz[2]),origin=(D_reduced[0],D_reduced[2],w_mean[0]),cellData={"LevelSetFunction": LS_mean.reshape(N[0],N[1],1,order='F').copy()})



sio.savemat('./Results/Visual_'+ID+'.mat', {'Pred':Pred,'Data':Data,'sqrt_Gamma':sqrt_Gamma,'K_values':np.vstack((k_B,k_A)),'C_values':np.vstack((c_B,c_A)),'R_int':R_int,'R_ext':R_ext,'w':w, 'Deltaw':Deltaw, 'beta':beta,'Lev_pos':Level_en, 'Ei':Error_int_en, 'E_e':Error_ext_en, 'Fluxes':Fluxes})


print('done')


    
