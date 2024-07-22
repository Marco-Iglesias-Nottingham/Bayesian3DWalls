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
from Miscellaneous import get_scalar_box, get_scalar_low, get_scalar

ID='3D_syn'




with h5py.File('Ensemble_'+ID+'.hdf5', 'r') as f:
    inv=f['Inversion']
    N_En=inv.attrs['N_En']
    g=f['FwdModel']
    N=g.attrs['RF_N']
    D=g.attrs['D']
    D_reduced=g.attrs['D_reduced']
    num_steps=g.attrs['num_steps_pred']
    Level_en1= f['Unknown/Level_en1'][:]
    Level_en2= f['Unknown/Level_en2'][:]
    c_B1=get_scalar_box(ID,'c_B1',f['Unknown/Un_en'][0][:])
    c_B2=get_scalar_box(ID,'c_B2',f['Unknown/Un_en'][1][:])
    c_B3=get_scalar_box(ID,'c_B3',f['Unknown/Un_en'][2][:])
    k_B1=get_scalar_box(ID,'k_B1',f['Unknown/Un_en'][3][:])
    k_B2=get_scalar_box(ID,'k_B2',f['Unknown/Un_en'][4][:])
    k_B3=get_scalar_box(ID,'k_B3',f['Unknown/Un_en'][5][:])
    c_A1=get_scalar_box(ID,'c_A1',f['Unknown/Un_en'][6][:])
    c_A2=get_scalar_box(ID,'c_A2',f['Unknown/Un_en'][7][:])
    k_A1=get_scalar_box(ID,'k_A1',f['Unknown/Un_en'][8][:])
    k_A2=get_scalar_box(ID,'k_A2',f['Unknown/Un_en'][9][:])
    R_int=get_scalar_box(ID,'R_I',f['Unknown/Un_en'][10][:])
    R_ext=get_scalar_box(ID,'R_E',f['Unknown/Un_en'][11][:])
    w_en1 = f['Unknown/Un_en'][12][:]
    w_en2 = f['Unknown/Un_en'][13][:]
    Deltaw1 = f['Unknown/Un_en'][14][:]
    Deltaw2 = f['Unknown/Un_en'][15][:]
    beta1 =get_scalar_box(ID,'beta1', f['Unknown/Un_en'][16][:])
    beta2 =get_scalar_box(ID,'beta2', f['Unknown/Un_en'][17][:])


#M=Data.shape[0]
#inv_sqrt_Gamma=np.reciprocal(sqrt_Gamma)
#sio.savemat('data_pred.mat', {'data':Data, 'inv_sqrt_Gamma':inv_sqrt_Gamma})

with h5py.File('Ensemble_'+ID+'.hdf5', 'r') as f:
    g=f['Geo1']
    a=g.attrs['min']
    b=g.attrs['max']


w1=np.minimum(np.maximum(a, w_en1) , b)
w2=np.minimum(np.maximum(a, w_en2) , b)




pred_all=[]
flux_all=[]


for en in range(1000):
    mat=mat = sio.loadmat('Output_'+ID+'/pred_'+str(en+1)+'.mat')
    pred=mat['pred']
    pred_all=np.append(pred_all,pred)
    flux=mat['Flux']
    flux_all=np.append(flux_all,flux)

 
Fluxes=flux_all.reshape((int(flux_all.shape[0]/N_En),N_En),order='F').copy()
Data=pred_all.reshape((int(pred_all.shape[0]/N_En),N_En),order='F').copy()

beta_mean1=np.mean(beta1, axis=1)
w_mean1=np.mean(w1,axis=1)
Deltaw_mean1=np.mean(Deltaw1,axis=1)
print('mean Geo Vales:',beta_mean1,w_mean1,Deltaw_mean1)
LS_mean1= np.mean(Level_en1, axis=1)
print('LS_shape:',LS_mean1.shape)

print('Dreduced1',D_reduced[0])
x = np.linspace(D_reduced[0][0], D_reduced[0][1], N[0])
y = np.linspace(D_reduced[0][2], D_reduced[0][3], int(N[1]/2))
Anomaly1=np.zeros((N[0]*int(N[1]/2),1))

Ones=np.ones((N[0]*int(N[1]/2),1))
ind=np.where (LS_mean1>  beta_mean1)
Anomaly1[ind]=Ones[ind]

dx_dy_dz=[x[1]-x[0],y[1]-y[0],Deltaw_mean1[0]]
origin=(D_reduced[0][0],D_reduced[0][2],w_mean1[0])
print('o1:',origin)
print('dx_dy_dz1',dx_dy_dz)

imageToVTK('./Results_Syn/level_set1_syn',spacing=(dx_dy_dz[0],dx_dy_dz[1],dx_dy_dz[2]),origin=(D_reduced[0][0],D_reduced[0][2],w_mean1[0]),cellData={"AnomalyRegion": Anomaly1.reshape(N[0],int(N[1]/2),1,order='F').copy()})


beta_mean2=np.mean(beta2, axis=1)
w_mean2=np.mean(w2,axis=1)
Deltaw_mean2=np.mean(Deltaw2,axis=1)
print('mean Geo Vales:',beta_mean2,w_mean2,Deltaw_mean2)
LS_mean2= np.mean(Level_en2, axis=1)

print('Dreduced2',D_reduced[1])
x = np.linspace(D_reduced[1][0], D_reduced[1][1], N[0])
y = np.linspace(D_reduced[1][2], D_reduced[1][3], int(N[1]/2))
Anomaly2=np.zeros((N[0]*int(N[1]/2),1))

Ones=np.ones((N[0]*int(N[1]/2),1))
ind=np.where (LS_mean2>  beta_mean2)
Anomaly2[ind]=Ones[ind]

dx_dy_dz=[x[1]-x[0],y[1]-y[0],Deltaw_mean2[0]]
print('dx_dy_dz2',dx_dy_dz)
origin=(D_reduced[1][0],D_reduced[1][2],w_mean2[0])
print('o2:',origin)
imageToVTK('./Results_Syn/level_set2_syn',spacing=(dx_dy_dz[0],dx_dy_dz[1],dx_dy_dz[2]),origin=(D_reduced[1][0],D_reduced[1][2],w_mean2[0]),cellData={"AnomalyRegion": Anomaly2.reshape(N[0],int(N[1]/2),1,order='F').copy()})


sio.savemat('./Results_Syn/Visual_'+ID+'.mat', {'Data':Data,'K_values':np.vstack((k_B1,k_B2,k_B3,k_A1,k_A2)),'C_values':np.vstack((c_B1,c_B2,c_B3,c_A1,c_A2)),'R_int':R_int,'R_ext':R_ext,'w':np.vstack((w1,w2)) ,'Deltaw':np.vstack((Deltaw1,Deltaw2)),'beta':np.vstack((beta1,beta2)),'Lev_pos1':Level_en1, 'Lev_pos2': Level_en2 ,'Fluxes':Fluxes})




print('done')


    
