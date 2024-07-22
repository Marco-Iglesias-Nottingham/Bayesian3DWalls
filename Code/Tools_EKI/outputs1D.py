#!/usr/bin/env python
from __future__ import print_function
import h5py
import os
import sys
import numpy as np
import scipy.io as sio
#import matplotlib.pyplot as plt
import sys
from Miscellaneous import get_scalar, get_scalar_low

ID='1D'



with h5py.File('Ensemble_'+ID+'.hdf5', 'r') as f:
    inv=f['Inversion']
    N_En=inv.attrs['N_En']
    g=f['FwdModel']
    num_steps=g.attrs['num_steps_pred']
    k_en  = get_scalar(ID,'K', f['Unknown/Un_KC_en'][0])
    c_en =  get_scalar(ID,'C', f['Unknown/Un_KC_en'][1])
    R_I_en = get_scalar_low(ID,'R_I', f['Unknown/Un_R_en'][0])
    R_E_en = get_scalar_low(ID,'R_E', f['Unknown/Un_R_en'][1])

#Fluxes=np.zeros(( 2*(num_steps+1) ,N_En))

#for en in range(N_En):
#    mat=mat = sio.loadmat('Output_'+ID+'/pred_'+str(en+1)+'.mat')
#    Fluxes[:,en]=mat['pred'][0]

pred_all=[]
for en in range(10):
    mat=mat = sio.loadmat('Output_'+ID+'/pred_'+str(en+1)+'.mat')
    pred=mat['pred']
    pred_all=np.append(pred_all,pred)

print(pred_all.shape)
Fluxes=pred_all.reshape((int(pred_all.shape[0]/N_En),N_En),order='F').copy()
sio.savemat('./Results/FlowRates_pred_Equi.mat', {'FlowRates_Equi':Fluxes,'K_values':k_en,'C_values':c_en,'R_int':R_I_en,'R_ext':R_E_en})
#sio.savemat('Fluxes_pred_Equi.mat', {'K_values':k_en,'C_values':c_en,'R_int':R_I_en,'R_ext':R_E_en})

print('done')


    
