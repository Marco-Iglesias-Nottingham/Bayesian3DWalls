#Copyright (C) 2021: The University of Nottingham
#                 Author: Marco Iglesias
#This file contains utility functions used in EnKIMax

import numpy as np
import h5py
import os
import sys
import scipy.io as sio

def SetFoldersFiles(ID):
    os.system('rm Ensemble_'+ID+'.hdf5')
    os.system('rm -r Output_'+ID)
    os.system('mkdir Output_'+ID)
    os.system('rm ./Results/Results_'+ID+'.mat')
    os.system('rm Converged_'+ID+'.mat')


def SaveData(ID,Data,inv_sqrt_Gamma):
    with h5py.File('Ensemble_'+ID+'.hdf5', 'a') as g:
        f=g.create_group('Data')
        dset1 = f.create_dataset('data', data=Data)
        dset2 = f.create_dataset('inv_sqrt', data=inv_sqrt_Gamma)

def LoadData(ID):
    with h5py.File('Ensemble_'+ID+'.hdf5', 'r') as f:
        data = np.array(f['Data/data'])
        inv_sqrt_Gamma = np.array(f['Data/inv_sqrt'])
    return data, inv_sqrt_Gamma


def SavePriorForwardOneD(ID,D,Area,layers,N,Num_Ks,num_steps_inv,num_steps_pred,dt,Ti,Te,N_En,U_KC,U_R):
    with h5py.File('Ensemble_'+ID+'.hdf5', 'a') as g:
        f=g.create_group('FwdModel')
        f.attrs['ID']=ID
        f.attrs['D']=D
        f.attrs['layers']=layers
        f.attrs['N']=N
        f.attrs['Num_Ks']=Num_Ks
        f.attrs['Area']=Area
        f.attrs['num_steps_inv']=num_steps_inv
        f.attrs['num_steps_pred']=num_steps_pred
        f.attrs['dt']=dt
        f.attrs['Ti']=Ti
        f.attrs['Te']=Te

        inv=g.create_group('Inversion')
        inv.attrs['N_En']=N_En
        p=g.create_group('Unknown')
        p.create_dataset('Un_KC_en', data=U_KC)
        p.create_dataset('Un_R_en', data=U_R)


def get_scalar(ID,Filename,var):

    with h5py.File('Ensemble_'+ID+'.hdf5', 'r') as f:
        g=f[Filename]
        a=g.attrs['min']
        b=g.attrs['max']
    return a+(b-a)/(1+np.exp(var))

def get_scalar_box(ID,Filename,var):
    with h5py.File('Ensemble_'+ID+'.hdf5', 'r') as f:
        g=f[Filename]
        a=g.attrs['min']
        b=g.attrs['max']
    return np.minimum(np.maximum(a, var) , b)

def get_scalar_low(ID,Filename,var):
    with h5py.File('Ensemble_'+ID+'.hdf5', 'r') as f:
        g=f[Filename]
        a=g.attrs['min']
    return np.maximum(a,var)


def ExtractParameters(ID,task):
    if ID=='3D':
        with h5py.File('Ensemble_'+ID+'.hdf5', 'r') as f:
            g=f['FwdModel']
            RF_N=g.attrs['RF_N']
            Lev_en  = f['Unknown/Level_en'][:,task-1]
            Error_int_en = f['Unknown/Error_int_en'][:,task-1]
            Error_ext_en = f['Unknown/Error_ext_en'][:,task-1]
            c_B_en  = f['Unknown/Un_en'][0][0,task-1]
            k_B_en = f['Unknown/Un_en'][1][0,task-1]
            c_A_en = f['Unknown/Un_en'][2][0,task-1]
            k_A_en = f['Unknown/Un_en'][3][0,task-1]
            R_I_en = f['Unknown/Un_en'][4][0,task-1]
            R_E_en = f['Unknown/Un_en'][5][0,task-1]
            w_en = f['Unknown/Un_en'][6][0,task-1]
            Deltaw_en = f['Unknown/Un_en'][7][0,task-1]
            beta_en = f['Unknown/Un_en'][8][0,task-1]

        LS=Lev_en.reshape(RF_N[0],RF_N[1],order='F').copy()
        Error_int=Error_int_en.reshape(RF_N[0],RF_N[1],order='F').copy()
        Error_ext=Error_ext_en.reshape(RF_N[0],RF_N[1],order='F').copy()

        beta=get_scalar_box(ID,'beta',beta_en)
        with h5py.File('Ensemble_'+ID+'.hdf5', 'r') as f:
            g=f['Geo']
            a=g.attrs['min']
            b=g.attrs['max']

        Deltaw=Deltaw_en
        w=min(max(a, w_en) , b)

        k_B=get_scalar_low(ID,'k_B',k_B_en)
        k_A=get_scalar_low(ID,'k_A',k_A_en)
        c_B=get_scalar_low(ID,'c_B',c_B_en)
        c_A=get_scalar_low(ID,'c_A',c_A_en)
        R_int=get_scalar_low(ID,'R_I',R_I_en)
        R_ext=get_scalar_low(ID,'R_E',R_E_en)
    elif ID=='3D_syn':
        with h5py.File('Ensemble_'+ID+'.hdf5', 'r') as f:
            g=f['FwdModel']
            RF_N=g.attrs['RF_N']
            Lev_en1  = f['Unknown/Level_en1'][:,task-1]
            Lev_en2  = f['Unknown/Level_en2'][:,task-1]
            c_B1_en  = f['Unknown/Un_en'][0][0,task-1]
            c_B2_en  = f['Unknown/Un_en'][1][0,task-1]
            c_B3_en  = f['Unknown/Un_en'][2][0,task-1]
            k_B1_en = f['Unknown/Un_en'][3][0,task-1]
            k_B2_en = f['Unknown/Un_en'][4][0,task-1]
            k_B3_en = f['Unknown/Un_en'][5][0,task-1]
            c_A1_en = f['Unknown/Un_en'][6][0,task-1]
            c_A2_en = f['Unknown/Un_en'][7][0,task-1]
            k_A1_en = f['Unknown/Un_en'][8][0,task-1]
            k_A2_en = f['Unknown/Un_en'][9][0,task-1]
            R_I_en = f['Unknown/Un_en'][10][0,task-1]
            R_E_en = f['Unknown/Un_en'][11][0,task-1]
            w1_en = f['Unknown/Un_en'][12][0,task-1]
            w2_en = f['Unknown/Un_en'][13][0,task-1]
            Deltaw1_en = f['Unknown/Un_en'][14][0,task-1]
            Deltaw2_en = f['Unknown/Un_en'][15][0,task-1]
            beta1_en = f['Unknown/Un_en'][16][0,task-1]
            beta2_en = f['Unknown/Un_en'][17][0,task-1]

        LS1=Lev_en1.reshape(RF_N[0],int(RF_N[1]/2),order='F').copy()
        LS2=Lev_en2.reshape(RF_N[0],int(RF_N[1]/2),order='F').copy()
        beta1=get_scalar_box(ID,'beta1',beta1_en)
        beta2=get_scalar_box(ID,'beta2',beta2_en)

        with h5py.File('Ensemble_'+ID+'.hdf5', 'r') as f:
            g=f['Geo1']
            a=g.attrs['min']
            b=g.attrs['max']


        Deltaw1=Deltaw1_en
        w1=min(max(a, w1_en) , b)
        Deltaw2=Deltaw2_en
        w2=min(max(a, w2_en) , b)


        k_B1=get_scalar_box(ID,'k_B1',k_B1_en)
        k_B2=get_scalar_box(ID,'k_B2',k_B2_en)
        k_B3=get_scalar_box(ID,'k_B3',k_B3_en)
        k_A1=get_scalar_box(ID,'k_A1',k_A1_en)
        k_A2=get_scalar_box(ID,'k_A2',k_A2_en)
        c_B1=get_scalar_box(ID,'c_B1',c_B1_en)
        c_B2=get_scalar_box(ID,'c_B2',c_B2_en)
        c_B3=get_scalar_box(ID,'c_B3',c_B3_en)
        c_A1=get_scalar_box(ID,'c_A1',c_A1_en)
        c_A2=get_scalar_box(ID,'c_A2',c_A1_en)
        R_int=get_scalar_box(ID,'R_I',R_I_en)
        R_ext=get_scalar_box(ID,'R_E',R_E_en)
        
        k_B=[k_B1,k_B2,k_B3]
        c_B=[c_B1,c_B2,c_B3]
        k_A=[k_A1,k_A2]
        c_A=[c_A1,c_A2]
        w=[w1,w2]
        Deltaw=[Deltaw1,Deltaw2]
        beta=[beta1,beta2]
        LS=[LS1,LS2]
    

        Error_int=np.zeros((RF_N[0],RF_N[1]))
        Error_ext=np.zeros((RF_N[0],RF_N[1]))
    return k_B,k_A,c_B,c_A,LS, Error_int,Error_ext, R_int,R_ext, w,Deltaw,beta


'''beta_old=beta
beta_low=beta
tuning=0.1

RHS=np.dot(Delta_Z.T,Z_m)
newMat=1/beta*np.dot(Delta_Z.T,Delta_Z)+np.identity(N_En)
Inter = np.linalg.solve(newMat,RHS)
xx=1/beta*Z_m-1/(beta**2)*np.dot(Delta_Z,Inter)
TT=tuning*np.linalg.norm(Z_m)-beta*np.linalg.norm(xx)
print(TT)
print(tuning*np.linalg.norm(Z_m))
print(beta*np.linalg.norm(xx))

while TT>0:
    print('Cabron')
    print(beta)
    beta_low=beta
    beta=2*beta
    newMat=1/beta*np.dot(Delta_Z.T,Delta_Z)+np.identity(N_En)
    Inter = np.linalg.solve(newMat,RHS)
    xx=1/beta*Z_m-1/(beta**2)*np.dot(Delta_Z,Inter)
    TT=tuning*np.linalg.norm(Z_m)-beta*np.linalg.norm(xx)
alpha=beta
'''
