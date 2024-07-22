#!/usr/bin/env python
from __future__ import print_function
import h5py
import os
import sys
import numpy as np
import scipy.io as sio
#import matplotlib.pyplot as plt
import sys
from Miscellaneous import  LoadData
from dolfin import *



ID=sys.argv[1]
data, inv_sqrt_Gamma= LoadData(ID)

def UpdateKalman(En,Delta_z,A,N_En):
    En_mean=np.mean(En, axis=1)
    Mat5=1/np.sqrt(N_En-1)*(En-En_mean[:,np.newaxis])
    CrossCov=np.dot(Mat5 , Delta_z.T)
    En+=-np.dot(CrossCov,A)
    return En

M=data.size
print(inv_sqrt_Gamma.shape)

if ID=='1D' or ID=='3D':
    filename='./Results/Results_'+ID+'.mat'
else:
    filename='./Results_Syn/Results_'+ID+'.mat'

with h5py.File('Ensemble_'+ID+'.hdf5', 'r') as f:
    inv=f['Inversion']
    N_En=inv.attrs['N_En']
    g=f['FwdModel']
    if ID=='1D' or ID=='1D_syn':
        Un_KC=f['Unknown/Un_KC_en'][:]
        Un_R=f['Unknown/Un_R_en'][:]
        A=np.diag(inv_sqrt_Gamma[0,:])
        Batch=10
        
    elif ID=='3D':
        Level_en= f['Unknown/Level_en'][:]
        Error_int_en= f['Unknown/Error_int_en'][:]
        print('sizeError', Error_int_en.shape)
        Error_ext_en= f['Unknown/Error_ext_en'][:]
        Un=f['Unknown/Un_en'][:]
        A=np.diag(inv_sqrt_Gamma)
        Batch=1000
    else:
        Level_en1= f['Unknown/Level_en1'][:]
        Level_en2= f['Unknown/Level_en2'][:]
        Un=f['Unknown/Un_en'][:]
        A=np.diag(inv_sqrt_Gamma)
        Batch=1000
    
    
exists = os.path.exists(filename)
if exists:
    mat = sio.loadmat(filename)
    t=list(mat['t'][0])
    Misfit_ave=list(mat['Misfit_ave'][0])
    iter=int(mat['iter'])
    np.random.seed(400*iter)
else:
    iter=0
    t=[]
    t.append(iter)
    Misfit_ave =[]
    np.random.seed(20)


Z=np.zeros((M,N_En))
time=np.zeros((1,N_En))
pred_all=[]
time_all=[]

for en in range(Batch):
    name1='Output_'+ID+'/pred_'+str(en+1)+'.mat'
    mat = sio.loadmat(name1)
    pred=mat['pred']
    time=mat['time']
    pred_all=np.append(pred_all,pred)
    time_all=np.append(time_all,time)



result=pred_all.reshape((int(pred_all.shape[0]/N_En),N_En),order='F').copy()
#A=np.diag(inv_sqrt_Gamma[0,:])

print(A.shape, result.shape,data.shape,ID)
if ID=='1D' or ID=='1D_syn':
    Z=np.dot(A, result-data.T )
else:
    Z=np.dot(A, result-data[:,np.newaxis] )
    

#print(Z.shape)
#    print(time.shape,result2.shape,data.shape)
#    check=inv_sqrt_Gamma*( result-data )

#for en in range(N_En):
#    name1='Output_'+ID+'/pred_'+str(en+1)+'.mat'
#    mat = sio.loadmat(name1)
#    result=mat['pred']
#    time=mat['time']
#    print(time.shape,result.shape)
#    result2=np.reshape(result,(int(result.shape[1]/50),50))
#    print(time.shape,result2.shape,data.shape)
#    check=inv_sqrt_Gamma*( result-data )
#    print(check.shape)
#    Z[:,en]=inv_sqrt_Gamma*( result[:,en]-data )

#for en in range(200):
#    name1='Output_'+ID+'/pred_'+str(en+1)+'.mat'
#    mat = sio.loadmat(name1)
#    result=mat['pred']
#    time=mat['time']
#    print(time.shape,result.shape)
#    result2=np.reshape(result,(int(result.shape[1]/50),50))
#    print(time.shape,result2.shape,data.shape)
#    check=inv_sqrt_Gamma*( result-data )
#    print(check.shape)
 #   Z[:,en]=inv_sqrt_Gamma*( result-data )

#sio.savemat('DebugV2.mat', {'ZV2':Z})

Z_mean=np.mean(Z, axis=1)
time_mean=np.mean(time_all, axis=0)
print('mean time',time_mean)
Mis=np.sum(Z_mean**2)/M


Z_norm=np.linalg.norm(Z, axis=0)

alpha=1/M*np.mean(Z_norm**2)
print('alpha',alpha)
Misfit_ave.append(Mis)
print(Misfit_ave)

if t[iter]+1/alpha>1:
    alpha=1/(1-t[iter])


        
E=np.sqrt(alpha)*np.random.randn(M,N_En)
E_mean=np.mean(E, axis=1)
E=E-E_mean[:,np.newaxis]
Delta_Z=np.sqrt(1/(N_En-1))*(Z-Z_mean[:,np.newaxis])
C=np.dot( Delta_Z,Delta_Z.T)
Mat4 = np.linalg.solve(C+alpha*np.identity(M),Z+E)


 



#Mat0=np.dot(Delta_Z.T,Z+E)
#Mat1=1/alpha*np.dot(Delta_Z.T,Delta_Z)+np.identity(N_En)
#Mat2 = np.linalg.solve(Mat1,Mat0)
#Mat3=1/alpha*(Z+E)-1/(alpha**2)*np.dot(Delta_Z,Mat2)
#Mat4=np.dot(Delta_Z.T,Mat3)

#sio.savemat('B.mat', {'Mat4':Mat4})#,'Err':Err})


print(iter)
print(Misfit_ave)

if ID=='1D' or ID=='1D_syn':
    Un_KC[:] = [UpdateKalman(u,Delta_Z,Mat4,N_En) for u in Un_KC]
    Un_R[:] = [UpdateKalman(u,Delta_Z,Mat4,N_En) for u in Un_R]
    with h5py.File('Ensemble_'+ID+'.hdf5', 'a') as f:
        f['Unknown/Un_KC_en'][...]=Un_KC
        f['Unknown/Un_R_en'][...]=Un_R
elif ID=='3D':
    Level_en=UpdateKalman(Level_en,Delta_Z,Mat4,N_En)
    Error_int_en=UpdateKalman(Error_int_en,Delta_Z,Mat4,N_En)
    Error_ext_en=UpdateKalman(Error_ext_en,Delta_Z,Mat4,N_En)
    Un[:] = [UpdateKalman(u,Delta_Z,Mat4,N_En) for u in Un]
    with h5py.File('Ensemble_'+ID+'.hdf5', 'a') as f:
        f['Unknown/Level_en'][...]=Level_en
        f['Unknown/Error_int_en'][...]=Error_int_en
        f['Unknown/Error_ext_en'][...]=Error_ext_en
        f['Unknown/Un_en'][...]=Un
else:
    Level_en1=UpdateKalman(Level_en1,Delta_Z,Mat4,N_En)
    Level_en2=UpdateKalman(Level_en2,Delta_Z,Mat4,N_En)
    Un[:] = [UpdateKalman(u,Delta_Z,Mat4,N_En) for u in Un]
    with h5py.File('Ensemble_'+ID+'.hdf5', 'a') as f:
        f['Unknown/Level_en1'][...]=Level_en1
        f['Unknown/Level_en2'][...]=Level_en2
        f['Unknown/Un_en'][...]=Un




t.append(t[iter]+1/alpha)



if (np.abs(t[iter]+1/alpha-1)<1e-3):
    print('Converged!!!')
    sio.savemat('Converged_'+ID+'.mat', {'Misfit_ave':Misfit_ave, 't':t,'iter':iter,'alpha':alpha,'time':time,'Z':Z})


iter=iter+1
sio.savemat(filename, {'Misfit_ave':Misfit_ave, 't':t,'iter':iter,'alpha':alpha})


print('Finish iteration')
print(t)
#os.system('dijitso clean')

flag=1
print('done')


