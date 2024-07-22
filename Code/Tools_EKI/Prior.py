#Copyright (C) 2021: The University of Nottingham
#                 Author: Marco Iglesias
#This file contains the function to sample the prior ensemble used in EnKIMax

import numpy as np
import pyDOE
import h5py




def writePrior_scalar(ID,N,FieldName,N_En,Range,min,max):
    #print(FieldName)
    X=pyDOE.lhs(N, samples=N_En)
    #produce samples from the uniform prior (need to use exponentiate first) of lengthscales using latin hypercube sampling
    K=Range[0]+(Range[1]-Range[0])*X
    

    with h5py.File('Ensemble_'+ID+'.hdf5', 'a') as g:
        f=g.create_group(FieldName)
        f.attrs['min']=min
        f.attrs['max']=max
    return K.T
            
            
def writePrior_scalar_joint(ID,N,FieldName,N_En,Range,min,max,D):
    #print(FieldName)
    X=pyDOE.lhs(N, samples=N_En)
    #produce samples from the uniform prior (need to use exponentiate first) of lengthscales using latin hypercube sampling
    w=Range[0]+(Range[1]-Range[0])*X
#    if ID=='3D':
#        B=np.random.beta(5, 5, size=[N_En,1])
#    else:
    B=np.random.beta(2, 1, size=[N_En,1])
    Deltaw=  np.multiply(D-w,B);

    with h5py.File('Ensemble_'+ID+'.hdf5', 'a') as g:
        f=g.create_group(FieldName)
        f.attrs['min']=min
        f.attrs['max']=max


    return w.T,Deltaw.T


def writePrior_scalar_transform(ID,N,FieldName,N_En,Range,min,max):
    X=pyDOE.lhs(N, samples=N_En)
    #produce samples from the uniform prior (need to use exponentiate first) of lengthscales using latin hypercube sampling
    K=Range[0]+(Range[1]-Range[0])*X
    log_K =np.log((max-K)/(K-min))
    
    with h5py.File('Ensemble_'+ID+'.hdf5', 'a') as g:
        f=g.create_group(FieldName)
        f.attrs['min']=min
        f.attrs['max']=max

    return log_K.T
