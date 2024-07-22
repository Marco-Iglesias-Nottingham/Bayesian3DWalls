
from __future__ import print_function
from fenics import *
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.interpolate import griddata
from scipy.interpolate import interp2d
import time
import ufl
import scipy.io as sio

import numpy as np

class TwoDfield(UserExpression):
    def __init__(self,  xv,yv,RF, **kwargs):
        super().__init__(**kwargs)
        self.xv, self.yv = xv,yv
        self.RF=RF
    def value_shape(self):
        return ()
        
    def eval(self, value, x):
        o1=np.where(self.xv <= x[0] )
        o2=np.where(self.yv <= x[1] )
        value[0]=self.RF[o1[0][-1],o2[0][-1]]
        
class OneDThreeLayersThermalProperties(UserExpression):
    def __init__(self,layers,prop_values, **kwargs):
        super().__init__(**kwargs)
        self.prop_values=prop_values
        self.layers=layers

    def value_shape(self):
        return ()
    def eval(self, value, x):
        if 0<=x[0]<self.layers[0]:
            value[0] = self.prop_values[0]
        elif self.layers[0]<x[0]<self.layers[1]:
            value[0] = self.prop_values[1]
        else:
            value[0] = self.prop_values[2]

class OneDFourLayersThermalProperties(UserExpression):
    def __init__(self,layers,prop_values, **kwargs):
        super().__init__(**kwargs)
        self.prop_values=prop_values
        self.layers=layers

    def value_shape(self):
        return ()
    def eval(self, value, x):
        if 0<=x[0]<self.layers[0]:
            value[0] = self.prop_values[0]
        elif self.layers[0]<x[0]<self.layers[1]:
            value[0] = self.prop_values[1]
        elif self.layers[1]<x[0]<self.layers[2]:
            value[0] = self.prop_values[2]
        else:
            value[0] = self.prop_values[3]

class OneDFiveLayersThermalProperties(UserExpression):
    def __init__(self,layers,prop_values, **kwargs):
        super().__init__(**kwargs)
        self.prop_values=prop_values
        self.layers=layers

    def value_shape(self):
        return ()
    def eval(self, value, x):
        if 0<=x[0]<self.layers[0]:
            value[0] = self.prop_values[0]
        elif self.layers[0]<x[0]<self.layers[1]:
            value[0] = self.prop_values[1]
        elif self.layers[1]<x[0]<self.layers[2]:
            value[0] = self.prop_values[2]
        elif self.layers[2]<x[0]<self.layers[3]:
            value[0] = self.prop_values[3]
        else:
            value[0] = self.prop_values[4]


class ThreeDThermalProperties(UserExpression):
    def __init__(self, xv,yv,D,Level,defe,back,beta,w,Deltaw, **kwargs):
        super().__init__(**kwargs)
        self.xv, self.yv = xv,yv
        self.back, self.defe=back,defe
        self.Level=Level
        self.D=D
        self.w, self.Deltaw, self.beta=w, Deltaw,beta
        
    def value_shape(self):
        return ()

    def eval(self, value, x):
        if  (self.D[2]<x[1]<self.D[3])&(self.D[0]<x[0]<self.D[1]):
            idx=np.where(self.xv <= x[0] )
            idy=np.where(self.yv <= x[1] )
            Lev=self.Level[idx[0][-1],idy[0][-1]]
            if (Lev>self.beta) & (self.w<=x[2]<=self.w+self.Deltaw):
                value[0]=self.defe
            else:
                value[0]=self.back
        else:
            value[0]=self.back

class ThreeDComplexCase(UserExpression):
    def __init__(self, xv1,yv1,xv2,yv2,D,layers,Level,defe,back,beta,w,Deltaw, **kwargs):
        super().__init__(**kwargs)
        self.xv1, self.yv1 = xv1,yv1
        self.xv2, self.yv2 = xv2,yv2
        self.back, self.defe=back,defe
        self.Level1=Level[0]
        self.Level2=Level[1]
        self.D1=D[0]
        self.D2=D[1]
        self.layers=layers
        self.w, self.Deltaw, self.beta=w, Deltaw,beta
    def value_shape(self):
        return ()

    def eval(self, value, x):
        if 0<=x[2]<self.layers[0]:
            back = self.back[0]
        elif self.layers[0]<x[2]<self.layers[1]:
            back = self.back[1]
        else:
            back = self.back[2]

        if  (self.D1[2]<x[1]<self.D1[3])&(self.D1[0]<x[0]<self.D1[1]):
            idx=np.where(self.xv1 <= x[0] )
            idy=np.where(self.yv1 <= x[1] )
            Lev=self.Level1[idx[0][-1],idy[0][-1]]
            if (Lev>self.beta[0]) & (self.w[0]<=x[2]<=self.w[0]+self.Deltaw[0]):
                value[0]=self.defe[0]
            else:
                value[0]=back
        elif  (self.D2[2]<x[1]<self.D2[3])&(self.D2[0]<x[0]<self.D2[1]):
            idx=np.where(self.xv2 <= x[0] )
            idy=np.where(self.yv2 <= x[1] )
            Lev=self.Level2[idx[0][-1],idy[0][-1]]
            if (Lev>self.beta[1]) & (self.w[1]<=x[2]<=self.w[1]+self.Deltaw[1]):
                value[0]=self.defe[1]
            else:
                value[0]=back
        else:
            value[0]=back


class ThreeDThermalPropertiesThreeLayersTruth(UserExpression):
    def __init__(self, layers,back,defe, **kwargs):
        super().__init__(**kwargs)
        self.back, self.defe=back,defe
        self.layers=layers
    def value_shape(self):
        return ()
    def eval(self, value, x):

# 0.13<x[2]<0.23 :
 #0.15<x[2]<0.25

        if  (x[0]-0.75)**2+(x[1]-0.5)**2 <= 0.3**2  and 0.13<x[2]<0.23 :
            value[0]=self.defe[0]
        elif (0.12<x[2]<0.25 and 0.6<x[0]<1.4 and 1.2<x[1]<1.45):
            value[0]=self.defe[1]
        else:
            if x[2]<=self.layers[0]:
                value[0]=self.back[0]
            elif self.layers[0]<x[2]<self.layers[1]:
                value[0]=self.back[1]
            elif x[2]>=self.layers[1]:
                value[0]=self.back[2]
                

class ThreeDThermalPropertiesThreeLayersPlain(UserExpression):
    def __init__(self, layers,back,defe, **kwargs):
        super().__init__(**kwargs)
        self.back, self.defe=back,defe
        self.layers=layers
    def value_shape(self):
        return ()
    def eval(self, value, x):
        if x[2]<=self.layers[0]:
            value[0]=self.back[0]
        elif self.layers[0]<x[2]<self.layers[1]:
            value[0]=self.back[1]
        elif x[2]>=self.layers[1]:
            value[0]=self.back[2]


np.random.seed(1087)
def set_boundary(D,mesh):
    class R1(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[2]) < 1E-14

    class R2(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[2]) > D[2]-1E-14

    boundary_markers= MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    boundary_markers.set_all(9999)
    bR1 = R1()
    bR2 = R2()
    bR1.mark(boundary_markers, 0)
    bR2.mark(boundary_markers, 1)
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
    boundarymesh = BoundaryMesh(mesh, 'exterior')
    bdim = boundarymesh.topology().dim()
    boundary_boundaries = MeshFunction('size_t', boundarymesh, bdim)
    boundary_boundaries.set_all(0)
    for i, facet in enumerate(entities(boundarymesh, bdim)):
        parent_meshentity = boundarymesh.entity_map(bdim)[i]
        parent_boundarynumber = boundary_markers.array()[parent_meshentity]
        boundary_boundaries.array()[i] = parent_boundarynumber

    submesh1 = SubMesh(boundarymesh, boundary_boundaries, 1) # works
    submesh0 = SubMesh(boundarymesh, boundary_boundaries, 0) # works

    return ds, submesh0, submesh1


def boundary(x, on_boundary):
    return on_boundary



    
 

def Heat_predictions(ID,LS,Error_int,Error_ext, k_A,k_B,c_A,c_B,beta,w,Deltaw, a_int,a_ext,flag):
    tol = 1e-14
    n0=Constant((0.0,0.0,-1.0))
    n1=Constant((0.0,0.0,1.0))
    with h5py.File('Ensemble_'+ID+'.hdf5', 'r') as f:
        g=f['FwdModel']
        D=g.attrs['D']
        if flag==0:
            num_steps=g.attrs['num_steps_inv']
        else:
            num_steps=g.attrs['num_steps_pred']
        RF_N=g.attrs['RF_N']
        layers=g.attrs['layers']
        Fenics_N=g.attrs['Fenics_N']
        D=g.attrs['D']
        D_reduced=g.attrs['D_reduced']
        dt=g.attrs['dt']
        Po_temp=g.attrs['Po_temp']
        Po_flux=g.attrs['Po_flux']
        T_i=g.attrs['Ti']
        T_e=g.attrs['Te']

    
    mesh = BoxMesh(Point(0,0,0), Point(D[0],D[1],D[2]),Fenics_N[0],Fenics_N[1],Fenics_N[2])
    ds, submesh0, submesh1=set_boundary(D,mesh)
    if flag==1:
        Vb0 = FunctionSpace(submesh0, 'CG', 1)
        num0 = Vb0.dim()
        ub0 = Function(Vb0)
        Vb1 = FunctionSpace(submesh1, 'CG', 1)
        num1 = Vb1.dim()
        ub1 = Function(Vb1)
        ds1 = Measure('dx', domain=submesh1)
        ds0 = Measure('dx', domain=submesh0)


    V = FunctionSpace(mesh, 'Lagrange', 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    
    x = np.linspace(0, D[0], RF_N[0])
    y = np.linspace(0, D[1], RF_N[1])


    if len(layers)>1:
        x1 = np.linspace(D_reduced[0][0], D_reduced[0][1], RF_N[0])
        y1 = np.linspace(D_reduced[0][2], D_reduced[0][3], int(RF_N[1]/2))
        x2 = np.linspace(D_reduced[1][0], D_reduced[1][1], RF_N[0])
        y2 = np.linspace(D_reduced[1][2], D_reduced[1][3], int(RF_N[1]/2))
        kappa = ThreeDComplexCase(x1,y1,x2,y2,D_reduced, layers,LS,k_A,k_B,beta,w,Deltaw,degree=1)
        c = ThreeDComplexCase(x1,y1,x2,y2,D_reduced,layers,LS,c_A,c_B,beta,w,Deltaw,degree=1)

        #kappa = ThreeDThermalPropertiesThreeLayersTruth(layers,k_B,k_A,degree=1)
        #c = ThreeDThermalPropertiesThreeLayersTruth(layers,c_B,c_A,degree=1)
        


    else:
        x2 = np.linspace(D_reduced[0], D_reduced[1], RF_N[0])
        y2 = np.linspace(D_reduced[2], D_reduced[3], RF_N[1])

        kappa = ThreeDThermalProperties(x2,y2,D_reduced, LS,k_A,k_B,beta,w,Deltaw,degree=1)
        c = ThreeDThermalProperties(x2,y2,D_reduced,LS,c_A,c_B,beta,w,Deltaw,degree=1)

    K =interpolate(kappa, V)
    C = interpolate(c, V)
#    vtkfile = File('Test/perm.pvd')
#    vtkfile << K
#    vtkfile = File('Test/c.pvd')
#    vtkfile << C

    E_i = TwoDfield(x,y,Error_int,degree=1)
    E_e = TwoDfield(x,y,Error_ext,degree=1)
    
  
    alpha_int =Constant(a_int)
    alpha_ext =Constant(a_ext)
    a1=alpha_int*u*v*ds(0)+alpha_ext*u*v*ds(1)

    a = dot(K*grad(u), grad(v))*dx+alpha_int*u*v*ds(0)+alpha_ext*u*v*ds(1)
    A=assemble(a)
    u = Function(V)
    n=0
    Ti=Constant(T_i[0])
    Te=Constant(T_e[0])
    
    
    L = alpha_int*Ti*v*ds(0)+ alpha_ext*Te*v*ds(1)+E_i*v*ds(0)+E_e*v*ds(1)
 
    b=assemble(L)
    solver = KrylovSolver('gmres', 'ilu')
    prm = solver.parameters

    prm["absolute_tolerance"] = 1E-12
    prm["relative_tolerance"] = 1E-10
    prm["maximum_iterations"]  = 50000    #solve(A, u.vector(), b,'gmres','ilu')
    solver.solve(A, u.vector(), b)
    Flux_Data=[]
    if flag==1:
        Ei =interpolate(E_i, Vb0)
        Ee =interpolate(E_e, Vb1)
        ub0.interpolate(u)
        NF_int=np.array(ub0.vector())
        ub1.interpolate(u)
        NF_ext=np.array(ub1.vector())
        Flux_int=np.zeros(num0,)
        Flux_ext=np.zeros(num1,)
        for jj in range(num0):
            Flux_int[jj]=a_int*(NF_int[jj]-T_i[n] )
        for jj in range(num1):
            Flux_ext[jj]=a_ext*(NF_ext[jj]-T_e[n] )
        ub0.vector()[:]=Flux_int
        ub1.vector()[:]=Flux_ext
        Flux_Data=[]
        flux_int =assemble((ub0-Ei)*ds0)
        flux_ext =assemble((ub1-Ee)*ds1)
        Flux_Data.append(flux_int)
        Flux_Data.append(flux_ext)

    Data=np.array([])
    
    for p in Po_temp:
        Data=np.append(Data,u(p[0],p[1],p[2]))
    for p in Po_flux:
        Data=np.append(Data,a_int*(u(p[0],p[1],p[2])-T_i[n] ))

    n=0

    u_n=u
    u = TrialFunction(V)
    
    a = 1/dt*dot(C*u, v)*dx +dot(K*grad(u), grad(v))*dx+alpha_int*u*v*ds(0)+alpha_ext*u*v*ds(1)
    A=assemble(a)
    u = Function(V)
    t=0

    Ti=Constant(0.0)
    Te=Constant(0.0)

    L =  1/dt*dot(C*u_n, v)*dx+ alpha_int*Ti*v*ds(0)+ alpha_ext*Te*v*ds(1)+E_i*v*ds(0)+E_e*v*ds(1)
    if flag==0:
        for n in range(num_steps):
            t += dt
            Ti.assign(T_i[n+1])
            Te.assign(T_e[n+1])
            b=assemble(L)
            solver.solve(A, u.vector(), b)
            u_n.assign(u)

            for p in Po_temp:
                Data=np.append(Data,u(p[0],p[1],p[2]))
            for p in Po_flux:
                Data=np.append(Data,a_int*(u(p[0],p[1],p[2])-T_i[n] ))
    else:
        for n in range(num_steps):
            t += dt
            Ti.assign(T_i[n+1])
            Te.assign(T_e[n+1])
            b=assemble(L)
            solver.solve(A, u.vector(), b)
            u_n.assign(u)
            ub0.interpolate(u)
            NF_int=np.array(ub0.vector())
            ub1.interpolate(u)
            NF_ext=np.array(ub1.vector())
            Flux_int=np.zeros(num0,)
            Flux_ext=np.zeros(num1,)
            for jj in range(num0):
                Flux_int[jj]=a_int*(NF_int[jj]-T_i[n] )
            for jj in range(num1):
                Flux_ext[jj]=a_ext*(NF_ext[jj]-T_e[n] )
            ub0.vector()[:]=Flux_int
            ub1.vector()[:]=Flux_ext
            flux_int =assemble((ub0-Ei)*ds0)
            flux_ext =assemble((ub1-Ee)*ds1)

            Flux_Data.append(flux_int)
            Flux_Data.append(flux_ext)
            for p in Po_temp:
                Data=np.append(Data,u(p[0],p[1],p[2]))
            for p in Po_flux:
                Data=np.append(Data,a_int*(u(p[0],p[1],p[2])-T_i[n] ))

    return Data,Flux_Data


def Heat_predictions_oneD(ID,k_values, c_values, a_int,a_ext,num_steps):
    tol = 1e-14
    with h5py.File('Ensemble_'+ID+'.hdf5', 'r') as f:
        g=f['FwdModel']
        D=g.attrs['D']
        Area=g.attrs['Area']
        N=g.attrs['N']
        dt=g.attrs['dt']
        T_i=g.attrs['Ti']
        T_e=g.attrs['Te']
        layers=g.attrs['layers']
        Num_Ks=g.attrs['Num_Ks']

    
    mesh =IntervalMesh(N,0,D)

    class R1(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[0]) < 1E-14

    class R2(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[0]) > D-1E-14

    boundary_markers= MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    boundary_markers.set_all(9999)
    bR1 = R1()
    bR2 = R2()
    bR1.mark(boundary_markers, 0)
    bR2.mark(boundary_markers, 1)
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
 
 
    V = FunctionSpace(mesh, 'Lagrange', 1)
    u = TrialFunction(V)
    v = TestFunction(V)
   
    if Num_Ks==3:
        k=OneDThreeLayersThermalProperties(layers, k_values)
        c=OneDThreeLayersThermalProperties(layers, c_values)
    else:
        k=OneDFiveLayersThermalProperties(layers, k_values)
        c=OneDFiveLayersThermalProperties(layers, c_values)

    K =interpolate(k, V)
    C = interpolate(c, V)

    alpha_int =Constant(a_int)
    alpha_ext =Constant(a_ext)
    a = dot(K*grad(u), grad(v))*dx+alpha_int*u*v*ds(0)+alpha_ext*u*v*ds(1)
    A=assemble(a)
    u = Function(V)
    n=0
    Ti=Constant(T_i[0])
    Te=Constant(T_e[0])
    L = alpha_int*Ti*v*ds(0)+ alpha_ext*Te*v*ds(1)
    b=assemble(L)
    solver = KrylovSolver('gmres', 'ilu')
    prm = solver.parameters

    prm["absolute_tolerance"] = 1E-12
    prm["relative_tolerance"] = 1E-10
    prm["maximum_iterations"]  = 50000    #solve(A, u.vector(), b,'gmres','ilu')
    solver.solve(A, u.vector(), b)
    

    
    Flux_Data=[]
    Flux_int=a_int*(u(0)-T_i[n] )
    Flux_ext=a_ext*(u(D)-T_e[n] )
    flux_int =Flux_int*Area
    flux_ext =Flux_ext*Area

    
    Flux_Data.append(flux_int)
    Flux_Data.append(flux_ext)
 
    
    u_n=u
    
    u = TrialFunction(V)
    
    a = 1/dt*dot(C*u, v)*dx +dot(K*grad(u), grad(v))*dx+alpha_int*u*v*ds(0)+alpha_ext*u*v*ds(1)
    
    
    A=assemble(a)
    u = Function(V)
 
    L =  1/dt*dot(C*u_n, v)*dx+ alpha_int*Ti*v*ds(0)+ alpha_ext*Te*v*ds(1)

    
    t=0
    for n in range(num_steps):
        t += dt
        Ti.assign(T_i[n+1])
        Te.assign(T_e[n+1])
        b=assemble(L)
        solver.solve(A, u.vector(), b)
        u_n.assign(u)
        Flux_int=a_int*(u(0)-T_i[n] )
        Flux_ext=a_ext*(u(D)-T_e[n] )
        flux_int =Flux_int*Area
        flux_ext =Flux_ext*Area
        Flux_Data.append(flux_int)
        Flux_Data.append(flux_ext)
    return Flux_Data
    


def GetSyndata(ID,k_B,k_A,c_B,c_A, a_int,a_ext,sigma_TI, sigma_temp,sigma_flux ,flag,flag2):
    tol = 1e-14
    n0=Constant((0.0,0.0,-1.0))
    n1=Constant((0.0,0.0,1.0))
    with h5py.File('Ensemble_'+ID+'.hdf5', 'r') as f:
        g=f['FwdModel']
        D=g.attrs['D']
        layers=g.attrs['layers']
        if flag==0:
            num_steps=g.attrs['num_steps_inv']
        else:
            num_steps=g.attrs['num_steps_pred']
        RF_N=g.attrs['RF_N']
        Fenics_N=g.attrs['Fenics_N']
        D=g.attrs['D']
        dt=g.attrs['dt']
        Po_temp=g.attrs['Po_temp']
        Po_flux=g.attrs['Po_flux']
        T_i=g.attrs['Ti']
        T_e=g.attrs['Te']




    foldername='Truth'

    mesh = BoxMesh(Point(0,0,0), Point(D[0],D[1],D[2]),Fenics_N[0],Fenics_N[1],Fenics_N[2])
    ds, submesh0, submesh1=set_boundary(D,mesh)
    if flag==1:
        Vb0 = FunctionSpace(submesh0, 'CG', 1)
        num0 = Vb0.dim()
        ub0 = Function(Vb0)
        Vb1 = FunctionSpace(submesh1, 'CG', 1)
        num1 = Vb1.dim()
        ub1 = Function(Vb1)
        ds1 = Measure('dx', domain=submesh1)
        ds0 = Measure('dx', domain=submesh0)

    V = FunctionSpace(mesh, 'Lagrange', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    #K = kappa
    #V0 = FunctionSpace(mesh, 'DG', 0)

    if flag2==0:
        kappa = ThreeDThermalPropertiesThreeLayersTruth(layers,k_B,k_A,degree=1)
        c = ThreeDThermalPropertiesThreeLayersTruth(layers,c_B,c_A,degree=1)
    else:
        kappa = ThreeDThermalPropertiesThreeLayersPlain(layers,k_B,k_A,degree=1)
        c = ThreeDThermalPropertiesThreeLayersPlain(layers,c_B,c_A,degree=1)
        

    #K =interpolate(kappa, V0)
    #C = interpolate(c, V0)
    K =interpolate(kappa, V)
    C = interpolate(c, V)


    alpha_int =Constant(a_int)
    alpha_ext =Constant(a_ext)

    a = dot(K*grad(u), grad(v))*dx+alpha_int*u*v*ds(0)+alpha_ext*u*v*ds(1)
    A=assemble(a)
    u = Function(V)
    n=0
    Ti=Constant(T_i[n])
    Te=Constant(T_e[n])
    L = alpha_int*Ti*v*ds(0)+ alpha_ext*Te*v*ds(1)
    b=assemble(L)
    solver = KrylovSolver('gmres', 'ilu')
    prm = solver.parameters

    prm["absolute_tolerance"] = 1E-12
    prm["relative_tolerance"] = 1E-10
    prm["maximum_iterations"]  = 50000    #solve(A, u.vector(), b,'gmres','ilu')
    solver.solve(A, u.vector(), b)
    if flag==0:
        Flux_Data=0
    else:
        Flux_Data=[]

    if flag==1:
        ub0.interpolate(u)
        NF_int=np.array(ub0.vector())
        ub1.interpolate(u)
        NF_ext=np.array(ub1.vector())
        Flux_int=np.zeros(num0,)
        Flux_ext=np.zeros(num1,)
        for jj in range(num0):
            Flux_int[jj]=a_int*(NF_int[jj]-T_i[n] )
        for jj in range(num1):
            Flux_ext[jj]=a_ext*(NF_ext[jj]-T_e[n] )
        ub0.vector()[:]=Flux_int
        ub1.vector()[:]=Flux_ext
        Flux_Data=[]
        flux_int =assemble(ub0*ds0)
        flux_ext =assemble(ub1*ds1)
        Flux_Data.append(flux_int)
        Flux_Data.append(flux_ext)

    x1 = np.linspace(0, D[0], RF_N[0])
    y1 = np.linspace(0, D[1], RF_N[1])

    TI_data=np.zeros((RF_N[0],RF_N[1]))


    for mm1 in range(RF_N[0]):
        for mm2 in range(RF_N[1]):
            TI_data[mm1,mm2]=u(x1[mm1],y1[mm2],D[2])
    
    TI_data += sigma_TI*np.random.randn(RF_N[0],RF_N[1])

    if flag==0:
        Data = np.empty((3,0))
        Gamma = np.empty((3,0))
    else:
        Data = np.empty((3,0))
        Gamma = np.empty((3,0))

    for p in Po_temp:
        Data=np.append(Data,u(p[0],p[1],p[2]))
    Gamma=np.append(Gamma,sigma_temp*np.ones(len(Po_temp),))
    n=0
    for p in Po_flux:
        Fl=a_int*(u(p[0],p[1],p[2])-T_i[n] )
        Data=np.append(Data,Fl)
        Gamma=np.append(Gamma,np.maximum(sigma_flux*np.abs(Fl),0.5))


    u_n=u
    
    vtkfile = File(foldername+'/perm.pvd')
    vtkfile << K
    vtkfile = File(foldername+'/c.pvd')
    vtkfile << C
    
    u = TrialFunction(V)
    T=num_steps*dt
    a = 1/dt*dot(C*u, v)*dx +dot(K*grad(u), grad(v))*dx+alpha_int*u*v*ds(0)+alpha_ext*u*v*ds(1)

    A=assemble(a)
    vtkfile = File(foldername+'/solution.pvd')
    u = Function(V)
    t=0

    Ti=Constant(0.0)
    Te=Constant(0.0)
    L =  1/dt*dot(C*u_n, v)*dx+ alpha_int*Ti*v*ds(0)+ alpha_ext*Te*v*ds(1)
    if flag==0:
        for n in range(num_steps):
            t += dt
            Ti.assign(T_i[n+1])
            Te.assign(T_e[n+1])
            b=assemble(L)
            solver.solve(A, u.vector(), b)
            u_n.assign(u)

            for p in Po_temp:
                Data=np.append(Data,u(p[0],p[1],p[2]))
            Gamma=np.append(Gamma,sigma_temp*np.ones(len(Po_temp),))
            for p in Po_flux:
                Fl=a_int*(u(p[0],p[1],p[2])-T_i[n] )
                Data=np.append(Data,Fl)
#                print(sigma_flux*np.abs(Fl))
                Gamma=np.append(Gamma,np.maximum(sigma_flux*np.abs(Fl),0.5))

    else:
        for n in range(num_steps):
            t += dt
            print(n)
            Ti.assign(T_i[n+1])
            Te.assign(T_e[n+1])
            b=assemble(L)
            solver.solve(A, u.vector(), b)
            u_n.assign(u)
            ub0.interpolate(u)
            NF_int=np.array(ub0.vector())
            ub1.interpolate(u)
            NF_ext=np.array(ub1.vector())
            Flux_int=np.zeros(num0,)
            Flux_ext=np.zeros(num1,)
            for jj in range(num0):
                Flux_int[jj]=a_int*(NF_int[jj]-T_i[n] )
            for jj in range(num1):
                Flux_ext[jj]=a_ext*(NF_ext[jj]-T_e[n] )
            ub0.vector()[:]=Flux_int
            ub1.vector()[:]=Flux_ext
            flux_int =assemble(ub0*ds0)
            flux_ext =assemble(ub1*ds1)

            Flux_Data.append(flux_int)
            Flux_Data.append(flux_ext)
            for p in Po_temp:
                Data=np.append(Data,u(p[0],p[1],p[2]))
            Gamma=np.append(Gamma,sigma_temp*np.ones(len(Po_temp),))

            for p in Po_flux:
                Fl=a_int*(u(p[0],p[1],p[2])-T_i[n] )
                Data=np.append(Data,Fl)
                Gamma=np.append(Gamma,np.maximum(sigma_flux*np.abs(Fl),0.5))




    M=len(Gamma)
    noise =Gamma*np.random.randn(M,)
    Data =Data+noise.T

    return Data,TI_data,Flux_Data,Gamma




def GetPlain(ID,k_B,c_B, a_int,a_ext,sigma_TI, sigma_temp,sigma_flux ,flag):
    tol = 1e-14
    n0=Constant((0.0,0.0,-1.0))
    n1=Constant((0.0,0.0,1.0))
    with h5py.File('Ensemble_'+ID+'.hdf5', 'r') as f:
        g=f['FwdModel']
        D=g.attrs['D']
        layers=g.attrs['layers']
        if flag==0:
            num_steps=g.attrs['num_steps_inv']
        else:
            num_steps=g.attrs['num_steps_pred']
        RF_N=g.attrs['RF_N']
        Fenics_N=g.attrs['Fenics_N']
        D=g.attrs['D']
        dt=g.attrs['dt']
        Po_temp=g.attrs['Po_temp']
        Po_flux=g.attrs['Po_flux']
        T_i=g.attrs['Ti']
        T_e=g.attrs['Te']




    foldername='Truth'

    mesh = BoxMesh(Point(0,0,0), Point(D[0],D[1],D[2]),Fenics_N[0],Fenics_N[1],Fenics_N[2])
    ds, submesh0, submesh1=set_boundary(D,mesh)
    if flag==1:
        Vb0 = FunctionSpace(submesh0, 'CG', 1)
        num0 = Vb0.dim()
        ub0 = Function(Vb0)
        Vb1 = FunctionSpace(submesh1, 'CG', 1)
        num1 = Vb1.dim()
        ub1 = Function(Vb1)
        ds1 = Measure('dx', domain=submesh1)
        ds0 = Measure('dx', domain=submesh0)

    V = FunctionSpace(mesh, 'Lagrange', 1)
    u = TrialFunction(V)
    v = TestFunction(V)



    alpha_int =Constant(a_int)
    alpha_ext =Constant(a_ext)

    a = dot(k_B*grad(u), grad(v))*dx+alpha_int*u*v*ds(0)+alpha_ext*u*v*ds(1)
    A=assemble(a)
    u = Function(V)
    n=0
    Ti=Constant(T_i[n])
    Te=Constant(T_e[n])
    L = alpha_int*Ti*v*ds(0)+ alpha_ext*Te*v*ds(1)
    b=assemble(L)
    solver = KrylovSolver('gmres', 'ilu')
    prm = solver.parameters

    prm["absolute_tolerance"] = 1E-12
    prm["relative_tolerance"] = 1E-10
    prm["maximum_iterations"]  = 50000    #solve(A, u.vector(), b,'gmres','ilu')
    solver.solve(A, u.vector(), b)
    if flag==0:
        Flux_Data=0
    else:
        Flux_Data=[]

    if flag==1:
        ub0.interpolate(u)
        NF_int=np.array(ub0.vector())
        ub1.interpolate(u)
        NF_ext=np.array(ub1.vector())
        Flux_int=np.zeros(num0,)
        Flux_ext=np.zeros(num1,)
        for jj in range(num0):
            Flux_int[jj]=a_int*(NF_int[jj]-T_i[n] )
        for jj in range(num1):
            Flux_ext[jj]=a_ext*(NF_ext[jj]-T_e[n] )
        ub0.vector()[:]=Flux_int
        ub1.vector()[:]=Flux_ext
        Flux_Data=[]
        flux_int =assemble(ub0*ds0)
        flux_ext =assemble(ub1*ds1)
        Flux_Data.append(flux_int)
        Flux_Data.append(flux_ext)

    x1 = np.linspace(0, D[0], RF_N[0])
    y1 = np.linspace(0, D[1], RF_N[1])

    TI_data=np.zeros((RF_N[0],RF_N[1]))


    for mm1 in range(RF_N[0]):
        for mm2 in range(RF_N[1]):
            TI_data[mm1,mm2]=u(x1[mm1],y1[mm2],D[2])
    
    TI_data += sigma_TI*np.random.randn(RF_N[0],RF_N[1])

    if flag==0:
        Data = np.empty((3,0))
        Gamma = np.empty((3,0))
    else:
        Data = np.empty((3,0))
        Gamma = np.empty((3,0))

    for p in Po_temp:
        Data=np.append(Data,u(p[0],p[1],p[2]))
    Gamma=np.append(Gamma,sigma_temp*np.ones(len(Po_temp),))
    n=0
    for p in Po_flux:
        Fl=a_int*(u(p[0],p[1],p[2])-T_i[n] )
        Data=np.append(Data,Fl)
        Gamma=np.append(Gamma,np.maximum(sigma_flux*np.abs(Fl),0.5))


    u_n=u
    
#    vtkfile = File(foldername+'/perm.pvd')
#    vtkfile << K
#    vtkfile = File(foldername+'/c.pvd')
#    vtkfile << C
    
    u = TrialFunction(V)
    T=num_steps*dt
    a = 1/dt*dot(c_B*u, v)*dx +dot(k_B*grad(u), grad(v))*dx+alpha_int*u*v*ds(0)+alpha_ext*u*v*ds(1)

    A=assemble(a)
    vtkfile = File(foldername+'/solution.pvd')
    u = Function(V)
    t=0

    Ti=Constant(0.0)
    Te=Constant(0.0)
    L =  1/dt*dot(c_B*u_n, v)*dx+ alpha_int*Ti*v*ds(0)+ alpha_ext*Te*v*ds(1)
    if flag==0:
        for n in range(num_steps):
            t += dt
            Ti.assign(T_i[n+1])
            Te.assign(T_e[n+1])
            b=assemble(L)
            solver.solve(A, u.vector(), b)
            u_n.assign(u)

            for p in Po_temp:
                Data=np.append(Data,u(p[0],p[1],p[2]))
            Gamma=np.append(Gamma,sigma_temp*np.ones(len(Po_temp),))
            for p in Po_flux:
                Fl=a_int*(u(p[0],p[1],p[2])-T_i[n] )
                Data=np.append(Data,Fl)
#                print(sigma_flux*np.abs(Fl))
                Gamma=np.append(Gamma,np.maximum(sigma_flux*np.abs(Fl),0.5))

    else:
        for n in range(num_steps):
            t += dt
            print(n)
            Ti.assign(T_i[n+1])
            Te.assign(T_e[n+1])
            b=assemble(L)
            solver.solve(A, u.vector(), b)
            u_n.assign(u)
            ub0.interpolate(u)
            NF_int=np.array(ub0.vector())
            ub1.interpolate(u)
            NF_ext=np.array(ub1.vector())
            Flux_int=np.zeros(num0,)
            Flux_ext=np.zeros(num1,)
            for jj in range(num0):
                Flux_int[jj]=a_int*(NF_int[jj]-T_i[n] )
            for jj in range(num1):
                Flux_ext[jj]=a_ext*(NF_ext[jj]-T_e[n] )
            ub0.vector()[:]=Flux_int
            ub1.vector()[:]=Flux_ext
            flux_int =assemble(ub0*ds0)
            flux_ext =assemble(ub1*ds1)

            Flux_Data.append(flux_int)
            Flux_Data.append(flux_ext)
            for p in Po_temp:
                Data=np.append(Data,u(p[0],p[1],p[2]))
            Gamma=np.append(Gamma,sigma_temp*np.ones(len(Po_temp),))

            for p in Po_flux:
                Fl=a_int*(u(p[0],p[1],p[2])-T_i[n] )
                Data=np.append(Data,Fl)
                Gamma=np.append(Gamma,np.maximum(sigma_flux*np.abs(Fl),0.5))




    M=len(Gamma)
    noise =Gamma*np.random.randn(M,)
    Data =Data+noise.T

    return Data,TI_data,Flux_Data,Gamma
