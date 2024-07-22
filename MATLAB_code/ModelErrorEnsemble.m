clear all;
close all;

Nx=88;
D=1;
hx=D/Nx; 
[X,Y] = meshgrid(hx/2:hx:hx*Nx-hx/2);
N=Nx^2;
y=reshape(X,N,1);
x=reshape(Y,N,1);


rng('default')
rng(1239)
xi=randn(N,1);
nu=3.0;
L=0.05;
sigma=1.0;
C_prior=zeros(N,N);
for i=1:N
    %i
    v=[(x(i)-x(1:N))';(y(i)-y(1:N))'];
    h=sqrt(v(1,:).^2+v(2,:).^2);
    C_prior(i,1:N)=  sigma^2*2^(1-nu)/gamma(nu)*(h(1:N)/L).^(nu).*besselk(nu,h(1:N)/L);
end
for i=1:N
    C_prior(i,i)=  sigma^2;
end
    


T = cholcov(C_prior);
numSamples = 1000;
M=N;


Error_int =  T'*randn(M,numSamples);
Error_ext = T'*randn(M,numSamples);
save('../PriorEnsembles/ModErr_en','Error_int','Error_ext')
