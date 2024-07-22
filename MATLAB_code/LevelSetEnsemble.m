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
xi=randn(N,1);
nu=2.5;
L=0.09/sqrt(2*nu);
sigma=0.62;
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
    
load Image_save_830.mat



meas=(final(:)-mean(final(:)))./(std(final(:)));
sigma_n=0.1140;
M=length(meas);
y=(C_prior+sigma_n^2*eye(M))\meas;
pred=C_prior*y;
B=(C_prior+sigma_n^2*eye(M))\C_prior;
C_pos=C_prior-C_prior*B;

T = cholcov(C_pos);
numSamples = 1000;
ynew = pred + T'*randn(M,numSamples);
Lev=zeros(N,numSamples);
for i=1:numSamples
    temp=reshape(ynew(:,i),Nx,Nx)';
    Lev(:,i)=temp(:);
end

save('../PriorEnsembles/Lev_en','Lev')
