clear all
close all


Nt=501;
T=60*Nt/60/60;


load ../Results/Visual_3D.mat

f=figure(1);
f.Position = [10 10 600 400];
hold on
Pred=Fluxes(1:2:end,:);
tt=linspace(0,T,Nt);
low=prctile(Pred',2.5);
high=prctile(Pred',97.5);
X=[tt,fliplr(tt)];
Y=[low,fliplr(high)];

h1=fill(X,Y,'g');%,[0.8 0.8 0.8]);
h2=plot(tt,mean(Pred,2),'--r','linewidth',1.5);


drawnow
hold on
F=Fluxes(2:2:end,:);
tt=linspace(0,T,Nt);
low=prctile(F',2.5);
high=prctile(F',97.5);
X=[tt,fliplr(tt)];
Y=[low,fliplr(high)];


h3=fill(X,Y,[0.8 0.8 0.8]);
h4=plot(tt,mean(F,2),'-b','linewidth',1.0);
xlim([0,tt(end)])
drawnow
box on

legend([h1,h2,h3,h4],'CI (internal)','$$\overline{\textbf{H}}_{I}$$','CI (external)',...'
    '$$\overline{\textbf{H}}_{E}$$','interpreter','latex','fontsize',20,'location','west')
xlabel('Time (hrs)','FontSize',20,'Interpreter','latex')
ylabel('Heat flow rate (W)','FontSize',20,'Interpreter','latex')
filename='../Visualisation/Figures/wallEq_equiv';
saveas(gcf, filename,'epsc');

N=length(Fluxes(:,1));
d=mean(Fluxes,2);
ga=zeros(N,1);
ga(1:2:N)=(0.01*max(abs(d(1:2:end)))).^2;
ga(2:2:N)=(0.01*max(abs(d(2:2:end)))).^2;


data=d;
gamma=ga;
save('FlowRates_real_all','data','gamma')
clear data
clear gamma

temp_d1=d(1:2:end);
temp_d2=d(2:2:end);
temp_ga1=ga(1:2:end);
temp_ga2=ga(2:2:end);

N=301;
data(1:2:N*2)=temp_d1(1:N);
data(2:2:N*2)=temp_d2(1:N);
gamma(1:2:N*2)=temp_ga1(1:N);
gamma(2:2:N*2)=temp_ga2(1:N);
save('FlowRates_real','data','gamma')

