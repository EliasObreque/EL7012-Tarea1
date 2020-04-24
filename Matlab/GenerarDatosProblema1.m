clear all
clc

%------------Generar BPRS------------
Nd=600;         %Número de datos

fmin=0.2;       %frecuencia mínima
fmax=1;         %frecuencia máxima
Ts=0.01;        %Tiempo de muestreo
a=-1;           %[a b] Amplitud de la señal
b=1;


fc=2.5*fmax;    %Frecuencia de cambio de bit
fs=1/Ts;        %frecuencia de muestreo
Ns=fs/fc;       %Numero de muestras por bit
n=ceil(log(fc/fmin+1)/log(2)); %orden de la señal
Pmax=2^n-1;     %Período máximo

%Band = [0 B] B=1/Ns 
%u =- idinput([Pmax*Ns],'prbs',[0 1/Ns],[a b]); %PBRS
 u =[0; 0; - idinput([Pmax 1 40],'prbs',[0 1],[a b])]; %PBRS

r_y=2;          %Cantidad de regresores en y

e=zeros(Nd+r_y,1);
y=zeros(Nd+r_y,1);
r_b=wgn (Nd+r_y, 1, -100);                    %ruido blanco

for k=r_y+1:Nd
   e(k)=0.5*exp(-(y(k-1)^2))*r_b(k);  %error
   y(k)=(0.8-0.5*exp(-(y(k-1)^2)))*y(k-1)-(0.3+0.9*exp(-(y(k-1)^2)))*y(k-2)+u(k-1)+0.2*u(k-2)+0.1*u(k-1)*u(k-2)+e(k);
end