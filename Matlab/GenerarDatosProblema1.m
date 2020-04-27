clear all
clc
%------------Generar BPRS------------
Nd=610;         %Número de datos

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
u =- idinput([Pmax 1 41],'prbs',[0 1],[a b]); %PBRS

%APBRS
n_d=round(4*rand(size(u)));
u=u.*n_d;

r_y=2;          %Cantidad de regresores en y

e=zeros(Nd,1);
y=zeros(Nd,1);
r_b=wgn (Nd, 1, -100);                    %ruido blanco

%Serie de Chen
for k=r_y+1:Nd
   e(k)=0.5*exp(-(y(k-1)^2))*r_b(k);  %error
   y(k)=(0.8-0.5*exp(-(y(k-1)^2)))*y(k-1)-(0.3+0.9*exp(-(y(k-1)^2)))*y(k-2)+u(k-1)+0.2*u(k-2)+0.1*u(k-1)*u(k-2)+e(k);
end

%------------Contrucción del vector de datos
Dt=600;         %Tamaño del vector
f=length(y);
ry=5;
ru=5;
Y=zeros(Dt,1);
X=zeros(Dt,ry+ru);
for i=f:-1:f-Dt+1
    Y(Dt,1)=y(i);
    %Regresores de y
    for j=1:ry
        X(Dt,j)=y(i-j);
    end
    %Regresores de u
    for j=1:ru
        X(Dt,j+ry)=u(i-j);
    end
    Dt=Dt-1;
end

%Selección de datos Aleatoria
%55% para entrenamiento (330) 25% test (150) y 20% validación (120)
rndIDX = randperm(600);

Xent = X(rndIDX(1:330), :);
Yent = Y(rndIDX(1:330), :);

Xtest = X(rndIDX(331:480), :);
Ytest = Y(rndIDX(331:480), :);

Xval = X(rndIDX(481:600), :);
Yval = Y(rndIDX(481:600), :);

 savefile = 'DatosProblema1.mat';
 save(savefile, 'X', 'Xent', 'Xtest','Xval','Y', 'Yent', 'Ytest', 'Yval');