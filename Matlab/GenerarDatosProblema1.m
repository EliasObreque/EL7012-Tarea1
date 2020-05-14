clear all
clc
%------------Generar BPRS------------
Nd=6010;         %Número de datos

fmin=0.2;       %frecuencia mínima
fmax=1;         %frecuencia máxima
Ts=0.01;        %Tiempo de muestreo
minAmp=-1;           %[a b] Amplitud de la señal
maxAmp=1;
gain_aprbs=1;

[u, prbs] = createAPRBS(Nd, Ts, fmax, fmin, minAmp, maxAmp, gain_aprbs);

figure ()
stairs(u);
% xlim([1 90])
xlim([1 3000])
xlabel('Número de muestras')
ylabel('Amplitud')
title('Señal APRBS')

%------Construccion de la señal
r_y=2;          %Cantidad de regresores en y
e=zeros(Nd,1);
y=zeros(Nd,1);
r_b=wgn (Nd, 1, -10);                    %ruido blanco

%Serie de Chen
for k=r_y+1:Nd
   e(k)=0.5*exp(-(y(k-1)^2))*r_b(k);  %error
   y(k)=(0.8-0.5*exp(-(y(k-1)^2)))*y(k-1)-(0.3+0.9*exp(-(y(k-1)^2)))*y(k-2)+u(k-1)+0.2*u(k-2)+0.1*u(k-1)*u(k-2)+e(k);
end

figure ()
stairs(u)
hold on
stairs(y)
% stairs(e)
% xlim([1 90])
xlim([1 300])
xlabel('Número de muestras')
ylabel('Amplitud')
legend('u(k)','y(k)')
title('Serie no lineal dinámica')

%------------Contrucción del vector de datos
Dt=6000;         %Tamaño del vector
ry=4;
ru=4;
[X, Y] = createMatrixInput(Dt, ry, ru, y, u);


%--------------Selección de datos Aleatoria----------------
%55% para entrenamiento (3300) 25% test (1500) y 20% validación (1200)
% rndIDX = randperm(Dt);

Xent = X(1:3300, :);
Yent = Y(1:3300, :);

Xtest = X(3301:4800, :);
Ytest = Y(3301:4800, :);

Xval = X(4801:6000, :);
Yval = Y(4801:6000, :);

 savefile = 'P_DatosProblema1a.mat';
 save(savefile, 'X', 'Xent', 'Xtest','Xval','Y', 'Yent', 'Ytest', 'Yval','y','u','e');