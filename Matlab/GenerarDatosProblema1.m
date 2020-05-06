clear all
clc
%------------Generar BPRS------------
Nd=6010;         %N�mero de datos

fmin=0.2;       %frecuencia m�nima
fmax=1;         %frecuencia m�xima
Ts=0.01;        %Tiempo de muestreo
minAmp=-1;           %[a b] Amplitud de la se�al
maxAmp=1;
gain_aprbs=1;

[u, prbs] = createAPRBS(Nd, Ts, fmax, fmin, minAmp, maxAmp, gain_aprbs);

figure ()
stairs(u);
xlim([1 90])
xlabel('N�mero de muestras')
ylabel('Amplitud')
title('Se�al APRBS')

%------Construccion de la se�al
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
stairs(y)
hold on
stairs(u)
% stairs(e)
xlim([1 90])
xlabel('N�mero de muestras')
ylabel('Amplitud')
legend('y(k)', 'u(k)')
title('Serie no lineal din�mica')

%------------Contrucci�n del vector de datos
Dt=6000;         %Tama�o del vector
ry=5;
ru=5;
[X, Y] = createMatrixInput(Dt, ry, ru, y, u);


%--------------Selecci�n de datos Aleatoria----------------
%55% para entrenamiento (3300) 25% test (1500) y 20% validaci�n (1200)
rndIDX = randperm(Dt);

Xent = X(rndIDX(1:3300), :);
Yent = Y(rndIDX(1:3300), :);

Xtest = X(rndIDX(3301:4800), :);
Ytest = Y(rndIDX(3301:4800), :);

Xval = X(rndIDX(4801:6000), :);
Yval = Y(rndIDX(4801:6000), :);

 savefile = 'DatosProblema1.mat';
 save(savefile, 'X', 'Xent', 'Xtest','Xval','Y', 'Yent', 'Ytest', 'Yval','y','u','e');