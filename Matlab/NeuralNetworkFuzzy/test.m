clear all
clc
%------------Generar BPRS------------
Nd=6010;         %N�mero de datos

fmin=0.0;       %frecuencia m�nima
fmax=1/4;         %frecuencia m�xima
Ts = 0.01;        %Tiempo de muestreo
minAmp=-2;           %[a b] Amplitud de la se�al
maxAmp=2;
gain_aprbs=1;

[u, prbs] = createAPRBS(Nd, Ts, fmax, fmin, minAmp, maxAmp, gain_aprbs);
