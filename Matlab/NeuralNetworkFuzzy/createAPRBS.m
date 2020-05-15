function [aprbs, prbs] = createAPRBS(ndatos, Ts, fmax, fmin, minAmp, maxAmp, gain_aprbs)
%CREATEAPBRS Summary of this function goes here
%   Detailed explanation goes here

fc = 2.5*fmax;    %Frecuencia de cambio de bit
fs = 1/Ts;        %frecuencia de muestreo
Ns = fs/fc;       %Numero de muestras por bit
n  = ceil(log(fc/fmin+1)/log(2)); %orden de la señal
Pmax = 2^n-1;     %Período máximo

%NumPeriod = ceil(ndatos/Pmax);
% prbs =- idinput([Pmax 1 NumPeriod],'prbs',[0 1],[minAmp maxAmp]); %PBRS

Band = [0 1/Ns]; %B=1/Ns
Nmax=Pmax*Ns;
NumPeriod = ceil(ndatos/(Nmax));
prbs =- idinput([Nmax 1 NumPeriod],'prbs',Band,[minAmp maxAmp]); %PBRS

%APBRS
% n_d = round(rand(size(prbs)), 1);
% aprbs = gain_aprbs * prbs.*n_d;
temp=size(prbs)/Pmax;
n_d = round(rand(temp(1)), 1);

for i=1:size(n_d)

    aprbs((i-1)*Pmax+1:i*Pmax) = gain_aprbs * prbs((i-1)*Pmax+1:i*Pmax)*n_d(i);
end
end

