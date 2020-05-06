function [aprbs, prbs] = createAPRBS(ndatos, Ts, fmax, fmin, minAmp, maxAmp, gain_aprbs)
%CREATEAPBRS Summary of this function goes here
%   Detailed explanation goes here

fc = 2.5*fmax;    %Frecuencia de cambio de bit
fs = 1/Ts;        %frecuencia de muestreo
Ns = fs/fc;       %Numero de muestras por bit
n  = ceil(log(fc/fmin+1)/log(2)); %orden de la señal
Pmax = 2^n-1;     %Período máximo

NumPeriod = ceil(ndatos/Pmax);
%Band = [0 B] B=1/Ns 

prbs =- idinput([Pmax 1 NumPeriod],'prbs',[0 1],[minAmp maxAmp]); %PBRS

%APBRS
n_d = round(rand(size(prbs)), 1);
aprbs = gain_aprbs * prbs.*n_d;
end

