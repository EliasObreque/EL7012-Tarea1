function y = SignalConstruction(Nd, num_regresor, control_signal)
%SIGNALCONSTRUCTION Summary of this function goes here
%------Construccion de la señal

u = control_signal;
e = zeros(Nd,1);
y = zeros(Nd,1);
r_b = wgn (Nd, 1, -10);                    %ruido blanco

%Serie de Chen
    for k = num_regresor + 1:Nd
       e(k)=0.5*exp(-(y(k-1)^2))*r_b(k);  %error
       y(k)=(0.8-0.5*exp(-(y(k-1)^2)))*y(k-1)-(0.3+0.9*exp(-(y(k-1)^2)))*y(k-2)+u(k-1)+0.2*u(k-2)+0.1*u(k-1)*u(k-2)+e(k);
    end
end

