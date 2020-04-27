function [e] = MAPE(y,yest)
%Mean Absolute Percentage Error
%Debe multiplicarse por 100 para expresarlo en porciento
N=length(y);

e=(1/N)*sum(abs((y-yest)./y))*100;
end

