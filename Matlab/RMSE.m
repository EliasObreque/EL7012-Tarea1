function [e] = RMSE(y,yest)
%Root-mean-square deviation
N=length(y);

e=(1/N)*sqrt(sum(y-yest).^2);

end

