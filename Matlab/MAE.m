function [e] = MAE(y,yest)
%Mean Absolute Error
N=length(y);
e=(1/N)*sum(abs(y-yest));
end

