function [e] = RMSE(y,yest)
%Root-mean-square deviation

e=sqrt(mean(y-yest).^2);

end

