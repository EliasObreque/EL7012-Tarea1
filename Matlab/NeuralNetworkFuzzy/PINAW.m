function [e] = PINAW(y,yu,yl)
%Prediction Interval Normalized Average Width (%)

N=length(y);
R=max(y)-min(y);

e=1/(N*R)*sum(yu-yl)*100;

end

