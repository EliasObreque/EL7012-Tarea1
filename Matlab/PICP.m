function [e] = PICP(y,yu,yl)
%Prediction Interval Normalized Average Width (%)

N=length(y);

c=0;
for i=1:N
    if y(i)>yl(i) && y(i)<yu(i)
        c=c+1;
    end
end

e=1/(N)*c*100;

end

