function [y,x]=ysim_p_pasos(X,a,b,g,p)

[n,m]=size(X);
Xnew=[];

for j=1:p
    
   ynew=ysim(X,a,b,g);%prediccion a 1 paso
   Xnew=[ynew X(:,1:m-1)]; 
   X=Xnew;
end

y=ynew;
x=Xnew;

end