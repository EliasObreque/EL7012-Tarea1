function y=ysim_d(X,a,b,g)

% y is the vector of outputs when evaluating the TS defined by a,b,g
% X is the data matrix

% Nd number of point we want to evaluate
% n is the number of regressors of the TS model

[Nd,n]=size(X);

% NR is the number of rules of the TS model
NR=size(a,1);         
y=zeros(Nd,1);

ruido1=0.5;
ruido2=1.3;
ruido3=0.7;
     
for k=1:Nd 
    
    % W(r) is the activation degree of the rule r
    % mu(r,i) is the activation degree of rule r, regressor i
    W=ones(1,NR);
    mu=zeros(NR,n);
    for r=1:NR
     for i=1:n
       mu(r,i)=exp(-0.5*(a(r,i)*(X(k,i)-b(r,i)))^2);  
       W(r)=W(r)*mu(r,i);
     end
    end

    % Wn(r) is the normalized activation degree
    Wn=W/sum(W);
    %keyboard
    noise= [ruido1*rand(1);ruido2*rand(1);ruido3*rand(1)];
    % Now we evaluate the consequences
    yr_s=g*[1 ;X(k,:)'];  
    yr=yr_s+noise; 
    
    
    % Finally the output
    y(k,1)=Wn*yr;

end
