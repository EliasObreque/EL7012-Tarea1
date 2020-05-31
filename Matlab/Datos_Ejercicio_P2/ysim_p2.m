function [y,x]=ysim_p2(X,a,b,g,p,RA)

% y is the vector of outputs when evaluating the TS defined by a,b,g
% X is the data matrix
% p paso a obtenr la respuesta

% Nd number of point we want to evaluate
% n is the number of regressors of the TS model

[Nd,n]=size(X);

% NR is the number of rules of the TS model
NR=size(a,1);         
y=zeros(Nd,1);
x=zeros(Nd,n);
      
% Xest=X;
% Xest(:,1:2)=0;
     
for k=1:Nd-p 
    
    % W(r) is the activation degree of the rule r
    % mu(r,i) is the activation degree of rule r, regressor i
    W=ones(1,NR);
    mu=zeros(NR,n);
    
    Xest(k,:)=X(k,:);
    
    for j=k:k+p
        Xtemp=Xest(j,:);
        Xtemp(find(RA==0))=[];
        [~,n1]=size(Xtemp);
    for r=1:NR 
     for i=1:n1 
%        mu(r,i)=exp(-0.5*(a(r,i)*(Xest(j,i)-b(r,i)))^2);
        mu(r,i)=exp(-0.5*(a(r,i)*(Xtemp(i)-b(r,i)))^2);
       W(r)=W(r)*mu(r,i);
     end
    end

    % Wn(r) is the normalized activation degree
    if sum(W)==0
        Wn=W;
    else
        Wn=W/sum(W);
    end
    
    % Now we evaluate the consequences
%     yr=g*[1 ;Xest(j,:)'];  
    yr=g*[1 ;Xtemp'];
    sal=abs(Wn*yr);
%    if sal<0
%        sal=0;
%    end
    
    % Finally the output
    if k==1
        y(j,1)=sal;
        x(j,:)=Xest(j,:);
    end
    
    % Corrimiento de regresores
    Xest(j+1,1)=sal;  
    for i=2:n
    Xest(j+1,i)=Xest(j,i-1);
    end
    
    end
    

    y(j,1)=Xest(j+1,1);
    x(j,:)=Xest(j,:);
    
end
x(:,find(RA==0))=[];
end
