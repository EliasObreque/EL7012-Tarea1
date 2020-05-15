function [g_u,g_l]=MinMax(X,Y,a,b,g)
% X Datos de entrada
% Y Salida del conjunto

% n is the number of regressors of the TS model
[Nd,n]=size(X); % Nde numero de puntos del conjunto de entrenamiento
NR=size(a,1);       % NR is the number of rules of the TS model      


for k=1:Nd  %Numeros del conjunto de
    
    % W(r) is the activation degree of the rule r
    % mu(r,i) is the activation degree of rule r, regressor i
    W=ones(1,NR);
    mu=zeros(NR,n);
  
    for j=1:NR %Numero de reglas
     for i=1:n %regresores
       mu(j,i)=exp(-0.5*(a(j,i)*(X(k,i)-b(j,i)))^2);  
       W(j)=W(j)*mu(j,i); %Opeweracion producto W
     end
    end

    % Wn(r) is the normalized activation degree
    Wn(:,k)=W/sum(W); 

end


A = []; % No constraints
b = [];
Aeq = [];
beq = [];
lb = [];
ub = [];
nonlcon = [];
options = optimoptions('fminimax','AbsoluteMaxObjectiveCount',1,'MaxIterations',10);

f = @(x)myfun(x,Wn,Y,X,Nd);
g_u= fminimax(f,g,A,b,Aeq,beq,lb,ub,nonlcon,options);
f1 = @(x)-myfun(x,Wn,Y,X,Nd);
g_l = fminimax(f1,g,A,b,Aeq,beq,lb,ub,nonlcon,options);

end


function [sal] = myfun(x,Wn,Y,X,Nd)

for k=1:Nd
sal(k)=Y(k)-sum(sum(Wn(:,k)*[1 X(k,:)].*x));
end

end