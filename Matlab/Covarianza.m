function [y_s,yu_s,yl_s]=Covarianza(Xent,Yent,Xval,a,b,g,alpha)

% y is the vector of outputs when evaluating the TS defined by a,b,g
% X is the data matrix

% n is the number of regressors of the TS model

[Nde,n]=size(Xent); % Nde numero de puntos del conjunto de entrenamiento
[Ndv,~]=size(Xval); % NdV numero de puntos del conjunto de validacion
NR=size(a,1);       % NR is the number of rules of the TS model      

% y_t=zeros(Ndv,1);
% yu-t=zeros(Ndv,1);
% yl=zeros(Ndv,1);


for k=1:Nde  %Numeros del conjunto de entrenamiento
    
    % W(r) is the activation degree of the rule r
    % mu(r,i) is the activation degree of rule r, regressor i
    W=ones(1,NR);
    mu=zeros(NR,n);
  
    for j=1:NR %Numero de reglas
     for i=1:n %regresores
       mu(j,i)=exp(-0.5*(a(j,i)*(Xent(k,i)-b(j,i)))^2);  
       W(j)=W(j)*mu(j,i); %Opeweracion producto W
     end
       y1(j)=g(j,:)*[1; Xent(k,:)'];
    end

    % Wn(r) is the normalized activation degree
    Wn(:,k)=W/sum(W); 
    psi(:,:,k)=Wn(:,k)*[1 Xent(k,:)];
    
    
    % Now we evaluate the consequences
%     yr=g*[1 ;Xent(k,:)'];  
    
    % Finally the output
    y(:,k)=Wn(:,k)*Yent(k); %Yent(k,1)' %.*y1'
    for j=1:NR %Numero de reglas
    e(j,k)=y(j,k)-psi(j,:,k)*g(j,:)';
    end

end

for k=1:Ndv  %Numeros del conjunto de validación
    
    % W(r) is the activation degree of the rule r
    % mu(r,i) is the activation degree of rule r, regressor i
    W=ones(1,NR);
    mu=zeros(NR,n);
  
    for j=1:NR %Numero de reglas
     for i=1:n %regresores
       mu(j,i)=exp(-0.5*(a(j,i)*(Xval(k,i)-b(j,i)))^2);  
       W(j)=W(j)*mu(j,i); %Opeweracion producto W
     end
       y1(j)=g(j,:)*[1; Xval(k,:)'];
    end

    % Wn(r) is the normalized activation degree
    if sum(W)==0
        WnV(:,k)=W; 
    else
       WnV(:,k)=W/sum(W); 
    end
    
    
    
    psiV(:,:,k)=WnV(:,k)*[1 Xval(k,:)];
    
    
%     % Now we evaluate the consequences
% %     yr=g*[1 ;Xent(k,:)'];  
%     
%     % Finally the output
    yV(:,k)=WnV(:,k).*y1'; %Yent(k,1)'
%     for j=1:NR %Numero de reglas
%     e(j,k)=y(j,k)-psiV(j,:,k)*g(j,:)';
%     if e(j,k)~=0
%         e(j,k)
%     end
%     end

end

for j=1:NR %Numero de reglas
%   varE(j)=std(e(j,:)*e(j,:)');
  varE(j)=std(e(j,:));
  
  temp(:,:)=psi(j,:,:);
  
  for k=1:Ndv  %Numeros del conjunto de validación
      
       
      tempV(:,:)=psiV(j,:,k)';
      
      varS(j,k)=varE(j)^2*(1+tempV'*inv(temp*temp')*tempV);
      
      yu(j,k)=[1 Xval(k,:)]*g(j,:)'+alpha*sqrt(varS(j));
      yl(j,k)=[1 Xval(k,:)]*g(j,:)'-alpha*sqrt(varS(j));
  end
end

%Salida Modelo
y_s=sum(yV);

for k=1:Ndv  %Numeros del conjunto de validación
yu_s(k)=WnV(:,k)'*yu(:,k);
yl_s(k)=WnV(:,k)'*yl(:,k);
end
end
