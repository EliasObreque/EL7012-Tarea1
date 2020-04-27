% %Generar señal PBRS de [0,1] de período máximo
% a_k=[1 0 0 1]'; %coeficientes para n=4
% s=ones(Pmax,1);
% for k=2:Pmax
%     if (k<=n)
%        s(k)=mod(sum(s(1:k-1).*a_k(1:k-1)),2);
%     else
%        s(k)=mod(sum(s(k-n:k-1).*a_k),2);
%     end
% end
% 
% u=zeros(Ns*Pmax,1);
% for k=1:Pmax
%     u((k-1)*Ns+1:k*Ns)=s(k);
% end
% 
% %Generar PBRS de [a,b]
% %u(k)=(b-a)*u(k)+a
% u=(b-a)*u+a;
%-------------
sum
