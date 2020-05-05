%Genera una señal PRBS
N=100;
u=idinput(N);
figure(1)
plot(N)
%***************************************
Band=[0 0.2];
Range=[-4,4];
u1 = idinput(N,'prbs',Band,Range);
k=1:1:N;
figure(2)
plot(k,u1)
%stairs(u1)
%Genera una señal APRBS
u=u1.*rand(N,1);

figure(3)
stairs(k,u)
