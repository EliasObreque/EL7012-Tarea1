clc
clear all

% Ejemplo con Toolbox de un modelo TS
% Datos del Ejemplo:
% 

ndatos = 400;
train_prc = 0.5;
test_prc = 0.25;
val_prc = 0.25;

ntrain = ndatos * train_prc;
ntest = ndatos * test_prc;
nval = ndatos * val_prc;

max_x = 100;
x = linspace(0, max_x, ndatos);
noise = normrnd(0, sqrt(25), [1, ndatos]);
y = 1e-4 * sin(0.001 * x.^2) .* x.^3 + noise;

index_train = datasample(1:ndatos, ntrain);
index_test = datasample(1:ndatos, ntest);
index_val = datasample(1:ndatos, nval);

x_train = x(index_train(1:ntrain));
y_train = y(index_train(1:ntrain));
x_test = x(index_test(1:ntest));
y_test = y(index_test(1:ntest));
x_val = x(index_val(1:nval));
y_val = y(index_val(1:nval));

figure()
hold on
title('Origina - Nonlinear function')
plot(x, y, '.k')
plot(x_train, y_train, '.r')
plot(x_test, y_test, '.b')
plot(x_val, y_val, '.g')
grid on


%% Dependiendo de modelo se tendra y_test, y_train, x_test, x_train 
% con su largo correspondiente dado por: ntest, ntrain
%
max_clusters = 6;
[errtest,errent] = clusters_optimo(y_test,y_train,x_test,x_train, max_clusters);


%% Calcular el error con el numero de cluster (reglas)
% err=errortest(yent,Xent,ytest,Xtest,reglas)


%% Analisis de sensibilidad
% [p indice]=sensibilidad(yent,Xent,reglas)

%% Obtener modelo
% [model, result]=takagisugeno1(iden_y,iden_x,reglas,opcion)

%% Simulacion
% y=ysim(X,a,b,g)



