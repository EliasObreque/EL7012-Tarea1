clc
clear all

% Ejemplo con Toolbox de un modelo TS
% Datos del Ejemplo:
% 

ndatos = 5000;
train_prc = 0.5;
test_prc = 0.25;
val_prc = 0.25;

ntrain = ndatos * train_prc;
ntest = ndatos * test_prc;
nval = ndatos * val_prc;

%% Dependiendo de modelo se tendra y_test, y_train, x_test, x_train 
% con su largo correspondiente dado por: ntest, ntrain
% Luego 

% [errtest,errent] = clusters_optimo(ytest,yent,Xtest,Xent,max_clusters);

%% Calcular el error con el numero de cluster (reglas)
% err=errortest(yent,Xent,ytest,Xtest,reglas)


%% Analisis de sensibilidad
% [p indice]=sensibilidad(yent,Xent,reglas)

%% Obtener modelo
% [model, result]=takagisugeno1(iden_y,iden_x,reglas,opcion)

%% Simulacion
% y=ysim(X,a,b,g)



