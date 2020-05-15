clc
clear all
close all

%------------Generar Datos del modelo------------
Nd = 6000;         %Número de datos

fmin = 0.2;        %frecuencia mínima
fmax = 1;         %frecuencia máxima
Ts   = 0.01;      %Tiempo de muestreo
a    = -1;        %[a b] Amplitud de la señal
b    = 1;
gain_aprbs = 4;

num_regresor_y = 2;
num_regresor_u = 1;
num_regresor = num_regresor_y  + num_regresor_u;

loaded_data = load('P_DatosProblema1.mat');

y_model = loaded_data.y;
aprbs = loaded_data.u;

% Contrucción del vector de datos con Nd datos ajustable
% X: y(k-1), ..., y(k-ry), u(k - 1), ..., u(k - ru)
ry = num_regresor_y;
ru = num_regresor_u;

x_train = loaded_data.Xent;
y_train = loaded_data.Yent;
x_test = loaded_data.Xtest;
y_test = loaded_data.Ytest;
x_val = loaded_data.Xval;
y_val = loaded_data.Yval;

%Elimina el ultima regresor
x_train(:, 4) = [];
x_test(:, 4) = [];
x_val(:, 4) = [];

x_data = [x_train; x_test; x_val];
y_data = [y_train; y_test; y_val];

% Numero optimo de neuronas calculadas en Identification.m
NUM_OPT_NEU = 10;


%% NEURALNETWORK
[net_trained, tr] = NeuralNetwork(NUM_OPT_NEU, x_train, y_train, x_test, y_test, x_val, y_val);
y_train_nn = net_trained(x_train');
y_test_nn = net_trained(x_test');
y_val_nn = net_trained(x_val');
%% Interval by covariance
jpasos = 16;
[y_j_train, y_j_test, y_j_val] = predictiveNN(jpasos, net_trained, x_train, x_test, x_val);

[y_u, y_l] = Covariance(net_trained, x_train, y_train, x_test, y_j_test);

plot_Intervalos(y_test_nn,y_u,y_l)
%%
figure()
stairs(y_test_nn)
grid on
hold on
stairs(y_u)
stairs(y_l)
xlim([1 300])



