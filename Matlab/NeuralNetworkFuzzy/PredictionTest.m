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
num_regresor_u = 2;
num_regresor = num_regresor_y  + num_regresor_u;

loaded_data = load('DatosProblema1a.mat');

y_model = loaded_data.y;
aprbs = loaded_data.u;

figure ()
stairs(y_model)
hold on
stairs(aprbs)
% stairs(e)
xlim([1 90])
xlabel('Número de muestras')
ylabel('Amplitud')
legend('y(k)', 'u(k)')
title('Serie no lineal dinámica')

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

% Porcentaje de cada set
train_prc = 0.5;
test_prc = 0.25;
val_prc = 0.25;

ntrain = Nd * train_prc;
ntest = Nd * test_prc;
nval = Nd * val_prc;


num_neu_min = 2;
num_neu_max = 21;

rmse_train = zeros(1, num_neu_max - num_neu_min);
rmse_test  = zeros(1, num_neu_max - num_neu_min);
rmse_val   = zeros(1, num_neu_max - num_neu_min);

% Numero optimo de neuronas calculadas en Identification.m
NUM_OPT_NEU = 6;


%% NEURALNETWORK
[net_trained, tr] = NeuralNetwork(NUM_OPT_NEU, x_train, y_train, x_test, y_test, x_val, y_val);%, test_prc, val_prc);

x_train = x_data.*tr.trainMask{1}';
y_train = y_data.*tr.trainMask{1}';
x_test = x_data.*tr.testMask{1}'; 
y_test = y_data.*tr.testMask{1}';
x_val = x_data.*tr.valMask{1}'; 
y_val = y_data.*tr.valMask{1}';

y_train_nn = net_trained(x_train');
y_test_nn = net_trained(x_test');
y_val_nn = net_trained(x_val');

error_train = y_train - y_train_nn';
error_test  = y_test - y_test_nn';
error_val   = y_val- y_val_nn';

%% Sensitivity analysis
I = SensitivityCalc('tanh', num_regresor, x_test, net_trained);

figure('Name', 'Indicator by number of neurons')
hold on
grid on
xlabel('Regresors')
bar(I)
ylabel(join(['I - ', num2str(NUM_OPT_NEU), ' nn']))

