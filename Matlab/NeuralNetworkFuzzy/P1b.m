clc
clear all
close all

%------------Generar Datos del modelo------------
Nd = 6000;         %Número de datos

fmin = 0.2;       %frecuencia mínima
fmax = 1;         %frecuencia máxima
Ts   = 0.01;      %Tiempo de muestreo
a    = -1;        %[a b] Amplitud de la señal
b    = 1;
gain_aprbs = 4;

num_regresor = 2;

[aprbs, prbs] = createAPRBS(Nd, Ts, fmax, fmin, a, b, gain_aprbs);
Nd_ = size(aprbs, 1);
y_model = SignalConstruction(Nd_, num_regresor, aprbs);

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
ry = 2;
ru = 2;

[x_data, y_data] = createMatrixInput(Nd, ry, ru, y_model, aprbs);

% Porcentaje de cada set
train_prc = 0.5;
test_prc = 0.25;
val_prc = 0.25;

ntrain = Nd * train_prc;
ntest = Nd * test_prc;
nval = Nd * val_prc;

%% NEURALNETWORK
num_neu = 10;
[net_trained, tr] = NeuralNetwork(num_neu, x_data, y_data, train_prc, test_prc, val_prc);

%%
x_train = x_data.*tr.trainMask{1}';
y_train = y_data.*tr.trainMask{1}';
x_test = x_data.*tr.testMask{1}'; 
y_test = y_data.*tr.testMask{1}';
x_val = x_data.*tr.valMask{1}'; 
y_val = y_data.*tr.valMask{1}';

y_train_nn = net_trained(x_train');
y_test_nn = net_trained(x_test');
y_val_nn = net_trained(x_val');

% Error
error_train = y_train - y_train_nn';
error_test = y_test - y_test_nn';
error_val = y_val - y_val_nn';


%% Sensitivity analysis
I = SensitivityCalc(type_actfunc, num_regresor, x_test, net_properties)


%% PLOT

% plots error vs. epoch for the training, validation, and test performances of the training record TR 
figure()
plotperform(tr)
% Plot training state values
figure()
plottrainstate(tr)

% Plot Output and Target Values
%plotfit(net_trained, x_data, y_data)

% Plot Linear Regression
% plotregression(x_test, y_test_nn,'Regression')

% Plot Histogram of Error Values
figure()
ploterrhist(error_train,'Train', error_test, 'Test', error_val, 'Validation')

figure()
stairs(y_train_nn)
hold on
stairs(y_train)
xlim([1 100])
%[ exported_ann_structure ] = my_ann_exporter(net_trained);
%y_test_nn = my_ann_evaluation(exported_ann_structure, x_test)';

