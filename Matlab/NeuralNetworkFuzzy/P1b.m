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

num_regresor_y = 2;
num_regresor_u = 2;
num_regresor = num_regresor_y  + num_regresor_u;

[aprbs, prbs] = createAPRBS(Nd, Ts, fmax, fmin, a, b, gain_aprbs);
Nd_ = size(aprbs, 1);
y_model = SignalConstruction(Nd_, num_regresor_y, aprbs);

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

[x_data, y_data] = createMatrixInput(Nd, ry, ru, y_model, aprbs);

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

figure('Name', 'Indicator by number of neurons')

for num_neu = num_neu_min: 1: num_neu_max
    %% NEURALNETWORK
    [net_trained, tr] = NeuralNetwork(num_neu, x_data, y_data, train_prc, test_prc, val_prc);

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

    subplot(5, 4, num_neu - num_neu_min + 1)
    hold on
    grid on
    xlabel('Regresors')
    bar(I)
    ylabel(join(['I - ', num2str(num_neu), ' nn']))


    %% RMSE
    rmse_train(num_neu - num_neu_min + 1) = RMSE(y_train, y_train_nn');
    rmse_test(num_neu - num_neu_min + 1)  = RMSE(y_test, y_test_nn');
    rmse_val(num_neu - num_neu_min + 1)   = RMSE(y_val, y_val_nn');
end

%%
figure()
hold on
grid on
title('RMSE vs Number of neurons')
ylabel('RMSE')
xlabel('Number of neurons')
[n, m] = min(rmse_train);
plot(m, n, '*b', 'DisplayName', 'min-Train')
[n, m] = min(rmse_test);
plot(m, n, '*r', 'DisplayName', 'min-Test')
[n, m] = min(rmse_val);
plot(m, n, '*k', 'DisplayName', 'min-Validation')
plot(num_neu_min:1:num_neu_max, rmse_train, 'b','DisplayName','Train')
plot(num_neu_min:1:num_neu_max, rmse_test, 'r','DisplayName','Test')
plot(num_neu_min:1:num_neu_max, rmse_val, 'k','DisplayName','Validation')
legend()

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

