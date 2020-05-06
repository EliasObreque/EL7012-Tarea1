clc
clear all
close all

ndatos = 1000;
train_prc = 0.5;
test_prc = 0.25;
val_prc = 0.25;

ntrain = ndatos * train_prc;
ntest = ndatos * test_prc;
nval = ndatos * val_prc;

max_x = 200;
x_data1 = linspace(0, max_x, ndatos);
x_data2 = linspace(-50, 50, ndatos);
noise = normrnd(0, sqrt(25), [1, ndatos]);
y_data = 1e-4 * sin(0.001 * x_data1.^2) .* x_data2.^3 + noise;

index_train = datasample(1:ndatos, ntrain);
index_test = datasample(1:ndatos, ntest);
index_val = datasample(1:ndatos, nval);
% 
% x_train = x(index_train(1:ntrain));
% y_train = y(index_train(1:ntrain));
% x_test = x(index_test(1:ntest));
% y_test = y(index_test(1:ntest));
% x_val = x(index_val(1:nval));
% y_val = y(index_val(1:nval));


%% NEURALNETWORK
num_neu = 10;
[net_trained, tr] = NeuralNetwork(num_neu, x_data, y_data, train_prc, test_prc, val_prc);

x_train = x_data.*tr.trainMask{1};
y_train = y_data.*tr.trainMask{1};
x_test = x_data.*tr.testMask{1}; 
y_test = y_data.*tr.testMask{1};
x_val = x_data.*tr.valMask{1}; 
y_val = y_data.*tr.valMask{1};

y_train_nn = net_trained(x_train);
y_test_nn = net_trained(x_test);
y_val_nn = net_trained(x_val);

% Error
error_train = y_train - y_train_nn;
error_test = y_test - y_test_nn;
error_val = y_val - y_val_nn;


%% Sensitivity analysis



%% PLOT
figure()
hold on
title('Nonlinear function')
plot(x_data, 1e-4 * sin(0.001 * x_data.^2) .* x_data.^3 , 'k')
plot(x_train, y_train, '.r')
plot(x_test, y_test, '.b')
plot(x_val, y_val, '.g')
grid on

% plots error vs. epoch for the training, validation, and test performances of the training record TR 
plotperform(tr)
% Plot training state values
plottrainstate(tr)
% Plot Output and Target Values
plotfit(net_trained, x_data, y_data)
% Plot Linear Regression
plotregression(x_test, y_test_nn,'Regression')
% Plot Histogram of Error Values
figure()
ploterrhist(error_train,'Train', error_test, 'Test', error_val, 'Validation')



%[ exported_ann_structure ] = my_ann_exporter(net_trained);
%y_test_nn = my_ann_evaluation(exported_ann_structure, x_test)';

