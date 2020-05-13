function [net_trained, tr] = NeuralNetwork(num_neu, x_train, y_train,...
    x_test, y_test, x_val, y_val)


x_data = [x_train; x_test; x_val];
y_data = [y_train; y_test; y_val];

% Crea una capa oculta en la red
net_fit = fitnet(num_neu,'trainlm');

% Funcion de activacion 
net_fit.layers{1}.transferFcn = 'tansig'; % tansig = tanh
net_fit.divideFcn = 'divideind';
net_fit.divideParam.trainInd = 1:size(x_train, 1); 
net_fit.divideParam.testInd = size(x_train, 1) + 1: size(x_train, 1) + size(x_test, 1);
net_fit.divideParam.valInd = size(x_train, 1) + size(x_test, 1) + 1: size(x_data, 1);   
net_fit.trainParam.max_fail = 50;

%TRAINING PARAMETERS
net_fit.trainParam.show = 200;     %# of ephocs in display
net_fit.trainParam.lr = 0.05;      %learning rate
net_fit.trainParam.epochs = 5000;  %max epochs
%net_fit.trainParam.goal=0.05^2;   %training goal

%Name of a network performance function %type help nnperformance
net_fit.performFcn = 'mse';  

[net_trained, tr] = train(net_fit, x_data', y_data');
return