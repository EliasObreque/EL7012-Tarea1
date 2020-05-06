function [net_trained, tr] = NeuralNetwork(num_neu, x_data, y_data, train_prc, test_prc, val_prc)
% Crea una capa oculta en la red
net_fit = fitnet(num_neu,'trainlm');

% Funcion de activacion 
net_fit.layers{1}.transferFcn = 'tansig'; % tansig = tanh
net_fit.divideFcn = 'dividerand';
net_fit.divideParam.trainRatio = train_prc; 
net_fit.divideParam.testRatio = test_prc;
net_fit.divideParam.valRatio = val_prc;


%TRAINING PARAMETERS
net_fit.trainParam.show = 20;  %# of ephocs in display
net_fit.trainParam.lr = 0.05;  %learning rate
net_fit.trainParam.epochs = 1000;  %max epochs
%net_fit.trainParam.goal=0.05^2;  %training goal

%Name of a network performance function %type help nnperformance
net_fit.performFcn = 'mse';  

[net_trained, tr] = train(net_fit, x_data', y_data');
return