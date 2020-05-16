%------------Generar BPRS------------
clc
clear all

loaded_data = load('P_DatosProblema1.mat');

y_model = loaded_data.y;
aprbs = loaded_data.u;

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

x_data = [x_train; x_val; x_test];
y_data = [y_train; y_val; y_test];

% Crea una capa oculta en la red
net_fit = fitnet(10,'trainlm');

[trainInd,valInd,testInd] = divideblock(6000, 0.55, 0.2, 0.25);

% Funcion de activacion 
net_fit.layers{1}.transferFcn = 'tansig'; % tansig = tanh
net_fit.divideFcn = 'divideind';
net_fit.divideParam.trainInd = trainInd;
net_fit.divideParam.testInd = testInd;
net_fit.divideParam.valInd = valInd;   


net_fit.trainParam.max_fail = 50;

%TRAINING PARAMETERS
net_fit.trainParam.show = 200;     %# of ephocs in display
net_fit.trainParam.lr = 0.05;      %learning rate
net_fit.trainParam.epochs = 5000;  %max epochs
%net_fit.trainParam.goal=0.05^2;   %training goal

%Name of a network performance function %type help nnperformance
net_fit.performFcn = 'mse';  

[net_trained, tr] = train(net_fit, x_data', y_data');