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
[net_trained, tr] = NeuralNetwork(NUM_OPT_NEU, x_train, y_train, x_test, y_test, x_val, y_val);%, test_prc, val_prc);

y_train_nn = net_trained(x_train');
y_test_nn = net_trained(x_test');
y_val_nn = net_trained(x_val');
%% Prediction
jpasos = 500;
if jpasos == 1
    y_j_train = y_train_nn;
    y_j_test = y_test_nn;
    y_j_val = y_val_nn;
else
    [y_j_train, y_j_test, y_j_val] = predictiveNN(jpasos, net_trained, x_train, x_test, x_val);
end
error_train = y_train(jpasos: end) - y_j_train';
error_test  = y_test(jpasos: end) - y_j_test';
error_val   = y_val(jpasos: end)- y_j_val';

rmse_train = RMSE(y_train(jpasos: end), y_j_train')
rmse_test = RMSE(y_test(jpasos: end), y_j_test')
rmse_val = RMSE(y_val(jpasos: end), y_j_val')

mse_train = MSE(y_train(jpasos: end), y_j_train')
mse_test = MSE(y_test(jpasos: end), y_j_test')
mse_val = MSE(y_val(jpasos: end), y_j_val')

mae_train = MAE(y_train(jpasos: end), y_j_train')
mae_test = MAE(y_test(jpasos: end), y_j_test')
mae_val = MAE(y_val(jpasos: end), y_j_val')

mape_train = MAPE(y_train(jpasos: end), y_j_train')
mape_test = MAPE(y_test(jpasos: end), y_j_test')
mape_val = MAPE(y_val(jpasos: end), y_j_val')

%%
% Plot Histogram of Error Values
figure()
ploterrhist(error_train,'Train', error_test, 'Test', error_val, 'Validation')

fig_train = figure('Renderer', 'painters', 'Units', 'centimeters', 'Position', [4 4 18 10]);
axes_train = axes('Parent', fig_train);
stairs(y_j_train)
hold on
grid on
ylabel('y(k)')
xlabel('Número de muestras')
stairs(y_train(jpasos: end))
xlim([1 300])
ylim([-2 4])
legend('NN-Entrenamiento', 'Dato Real', 'Location', 'southwest')
InSet_rmse = get(axes_train, 'TightInset');
set(gca(fig_train), 'Position', [InSet_rmse(1:2), 1-InSet_rmse(1)-InSet_rmse(3), 1-InSet_rmse(2)-InSet_rmse(4)]);
axes('position',[0.65,0.6349,0.3058,0.32])
box on % put box around new pair of axes
stairs(y_j_train)
hold on 
grid on
stairs(y_train(jpasos: end))
xlim([200, 250])

fig_test = figure('Renderer', 'painters', 'Units', 'centimeters', 'Position', [4 4 18 10])
axes_test = axes('Parent', fig_test);
stairs(y_j_test)
hold on
grid on
ylabel('y(k)')
xlabel('Número de muestras')
stairs(y_test(jpasos: end))
xlim([1 300])
ylim([-2 4])
legend('NN-Prueba', 'Dato Real', 'Location', 'southwest')
InSet_rmse = get(axes_test, 'TightInset');
set(gca(fig_test), 'Position', [InSet_rmse(1:2), 1-InSet_rmse(1)-InSet_rmse(3), 1-InSet_rmse(2)-InSet_rmse(4)]);
axes('position',[0.65,0.6349,0.3058,0.32])
box on % put box around new pair of axes
stairs(y_j_test)
hold on 
grid on
stairs(y_test(1+ jpasos: end))
xlim([200, 250])

fig_val = figure('Renderer', 'painters', 'Units', 'centimeters', 'Position', [4 4 18 10])
axes_val = axes('Parent', fig_val);
stairs(y_j_val)
hold on
grid on
ylabel('y(k)')
xlabel('Número de muestras')
stairs(y_val(jpasos: end))
xlim([1 300])
ylim([-2.5 4])
legend('NN-Validación', 'Dato Real', 'Location', 'southwest')
InSet_rmse = get(axes_val, 'TightInset');
set(gca(fig_val), 'Position', [InSet_rmse(1:2), 1-InSet_rmse(1)-InSet_rmse(3), 1-InSet_rmse(2)-InSet_rmse(4)]);
axes('position',[0.65,0.6349,0.3058,0.32])
box on % put box around new pair of axes
stairs(y_j_val)
hold on 
grid on
stairs(y_val(jpasos: end))
xlim([200, 250])
