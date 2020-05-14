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
% 
num_regresor_y = 2;
num_regresor_u = 1;
num_regresor = num_regresor_y  + num_regresor_u;

loaded_data = load('DatosProblema1.mat');

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
%%
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
train_prc = 0.55;
test_prc = 0.25;
val_prc = 0.2;

% Elimina el ultima regresor
x_train(:, 4) = [];
x_test(:, 4) = [];
x_val(:, 4) = [];

ntrain = Nd * train_prc;
ntest = Nd * test_prc;
nval = Nd * val_prc;

max_row = 5; max_col = 4;
num_neu_min = 2;
num_neu_max = 41;

I = zeros(num_neu_max - num_neu_min, num_regresor);

rmse_train = zeros(1, num_neu_max - num_neu_min);
rmse_test  = zeros(1, num_neu_max - num_neu_min);
rmse_val   = zeros(1, num_neu_max - num_neu_min);

for num_neu = num_neu_min: 1: num_neu_max
    % NEURALNETWORK
    [net_trained, tr] = NeuralNetwork(num_neu, x_train, y_train, x_test, y_test, x_val, y_val);

    y_train_nn = net_trained(x_train');
    y_test_nn = net_trained(x_test');
    y_val_nn = net_trained(x_val');
    
    % Sensitivity analysis
    % Datos de entrenamiento
    I(num_neu, :) = SensitivityCalc('tanh', num_regresor, x_train, net_trained);

    % RMSE
    rmse_train(num_neu - num_neu_min + 1) = RMSE(y_train, y_train_nn');
    rmse_test(num_neu - num_neu_min + 1)  = RMSE(y_test, y_test_nn');
    rmse_val(num_neu - num_neu_min + 1)   = RMSE(y_val, y_val_nn');
end
%%
fig1 = figure('Name', 'Indicator by number of neurons');
fig2 = figure('Name', 'Indicator by number of neurons');
for num_neu = num_neu_min: num_neu_max
    if num_neu <=21
        figure(fig1)
        if (num_neu - num_neu_min + 1) == max_col
            ax11 = subplot(max_row, max_col, num_neu - num_neu_min + 1);
        elseif (num_neu - num_neu_min + 1) == (1 + max_col*(max_row - 1))
            ax12 = subplot(max_row, max_col, num_neu - num_neu_min + 1);
        end  
        subplot(max_row, max_col, num_neu - num_neu_min + 1)          
    else
        figure(fig2)
        if (num_neu - num_neu_min + 1) == max_col + max_col*max_row
            ax21 = subplot(max_row, max_col,  num_neu - max_row*max_col - 1);
        elseif (num_neu - num_neu_min + 1) == (1 + max_col*(max_row - 1)) + max_col*max_row
            ax22 = subplot(max_row, max_col, num_neu - max_row*max_col - 1);
        end  
        subplot(max_row, max_col, num_neu - max_row*max_col - 1)
    end
    hold on
    grid on
    xlabel('Regresors')
    bar(I(num_neu, :))
    ylabel(join(['I-', num2str(num_neu), 'N_h']))
end



%InSet11 = get(ax11, 'TightInset');
%InSet12 = get(ax12, 'TightInset');
%InSet21 = get(ax21, 'TightInset');
%InSet22 = get(ax22, 'TightInset');

%set(gca(fig1), 'Position', [InSet12(1:2), 1-InSet1(1)-InSet1(3), 1-InSet1(2)-InSet1(4)]);
%set(gca(fig2), 'Position', [InSet22(1:2), 1-InSet2(1)-InSet2(3), 1-InSet2(2)-InSet2(4)]);


%%
fig_rmse = figure('Renderer', 'painters', 'Units', 'centimeters', 'Position', [4 4 18 10])
axes_rmse = axes('Parent', fig_rmse);
hold on
grid on
%set(gca, 'YScale', 'log')
ylabel('RMSE')
xlabel('Número de neuronas')
[n1, m1] = min(rmse_train);
plot(m1 + num_neu_min - 1, n1, '*b', 'DisplayName', 'min-Train')
[n2, m2] = min(rmse_test);
plot(m2 + num_neu_min - 1, n2, '*r', 'DisplayName', 'min-Test')
[n3, m3] = min(rmse_val);
plot(m3 + num_neu_min - 1, n3, '*k', 'DisplayName', 'min-Validation')
plot(num_neu_min:num_neu_max, rmse_train, 'b','DisplayName','Train')
plot(num_neu_min:num_neu_max, rmse_test, 'r','DisplayName','Test')
plot(num_neu_min:num_neu_max, rmse_val, 'k','DisplayName','Validation')
legend()
InSet_rmse = get(axes_rmse, 'TightInset');
set(gca(fig_rmse), 'Position', [InSet_rmse(1:2), 1-InSet_rmse(1)-InSet_rmse(3), 1-InSet_rmse(2)-InSet_rmse(4)]);
% create a new pair of axes inside current figure
axes('position',[.23 .5 .5 .42])
box on % put box around new pair of axes
plot(num_neu_min:num_neu_max, rmse_test, 'r')
hold on 
grid on
plot(m2 + num_neu_min - 1, n2, '*r')
ylim([3.3e-3, 3.6e-3])
xlim([3, 41])
%% PLOT

% plots error vs. epoch for the training, validation, and test performances of the training record TR 
figure()
plotperform(tr)
% Plot training state values
figure()
plottrainstate(tr)
