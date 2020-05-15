function [y_u, y_l] = Covariance(net_trained, x_train, y_train, x_test, y_test)
%COVARIANCE Summary of this function goes here
%   Detailed explanation goes here

LW = net_trained.IW{1};
RW = net_trained.LW{2};
B = net_trained.b{1}; 
Nh = size(B, 1);

Z_test = tanh(LW*x_test' + B);
Z_train = tanh(LW*x_train' + B);

z_train_2 = Z_train*Z_train';
z_train_inv = inv(z_train_2);
arg = 1 + Z_test' * z_train_inv  * Z_test; 

alpha = 1;

y_u = y_test' + alpha * var(arg).^0.5';
y_l = y_test' - alpha * var(arg).^0.5';
end

