function [y_u, y_l] = Covariance(alpha, net_trained, x_train_act, x_test_act, y_est)
%COVARIANCE Summary of this function goes here
%   Detailed explanation goes here
% 
 LW = net_trained.IW{1};
 RW = net_trained.LW{2};
 B = net_trained.b{1}; 
 Nh = size(B, 1);
 
 Nd = size(x_test_act, 1);
% 
Z_test = tanh(LW*x_test_act' + B);
Z_train = tanh(LW*x_train_act' + B);
% 

z_train_2 = Z_train*Z_train';
z_train_inv = inv(z_train_2);

arg = diag(ones(1, Nd)) + Z_test' * z_train_inv  * Z_test; 

y_u = y_est' + alpha * var(arg).^0.5';
y_l = y_est' - alpha * var(arg).^0.5';
end

