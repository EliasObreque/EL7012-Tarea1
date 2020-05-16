function [y_j_train, y_j_test, y_j_val, y_u, y_l] = predictiveNNC(jpasos, net_trained, x_train, x_test, x_val)
%PREDICTIVENN Summary of this function goes here

% Paso j = 2:
u_train_new = x_train(2:end, 3);
u_test_new  = x_test(2:end, 3);
u_val_new   = x_val(2:end, 3);

y_train_new_2 = x_train(1:end - 1, 1);
y_test_new_2 = x_test(1:end - 1, 1);
y_val_new_2 = x_val(1:end-1, 1);

y_train_new_1 = net_trained(x_train');
y_test_new_1 = net_trained(x_test');
y_val_new_1 = net_trained(x_val');

y_train_new_1 = y_train_new_1(1:end-1);
y_test_new_1 = y_test_new_1(1:end - 1);
y_val_new_1 = y_val_new_1(1:end-1);

x_train_new = [y_train_new_1', y_train_new_2, u_train_new];
x_test_new = [y_test_new_1', y_test_new_2, u_test_new];
x_val_new = [y_val_new_1', y_val_new_2, u_val_new];

y_train_new = net_trained(x_train_new');
y_test_new = net_trained(x_test_new');
y_val_new = net_trained(x_val_new');

%--------------------------------

% > 2

for i=3: jpasos
    u_train_new = u_train_new(2:end);
    u_test_new = u_test_new(2:end);
    u_val_new = u_val_new(2:end);
    
    y_train_new_2 = y_train_new_1(1:end-1);
    y_test_new_2 = y_test_new_1(1:end - 1);
    y_val_new_2 = y_val_new_1(1:end-1);
    
    y_train_new_1 = y_train_new(1:end - 1);
    y_test_new_1 = y_test_new(1:end - 1);
    y_val_new_1 = y_val_new(1:end - 1);
    
    x_train_new = [y_train_new_1', y_train_new_2', u_train_new];
    x_test_new = [y_test_new_1', y_test_new_2', u_test_new];
    x_val_new = [y_val_new_1', y_val_new_2', u_val_new];
    
    y_train_new = net_trained(x_train_new');
    y_test_new = net_trained(x_test_new');
    y_val_new = net_trained(x_val_new');
end

y_j_train = y_train_new;
y_j_test = y_test_new;
y_j_val = y_val_new;

[y_u, y_l] = Covariance(net_trained, x_train_new, x_val_new, y_j_val);
end

