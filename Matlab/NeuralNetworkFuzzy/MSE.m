function mse = MSE(y_model, y_estimated)
%MSE Summary of this function goes here
N = max(size(y_model));

error = (y_model  - y_estimated).^2;

sum_error = sum(error);
mse = (1/N) * sum_error;
end

