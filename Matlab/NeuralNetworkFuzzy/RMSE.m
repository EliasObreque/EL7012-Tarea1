function rmse = RMSE(y_model, y_estimated)
%MSE Summary of this function goes here
N = max(size(y_model));

error = (y_model(~isnan(y_model))  - y_estimated(~isnan(y_estimated))).^2;

sum_error = sum(error);
rmse = (1/N) * sqrt(sum_error);
end

