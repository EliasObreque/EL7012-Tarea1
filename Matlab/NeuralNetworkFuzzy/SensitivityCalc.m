function I = SensitivityCalc(type_actfunc, num_regresor, x_set, net_properties)
%SENSITIVITYCALC Summary of this function goes here

IW = net_properties.IW{1};
LW = net_properties.LW{2};
B = net_properties.b{1}; 
Nh = size(B, 1);
I = zeros(1, num_regresor);

if type_actfunc == 'tanh'
   for k = 1 : num_regresor
       sum_out = 0;
       for i = 1 : Nh
           w_ki = IW(i, k);
           w_i = LW(i);
           b_i = B(i);
           sum_int = 0;
           for j = 1 : num_regresor
               w_ji = IW(i, j);
               x_j = x_set(~isnan(x_set(:, j)), j);
               sum_int = sum_int + w_ji * x_j;
           end
           sum_out = sum_out + w_i * (1 - tanh(sum_int + b_i).^2) * w_ki;
       end
       xi_k = sum_out;
       mu2_xi = mean(xi_k)^2;
       var2_xi = var(xi_k);
       I(k) = mu2_xi + var2_xi;
   end
end
end

