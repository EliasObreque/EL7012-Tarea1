function I = SensitivityCalc(type_actfunc, num_regresor, x_test, net_properties)
%SENSITIVITYCALC Summary of this function goes here

xi = zeros(size(x_test));
IW = net_properties.IW(1);
LW = net_trained.LW{2};
B = net_trained.b{1} 

if type_actfunc == 'tanh'
   for i=1: num_regresor
       
   end
end
end

