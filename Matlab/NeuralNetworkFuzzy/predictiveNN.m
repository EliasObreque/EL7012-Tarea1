function y = predictiveNN(jpasos, net_trained, x_data)
%PREDICTIVENN Summary of this function goes here
for i=1: jpasos
    y = net_trained(x_data');
    
end

end

