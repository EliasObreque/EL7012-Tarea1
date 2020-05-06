function [X, Y] = createMatrixInput(Dt, ry, ru, y_model, u)
%CREATEMATRIXINPUT Summary of this function goes here
%------------Contrucción del vector de datos

f  = length(y_model);
Y  = zeros(Dt,1);
X  = zeros(Dt, ry + ru);
for i = f: -1 :f - Dt + 1
    Y(Dt,1) = y_model(i);
    %Regresores de y
        for j = 1:ry
            X(Dt, j) = y_model(i - j);
        end
    %Regresores de u
        for j = 1:ru
            X(Dt, j + ry) = u(i - j);
        end
    Dt = Dt-1;
end
end

