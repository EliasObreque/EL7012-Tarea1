
% Función para el cálculo del intervalo de confianza basado en el método de
% la covarianza
% interv es el término extra en las ecuaciones 35 y 36 del paper: I. Škrjanc, “Confidence interval of fuzzy models: An example using a waste-water treatment plant,” 
% Chemom. Intell. Lab. Syst., vol. 96, no. 2, pp. 182–187, Apr. 2009.  


function interval= Intervalo(model,val_x)

yts    = ysim(val_x,model.a,model.b,model.g);

interv = ysim2(val_x,model.a,model.b,model.P,model.std_j);

interval.yts      = yts;
interval.interv   = interv;
















