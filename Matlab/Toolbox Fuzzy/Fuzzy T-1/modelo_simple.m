% Ejemplo simple de identificación difusa
% Tres Clusters


clear all
close all

% generación de datos

model.a=[1/1.1; 1/0.9; 1/1.5];                      % Inverso de la varianza de las Gaussians de las premisas
model.b=[-5; 0; 5];                                 % Valor de las medias de las funciones Gaussianas
model.g=[5 0.5;6 1.5 ;8 -0.5];                      % parámetros de los consecuentes affines

% muestreo datos en tres puntos de operación
R1 = normrnd(model.b(1),1./model.a(1),[1,500]);
R2 = normrnd(model.b(2),1./model.a(2),[1,500]);
R3 = normrnd(model.b(3),1./model.a(3),[1,500]);
x_1= [R1 R2 R3]';                                   % Total de datos
x_m=sort(x_1);                                      % Datos ordenados para graficar


%Gráfica de las gaussianas generadas
for i=1:3
    ant=gaussmf(x_m,[1./model.a(i) model.b(i)]);
    figure(1)
    plot(x_m,ant,'--');
    hold on
end

y_noise=ysim_d(x_m,model.a,model.b,model.g);             % salida contaminada con ruidpo por regla


% Fin generación de datos contamionados con ruido
%-----------------------------------------------------------------------------------------------------%

%----Identificación difusa y cálculo de la covarianza para construcción del

  
x_in      = x_m;                     % Datos de entrada 
f         = size(x_in);              % Tamaño datos entrada
y         = y_noise;                 % Datos de Salida   
cluster   = 3;                       % Para este caso base este dato se conoce apriori
std_j     = zeros(1,cluster); 

%% Call TakagiSugeno function para la identificacipon difusa 

opcion=[1 2];                                                % Usar 1: Para que en el calculo de los consecuentes se obtengan los parámetros requeridos para la generación de los intervalos de confianza                        
                                                             % 2: Tipo de normalización de los datos
                                                             
[model_E, result_E]=TakagiSugeno(y,x_in,cluster,opcion);     % Salida, entrada, cluster, opcion  


% Gráfica funciones generadas con algoritmo de cluster Gustafon-Kessel 
j=1;
for i=1:cluster
    ant=gaussmf(sort(x_in(:,j)),[1./model_E.a(i,j) model_E.b(i,j)]);
    figure(1)
    plot(sort(x_in(:,j)),ant,'r');
    hold on
end
title('Blue: function original, Red: function Cluster');


% Intervalos de confianza basdo en: I. Škrjanc, “Confidence interval of fuzzy models: An example using a waste-water treatment plant,” 
%Chemom. Intell. Lab. Syst., vol. 96, no. 2, pp. 182–187, Apr. 2009.


beta     =   model_E.h;                                % Grados de Activación normalizados por regla obtenidos en el proceso de identificación 
                                                       % de los parámetros de los consecuentes
                                               
yE       = ysim(x_in,model_E.a,model_E.b,model_E.g);   % Calculo de la salida con el modelo difuso obtenido
                                                       % Entrada, model.a (inverso de la varianza),
                                                       % model.b (Valores de las medias de la gaussianas),
                                                       % model.g parámetros de los consecuentes

e_i=y-yE;                                              % Calculo del error ecuacióm (13) del paper 

[N_R,N_P]=size(model_E.g);                             % Numero de reglas y número de parametros del consecuente por regla

n=N_R*N_P;                                             % Número total de parámetros estimados valor necesario en ecuación (24) paper

e_mean=(e_i'*beta)./sum(beta);                         % ecuación (20) paper

miu=sum(beta.^2);                                      % ecuación (24b) paper

% varibales auxiliares ecuación (24)
aux_d=(miu-(n+1));                                     
beta_2=beta.^2;
err_2=(repmat(e_i,[1,N_R])-repmat(e_mean,[f(1),1])).^2;

for j=1:N_R
    std_j(j)=err_2(:,j)'*beta_2(:,j)./(aux_d(j));        % calculo ecuación (24a)
end

% RMSE
errorE=sqrt(mean((yE-y).^2));

%% format figure
figure()
hold on
plot(y,'b')
plot(yE,'r')
legend('Real Output','Model output','Location', 'NorthWest'  );

%% Outputs
M.model       =   model_E;
M.model.std_j =   std_j;
M.result      =   result_E;
M.RMSE        =   errorE;
