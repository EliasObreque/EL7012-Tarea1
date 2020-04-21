% Alfredo Núñez & Doris Sáez
% Suggestions to dsaez@ing.uchile.cl

% Entrenar modelo TS
%
% Inputs:
%   * y: Vector de salida
%   * Z: Matriz de datos de entrada            (Matrix with the regresors [z1(k) z2(k) ... zn(k)]
%   * reglas: Número de reglas (clusters)
%   * opcion: [Tipo de identificación, Tipo de normalización]

% opcion(1)=1 LMS with all the data (problems may arise with the inverse of this matrix)
% opcion(1)=2 LMS for each rule (data weighted by activation degree of each rule)
% opcion(1)=3 (Under development) a non-linear solver to obtain a good trade-off between global and local rules

% opcion(2)=1 Linear normalization
% opcion(2)=2 Gaussian normalization
% opcion(2)=3 First Gaussian and then linear
% opcion(2)=4 Not normalized (be carefull with the clusters)

% NOTES: We have to include in a generic way:
% 1) MERGE OF CLUSTERS after STEP 2, 2) Non linear optization for GK (sometimes the
% recursions diverge), 3) opcion(1)=3 a non-linear solver to obtain a good
% trade-off between global and local rules in STEP 4, 4) Structural optimization
% 5) Optimization of the number of clusters (the tasks 4 and 5
% could be included as another .m before using this one)


% Outputs:
%   * model: Estructura con el modelo TS
%       * model.a: (Std^-1) de los clusters
%       * model.b: Centros de los clusters
%       * model.g: Parámetros de las consecuencias
%       * model.exitflag: Condición de término del solver empleado (Sólo para opción(1) == {7,8,9})
%   * GK: Resultado del Clustering GK

function [model,GK] = TakagiSugeno(y,Z,reglas,opcion,varargin)

%% Normalización
data.X = [y,Z];   % Todos los datos

if opcion(2) == 1
    data = clust_normalize(data,'range');   % (data.min,data.max) para desnormalizar
elseif opcion(2) == 2
    data = clust_normalize(data,'var');     % (data.mean,data.std) para desnormalizar
elseif opcion(2) == 3
    data = clust_normalize(data,'var2');    % (data.mean,data.std) para desnormalizar
end


%% Identificación de Parámetros de las Premisas
% Parámetros GK
n           = length(Z(1,:));                % n: Número de variables
param.m     = 2;                             % Exponente de peso
param.e     = 1e-5;                          % Tolerancia
param.c     = reglas;                        % Número de clusters
param.ro    = ones(1,param.c);     	         % det(Ai) = ro_i = 1
param.gamma = 0.5;                 	         % Ponderador [0,1]
GK          = GKclustering(data,param);      % GK clustering


%% Ajuste de MF Gaussianas
a = zeros(reglas,n);            % (Std)^-1 de los clusters normalizados
b = GK.cluster.v(:,2:end);      % Centros de los clusters normalizados


for r = 1:reglas    % r: Clusters

	%% Original
    [~,eig_val] = eig(GK.cluster.F(2:end,2:end,r));      %~ Probar con la Matriz A y F dependiendo dimensión del problema

    a(r,:) = 1./sqrt(diag(eig_val))';                    % Inverso de los eig_val

    
% Otros métodos de ajustes de Gaussianas    
    %% Valores y Vectores Propios
%     [eig_vec,eig_val] = eig(GK.cluster.A(2:end,2:end,r));
%     seig_val = sqrt(eig_val)^-1;            % 1/sqrt(eig_val)
%     seig_val_p = abs(eig_vec*seig_val);     % Proyección
% %     a(r,:) = 1./mean(seig_val_p,2)';        % std^-1
%     a(r,:) = 1./max(seig_val_p,[],2)';        % std^-1
    
    %% Ajuste Gaussiana
%     for i = 1:n 
%         fitOptions = fitoptions('gauss1');
%         fitOptions.Normal = 'on';
%         fitOptions.Robust = 'on';
%         fitOptions.MaxIter = 2000;
%         fitOptions.Lower = [0 min(data.X(:,i+1)) 0];
%         fitOptions.Upper = [1 max(data.X(:,i+1)) inf];
%         fitOptions.StartPoint=[1 cl(r,i) 0.1];
% %         fitOptions.Lower = [0.9 cl(r,i)-2*rand(1) 0];
% %         fitOptions.Upper = [1   cl(r,i)+2*rand(1) inf];        
%         data_ord =sortrows([data.X(:,i+1),GK.data.U(:,r)]);
%         %keyboard 
%         %f = fit(data.X(:,i+1),GK.data.U(:,r),'gauss1',fitOptions);
%         f = fit(data_ord(:,1),data_ord(:,2),'gauss1',fitOptions);
%         a(r,i) = (f.c1/sqrt(2))^-1;
%         %a(r,i) = f.c1^-1;
%         b(r,i) = f.b1;
%     end
end


% figure()
% stem(Z(:,1),y);
% legend('Original Data','Location', 'NorthWest'  );
% 
% figure()
% stem(data.X(:,2),data.X(:,1));
% legend('Normalized Data','Location', 'NorthWest'  );


% figure()
% for i=1:reglas
%     if i==1
%         color='blue';
%     elseif i==2
%         color='red';
%     else
%         color='green';
%     end
%     data_ord =sortrows([data.X(:,2),GK.data.U(:,i)]);
%     plot(data_ord(:,1),data_ord(:,2),color)
%     %plot(data.X(:,2),GK.data.U(:,i),color)
%     hold on
% end
% title('membership degree of cluster algorithm')
% 
% for i=1:reglas
%      ant=gaussmf(sort(data.X(:,2)),[1./a(i) b(i)]);
%      figure(10)
%      if i==1
%          color='blue';
%      elseif i==2
%          color='red';
%      else
%          color='green';
%      end
%      plot(sort(data.X(:,2)),ant,color,'LineStyle','--');
%      hold on
%      data_ord =sortrows([data.X(:,2),GK.data.U(:,i)]);
%      plot(data_ord(:,1),data_ord(:,2),color)
%      %plot(data.X(:,2),GK.data.U(:,i),color)
%  end



%% Denormalización
if opcion(2) == 1       % Lineal
    for i = 1:n
        xmax = data.max(1,i+1);
        xmin = data.min(1,i+1);
        dx = xmax - xmin;
        a(:,i) = a(:,i)*(1/dx);
        b(:,i) = xmin + dx*b(:,i);
    end
end

if opcion(2) == 2       % Gaussiana
    for i = 1:n
        xmean = data.mean(1,i+1);
        dx = data.std(1,i+1);
        a(:,i) = a(:,i)*(1/dx);
        b(:,i)  = xmean + dx*b(:,i);
    end
end


if opcion(2) == 3       % Gaussiana + Lineal
    for i = 1:n         % Lineal
        xmax = data.max(1,i+1);
        xmin = data.min(1,i+1);
        dx = xmax - xmin;
        a(:,i) = a(:,i)*(1/dx);
        b(:,i) = xmin + dx*b(:,i);
    end
    for i = 1:n         % Gaussiana
        xmean = data.mean(1,i+1);
        dx = 2*data.std(1,i+1);
        a(:,i) = a(:,i)*(1/dx);
        b(:,i) = xmean + dx*b(:,i);
    end
end

model.a = a;    % (Std^-1) de los clusters desnormalizadas
model.b = b;    % Centros de los clusters desnormalizados

%% Identificación de Parámetros de las Consecuencias
% (Consecuencias afines || lineales)
% STEP 4: Calculation of the consequences

if(opcion(1)==1)
    [g, P, h]=taksug1_n(y,Z,a,b,1); %Global LMS with all the data and parameters for development of intervals confidence
    
    % g parámetros de los consecuentes 
    % P es el término (phi*phi^T)^-1 ecuaciones 35 y 36 paper: 
    % Intervalos de confianza basdo en: I. Škrjanc, “Confidence interval of fuzzy models: An example using a waste-water treatment plant,” 
    %Chemom. Intell. Lab. Syst., vol. 96, no. 2, pp. 182–187, Apr. 2009.
    
    % h grados de activación de las reglas    
end

if(opcion(1)==2)
    g=taksug3(y,Z,a,b);%Identification of each rule separately
end

if(opcion(1)==3)
    g=taksug1(y,Z,a,b);%Identification of each rule, in a fast way
end

if(opcion(1)==4)
    g=taksug2(y,Z,a,b);% Global LMS with all the data
end

% g parámetros de los consecuentes 
% P es el término (phi*phi^T)^-1 ecuaciones 35 y 36 paper: 
% Intervalos de confianza basdo en: I. Škrjanc, “Confidence interval of fuzzy models: An example using a waste-water treatment plant,” 
%Chemom. Intell. Lab. Syst., vol. 96, no. 2, pp. 182–187, Apr. 2009. 
% h grados de activación de las reglas  

model.g=g;
model.P=P;
model.h=h;

end

