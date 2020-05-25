clear all, clc
% load('DatosPV2015');
% load('DatosPV2017');
% % 
% % % %------------Contrucción del vector de datos-------------------
% Dt=2880;         %Tamaño del vector
% y=data2015;
% u=0;
% ry=24;
% ru=0;
% [X, Y] = createMatrixInput(Dt, ry, ru, y, u);
%  
% %60% para entrenamiento 
% Xent = X;
% Yent = Y;
%  
% Dt=1920;         %Tamaño del vector
% y=data2017;
% [X, Y] = createMatrixInput(Dt, ry, ru, y, u);
% 
% % %Selección de datos
% % %20% test y 20% validación (1920)
% % 
% Xtest = X(1:960, :);
% Ytest = Y(1:960, :);
% 
% Xval = X(961:1920, :);
% Yval = Y(961:1920, :);
% 
% savefile = 'DatosProblema2_r24.mat';
% save(savefile, 'Xent', 'Xtest','Xval', 'Yent', 'Ytest', 'Yval');
% % % %----------------------------------------------------------------------- 

%---------------------Modelo Difuso----------------------------------
load('DatosProblema2_r1');

%------Seleccion de Variables Relevantes---
% Seleccion del número óptimo de clusters
% max_clusters=10;
% [errtest,errent] = clusters_optimo(Ytest,Yent,Xtest,Xent,max_clusters);
reglas=5; %Clusters
err_r1=errortest(Yent,Xent,Ytest,Xtest,reglas);
%-----Obtención modelo. Parametros antecedentes y consecuentes-------------
[model_r1, result_r1]=TakagiSugeno(Yent,Xent,reglas,[1 2 2]);

load('DatosProblema2_r24');
%------Seleccion de Variables Relevantes---
% Seleccion del número óptimo de clusters
% max_clusters=10;
% [errtest,errent] = clusters_optimo(Ytest,Yent,Xtest,Xent,max_clusters);
reglas=5; %Clusters
err_r24=errortest(Yent,Xent,Ytest,Xtest,reglas);
%-----Obtención modelo. Parametros antecedentes y consecuentes-------------
[model_r24, result_r24]=TakagiSugeno(Yent,Xent,reglas,[1 2 2]);

% figure()
% plot(Yent,model.h (:,1),'b+',Yent,model.h (:,2),'r+',Yent,model.h (:,3),'g+')
% title('Clusters para  la salida')
% xlabel('y(k)')
% ylabel('Grado de pertenencia')
% 
% figure()
% plot(Xent(:,1),model.h(:,1),'b+',Xent(:,1),model.h (:,2),'r+', Xent(:,1),model.h (:,3),'g+')
% title('Clusters para  y(k-1)')
% xlabel('y(k-1)')
% ylabel('Grado de pertenencia')
% 
% figure()
% plot(Xent(:,2),model.h(:,1),'b+',Xent(:,2),model.h (:,2),'r+', Xent(:,2),model.h (:,3),'g+')
% title('Clusters para  y(k-2)')
% xlabel('y(k-2)')
% ylabel('Grado de pertenencia')
% 

% %Evaluación del modelo Original
y_r1=ysim(Xval(:,1),model_r1.a,model_r1.b,model_r1.g);
% % 
figure ()
plot(y_r1,'--')
hold on
plot(Yval,'red')
legend('Estimación','Modelo real')
xlabel('t')
ylabel('Salida del modelo')% %Evaluación del modelo Original

y_r24=ysim(Xval,model_r24.a,model_r24.b,model_r24.g);
% 
figure ()
plot(y_r24,'--')
hold on
plot(Yval,'red')
legend('Estimación 24','Modelo real')
xlabel('t')
ylabel('Salida del modelo')




% e1_1=RMSE(Yval,y_r1)
% e1_2=MAPE(Yval,y_r1)
% e1_3=MAE(Yval,y_r1)










%-----------Prediccion a p pasos

% %Prediccion a j-pasos del modelo Original
% y1=ysim_p(Xval,model.a,model.b,model.g,1);
% 
% figure ()
% plot(y1,'--')
% hold on
% plot(Yval,'red')
% legend('Estimación a 1 pasos','Modelo real')
% xlabel('t')
% ylabel('Salida del modelo')
% 
% y8=ysim_p(Xval,model.a,model.b,model.g,8);
% 
% figure ()
% plot(y8,'--')
% hold on
% plot(Yval,'red')
% legend('Estimación a 8 pasos','Modelo real')
% xlabel('t')
% ylabel('Salida del modelo')
% 
% y16=ysim_p(Xval,model.a,model.b,model.g,16);
% 
% figure ()
% plot(y16,'--')
% hold on
% plot(Yval,'red')
% legend('Estimación a 16 pasos','Modelo real')
% xlabel('t')
% ylabel('Salida del modelo')


% %-----------------No--------------------------------
% %Sensibilidad
% % figure()
% % c = categorical({'y(k-1)','y(k-2)','y(k-3)','y(k-4)','y(k-5)'},{'y(k-1)','y(k-2)','y(k-3)','y(k-4)','y(k-5)'});
% % bar(c,indice,'b','LineWidth',2);
% % xlabel('Variables de entrada')
% % ylabel('I')
% 
% 
% 
% [~, v]=size(Xent);
% errS=zeros(v,1);
% VarDelete=zeros(v,1);
% varD=1:v;
% 
% for i=v:-1:1
%     % Calcular el error con el numero de cluster (reglas)
%     errS(i)=errortest(Yent,Xent,Ytest,Xtest,reglas);
%     
%     % Analisis de sensibilidad
%     [p indice]=sensibilidad(Yent,Xent,reglas);
%     VarDelete(i)=p;
%     Xent(:,p)=[];
%     Xtest(:,p)=[];
%     Xval(:,p)=[];
%     varD(p)
%     varD(p)=[];
% end
% 
% plot(errS);

%  savefile = 'DatosProblema2Fuzzy.mat';
%  save(savefile,  'Xent', 'Xtest','Xval', 'Yent', 'Ytest', 'Yval','y','u','e');
