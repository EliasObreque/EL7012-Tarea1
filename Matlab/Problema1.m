clear all, clc
% load('DatosProblema1a'); %Conjunto con 8 regresores
% load('P_DatosProblema1a'); %Conjunto con 8 regresores

% % % %------Seleccion de Variables Relevantes. Análisis de sensibilidad---
% [errtest,errent] = clusters_optimo(Ytest,Yent,Xtest,Xent,40);
reglas=5; %Clusters
% [p, indice]=sensibilidad(Yent,Xent,reglas);
% 
% figure()
% c = categorical({'y(k-1)','y(k-2)','y(k-3)','y(k-4)','u(k-1)','u(k-2)','u(k-3)','u(k-4)'},{'y(k-1)','y(k-2)','y(k-3)','y(k-4)','u(k-1)','u(k-2)','u(k-3)','u(k-4)'});
% bar(c,indice,'b','LineWidth',2);
% xlabel('Variables de entrada')
% ylabel('I')

% Error Cuadrático Medio (EMS)
% errS_8=errortest(Yent,Xent,Ytest,Xtest,reglas)^2;
% % 
% % load('DatosProblema1'); %Conjunto con 4 regresores
load('P_DatosProblema1'); %Conjunto con 4 regresores
% errS_4=errortest(Yent,Xent,Ytest,Xtest,reglas)^2;

%-----Obtención modelo. Parametros antecedentes y consecuentes-------------
%Modelo con 4 regresores %y-1, y-2, u-1, u-2

% % Seleccion del número óptimo de clusters
% max_clusters=20;
% [errtest,errent] = clusters_optimo(Ytest,Yent,Xtest,Xent,max_clusters);

% %Obtencion del modelo
% reglas=12;
% [model, result]=TakagiSugeno(Yent,Xent,reglas,[1 2 2]);

% load('P1ModeloDifusoTipo1');
load('P1ModeloDifusoTipo1_P');

% %Cluster para la salida
% figure()
% [~,c]=size(model.h);
% for i=1:c
% plot(Yent,model.h (:,i),'+')
% hold on
% end
% hold off
% title('Clusters para  la salida')
% xlabel('y(k)')
% ylabel('Grado de pertenencia')
% 
% figure()
% for i=1:c
% plot(Xent(:,1),model.h(:,i),'+')
% hold on
% end
% hold off
% title('Clusters para  y(k-1)')
% xlabel('y(k-1)')
% ylabel('Grado de pertenencia')
% 
% figure()
% for i=1:c
% plot(Xent(:,2),model.h(:,i),'+')
% hold on
% end
% hold off
% title('Clusters para  y(k-2)')
% xlabel('y(k-2)')
% ylabel('Grado de pertenencia')
% 
% figure()
% for i=1:c
% plot(Xent(:,3),model.h(:,i),'+')
% hold on
% end
% hold off
% title('Clusters para  u(k-1)')
% xlabel('u(k-1)')
% ylabel('Grado de pertenencia')
% 
% figure()
% for i=1:c
% plot(Xent(:,4),model.h(:,i),'+')
% hold on
% end
% hold off
% title('Clusters para  u(k-2)')
% xlabel('u(k-2)')
% ylabel('Grado de pertenencia')

% %Evaluación del modelo Original
y=ysim(Xval,model.a,model.b,model.g);

% figure ()
% stairs(y,'--')
% hold on
% stairs(Yval,'red')
% legend('Estimación','Modelo real')
% xlabel('Número de muestras')
% ylabel('Salida del modelo')
% 
% figure ()
% stairs(y,'--')
% hold on
% stairs(Yval,'red')
% legend('Estimación','Modelo real')
% xlabel('Número de muestras')
% ylabel('Salida del modelo')
% xlim([1 300])

%Prediccion a j-pasos del modelo Original
y1=ysim_p(Xval,model.a,model.b,model.g,1);

% figure ()
% stairs(y1,'--')
% hold on
% stairs(Yval,'red')
% legend('Estimación a 1 pasos','Modelo real')
% xlabel('Número de muestras')
% ylabel('Salida del modelo')
% 
% figure ()
% stairs(y1,'--')
% hold on
% stairs(Yval,'red')
% legend('Estimación a 1 pasos','Modelo real')
% xlabel('Número de muestras')
% ylabel('Salida del modelo')
% xlim([1 300])

y8=ysim_p(Xval,model.a,model.b,model.g,8);

% figure ()
% stairs(y8,'--')
% hold on
% stairs(Yval,'red')
% legend('Estimación a 8 pasos','Modelo real')
% xlabel('Número de muestras')
% ylabel('Salida del modelo')
% 
% figure ()
% stairs(y8,'--')
% hold on
% stairs(Yval,'red')
% legend('Estimación a 8 pasos','Modelo real')
% xlabel('Número de muestras')
% ylabel('Salida del modelo')
% xlim([1 300])

y16=ysim_p(Xval,model.a,model.b,model.g,16);

% figure ()
% stairs(y16,'--')
% hold on
% stairs(Yval,'red')
% legend('Estimación a 16 pasos','Modelo real')
% xlabel('Número de muestras')
% ylabel('Salida del modelo')
% 
% figure ()
% stairs(y16,'--')
% hold on
% stairs(Yval,'red')
% legend('Estimación a 16 pasos','Modelo real')
% xlabel('Número de muestras')
% ylabel('Salida del modelo')
% xlim([1 300])

%Calculo de los errores
salida=[y y1 y8 y16];
[~,c]=size(salida);

for i=1:c
eRMSE(i)=RMSE(Yval,salida(:,i));
eMAPE(i)=MAPE(Yval,salida(:,i));
eMAE(i)=MAE(Yval,salida(:,i));
end

alpha=50;
[yEst,yEst_u,yEst_l]=Covarianza(Xent,Yent,Xval,model.a,model.b,model.g,alpha);
plot_Intervalos(yEst,yEst_u,yEst_l)
