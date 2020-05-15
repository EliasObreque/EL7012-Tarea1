clear all, clc
% load('P_DatosProblema1a'); %Conjunto con 8 regresores

% % % %------Seleccion de Variables Relevantes. An�lisis de sensibilidad---
% [errtest,errent] = clusters_optimo(Ytest,Yent,Xtest,Xent,40);
% reglas=5; %Clusters
% [p, indice]=sensibilidad(Yent,Xent,reglas);
% 
% figure()
% c = categorical({'y(k-1)','y(k-2)','y(k-3)','y(k-4)','u(k-1)','u(k-2)','u(k-3)','u(k-4)'},{'y(k-1)','y(k-2)','y(k-3)','y(k-4)','u(k-1)','u(k-2)','u(k-3)','u(k-4)'});
% bar(c,indice,'b','LineWidth',2);
% xlabel('Variables de entrada')
% ylabel('I')

% % % Ra�z Error Cuadr�tico Medio (REMS)
% errS_8=errortest(Yent,Xent,Ytest,Xtest,reglas); 
% 
load('P_DatosProblema1'); %Conjunto con 4 regresores
% errS_4=errortest(Yent,Xent,Ytest,Xtest,reglas);

%-----Obtenci�n modelo. Parametros antecedentes y consecuentes-------------
%Modelo con 4 regresores %y-1, y-2, u-1, u-2

% % % Seleccion del n�mero �ptimo de clusters
% max_clusters=20;
% [errtest,errent] = clusters_optimo(Ytest,Yent,Xtest,Xent,max_clusters);

% % %Obtencion del modelo
% reglas=5;  %Hasta 8
% [model, result]=TakagiSugeno(Yent,Xent,reglas,[1 2 2]);

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

% %Comprobaci�n del Modelo
% yE=ysim(Xent,model.a,model.b,model.g);
% eRMSE_E=RMSE(Yent,yE);
% eMAPE_E=MAPE(Yent,yE);
% eMAE_E=MAE(Yent,yE);
% yT=ysim(Xtest,model.a,model.b,model.g);
% eRMSE_T=RMSE(Ytest,yT);
% eMAPE_T=MAPE(Ytest,yT);
% eMAE_T=MAE(Ytest,yT);

% %Evaluaci�n del modelo Original
y=ysim(Xval,model.a,model.b,model.g);
% eRMSE_V=RMSE(Yval,y);
% eMAPE_V=MAPE(Yval,y);
% eMAE_V=MAE(Yval,y);

% figure ()
% stairs(y,'--')
% hold on
% stairs(Yval,'red')
% legend('Estimaci�n','Modelo real')
% xlabel('N�mero de muestras')
% ylabel('Salida del modelo')
% 
% figure ()
% stairs(y,'--')
% hold on
% stairs(Yval,'red')
% legend('Estimaci�n','Modelo real')
% xlabel('N�mero de muestras')
% ylabel('Salida del modelo')
% xlim([1 300])

% % Prediccion a j-pasos del modelo Original
y1=ysim_p(Xval,model.a,model.b,model.g,1);

% figure ()
% stairs(y1,'--')
% hold on
% stairs(Yval,'red')
% legend('Estimaci�n a 1 pasos','Modelo real')
% xlabel('N�mero de muestras')
% ylabel('Salida del modelo')
% 
% figure ()
% stairs(y1,'--')
% hold on
% stairs(Yval,'red')
% legend('Estimaci�n a 1 pasos','Modelo real')
% xlabel('N�mero de muestras')
% ylabel('Salida del modelo')
% xlim([1 300])

y8=ysim_p(Xval,model.a,model.b,model.g,8);

% figure ()
% stairs(y8,'--')
% hold on
% stairs(Yval,'red')
% legend('Estimaci�n a 8 pasos','Modelo real')
% xlabel('N�mero de muestras')
% ylabel('Salida del modelo')
% 
% figure ()
% stairs(y8,'--')
% hold on
% stairs(Yval,'red')
% legend('Estimaci�n a 8 pasos','Modelo real')
% xlabel('N�mero de muestras')
% ylabel('Salida del modelo')
% xlim([1 300])

y16=ysim_p(Xval,model.a,model.b,model.g,16);

% figure ()
% stairs(y16,'--')
% hold on
% stairs(Yval,'red')
% legend('Estimaci�n a 16 pasos','Modelo real')
% xlabel('N�mero de muestras')
% ylabel('Salida del modelo')
% 
% figure ()
% stairs(y16,'--')
% hold on
% stairs(Yval,'red')
% legend('Estimaci�n a 16 pasos','Modelo real')
% xlabel('N�mero de muestras')
% ylabel('Salida del modelo')
% xlim([1 300])

%%Calculo de los errores
salida=[y y1 y8 y16];
[~,c]=size(salida);

% for i=1:c
% eRMSE(i)=RMSE(Yval,salida(:,i));
% eMAPE(i)=MAPE(Yval,salida(:,i));
% eMAE(i)=MAE(Yval,salida(:,i));
% end

% alpha=10;
% [yEst,yEst_u,yEst_l]=Covarianza(Xent,Yent,Xval,model.a,model.b,model.g,alpha);
% for i=1:c
% ePINAW(i)= PINAW(salida(:,i)',yEst_u,yEst_l);
% ePICP(i)= PICP(salida(:,i)',yEst_u,yEst_l);
% plot_Intervalos(salida(:,i)',yEst_u,yEst_l)
% end

%Minimos cuadrados
[g_u,g_l]=MinMax(Xtest,Ytest,model.a,model.b,model.g);
y_u=ysim(Xval,model.a,model.b,g_u);
y_l=ysim(Xval,model.a,model.b,g_l);

for i=1:c
ePINAW_mm(i)= PINAW(salida(:,i)',y_u',y_l');
ePICP_mm(i)= PICP(salida(:,i)',y_u',y_l');
plot_Intervalos(salida(:,i)',y_u',y_l')
end

