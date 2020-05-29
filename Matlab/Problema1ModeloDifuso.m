clear all, clc
load('P_DatosProblema1a'); %Conjunto con 8 regresores

% % %------Seleccion de Variables Relevantes. Análisis de sensibilidad---
[errtest,errent] = clusters_optimo(Ytest,Yent,Xtest,Xent,40);
reglas=5; %Clusters
[p, indice]=sensibilidad(Yent,Xent,reglas);

figure()
c = categorical({'y(k-1)','y(k-2)','y(k-3)','y(k-4)','u(k-1)','u(k-2)','u(k-3)','u(k-4)'},{'y(k-1)','y(k-2)','y(k-3)','y(k-4)','u(k-1)','u(k-2)','u(k-3)','u(k-4)'});
bar(c,indice,'b','LineWidth',2);
xlabel('Variables de entrada')
ylabel('I')

% % Raíz Error Cuadrático Medio (REMS)
errS_8=errortest(Yent,Xent,Ytest,Xtest,reglas); 

load('P_DatosProblema1'); %Conjunto con 4 regresores
errS_4=errortest(Yent,Xent,Ytest,Xtest,reglas);

%-----Obtención modelo. Parametros antecedentes y consecuentes-------------
%Modelo con 4 regresores %y-1, y-2, u-1, u-2

% % Seleccion del número óptimo de clusters
max_clusters=20;
[errtest,errent] = clusters_optimo(Ytest,Yent,Xtest,Xent,max_clusters);

% %Obtencion del modelo
reglas=5;  %Hasta 8
[model, result]=TakagiSugeno(Yent,Xent,reglas,[1 2 2]);

load('P1ModeloDifusoTipo1_P');

%Cluster para la salida
figure()
[~,c]=size(model.h);
for i=1:c
plot(Yent,model.h (:,i),'+')
hold on
end
hold off
title('Clusters para  la salida')
xlabel('y(k)')
ylabel('Grado de pertenencia')

figure()
for i=1:c
plot(Xent(:,1),model.h(:,i),'+')
hold on
end
hold off
title('Clusters para  y(k-1)')
xlabel('y(k-1)')
ylabel('Grado de pertenencia')

figure()
for i=1:c
plot(Xent(:,2),model.h(:,i),'+')
hold on
end
hold off
title('Clusters para  y(k-2)')
xlabel('y(k-2)')
ylabel('Grado de pertenencia')

figure()
for i=1:c
plot(Xent(:,3),model.h(:,i),'+')
hold on
end
hold off
title('Clusters para  u(k-1)')
xlabel('u(k-1)')
ylabel('Grado de pertenencia')

figure()
for i=1:c
plot(Xent(:,4),model.h(:,i),'+')
hold on
end
hold off
title('Clusters para  u(k-2)')
xlabel('u(k-2)')
ylabel('Grado de pertenencia')

%Comprobación del Modelo
yE=ysim(Xent,model.a,model.b,model.g);
eRMSE_E=RMSE(Yent,yE);
eMAPE_E=MAPE(Yent,yE);
eMAE_E=MAE(Yent,yE);
yT=ysim(Xtest,model.a,model.b,model.g);
eRMSE_T=RMSE(Ytest,yT);
eMAPE_T=MAPE(Ytest,yT);
eMAE_T=MAE(Ytest,yT);

% % Evaluación del modelo Original
y=ysim(Xval,model.a,model.b,model.g);
eRMSE_V=RMSE(Yval,y);
eMAPE_V=MAPE(Yval,y);
eMAE_V=MAE(Yval,y);

figure ()
stairs(y,'--')
hold on
stairs(Yval,'red')
legend('Estimación','Modelo real')
xlabel('Número de muestras')
ylabel('Salida del modelo')

figure ()
stairs(y,'--')
hold on
stairs(Yval,'red')
legend('Estimación','Modelo real')
xlabel('Número de muestras')
ylabel('Salida del modelo')
xlim([1 300])

% Prediccion a j-pasos del modelo Original
[y8,x8]=ysim_p(Xval,model.a,model.b,model.g,7);

figure ()
stairs(y8,'--')
hold on
stairs(Yval,'red')
legend('Estimación a 8 pasos','Modelo real')
xlabel('Número de muestras')
ylabel('Salida del modelo')

figure ()
stairs(y8,'--')
hold on
stairs(Yval,'red')
legend('Estimación a 8 pasos','Modelo real')
xlabel('Número de muestras')
ylabel('Salida del modelo')
xlim([1 300])

[y16,x16]=ysim_p(Xval,model.a,model.b,model.g,15);

figure ()
stairs(y16,'--')
hold on
stairs(Yval,'red')
legend('Estimación a 16 pasos','Modelo real')
xlabel('Número de muestras')
ylabel('Salida del modelo')

figure ()
stairs(y16,'--')
hold on
stairs(Yval,'red')
legend('Estimación a 16 pasos','Modelo real')
xlabel('Número de muestras')
ylabel('Salida del modelo')
xlim([1 300])

Calculo de los errores
salida=[y y8 y16];
[~,c]=size(salida);

for i=1:c
eRMSE(i)=RMSE(Yval,salida(:,i));
eMAPE(i)=MAPE(Yval,salida(:,i));
eMAE(i)=MAE(Yval,salida(:,i));
end

%Intervalos Difusos
%Método de la Covarianza
alpha=10;
% Estimación a un paso
[~,yEst_u,yEst_l]=Covarianza(Xent,Yent,Xval,model.a,model.b,model.g,alpha);
% Estimación a  8 paso
[~,yEst_u8,yEst_l8]=Covarianza(Xent,Yent,x8,model.a,model.b,model.g,alpha);
% Estimación a 16 paso
[~,yEst_u16,yEst_l16]=Covarianza(Xent,Yent,x16,model.a,model.b,model.g,alpha);

y_u=[yEst_u; yEst_u8; yEst_u16];
y_l=[yEst_l; yEst_l8; yEst_l16];

for i=1:c
ePINAW(i)= PINAW(Yval,y_u(i,:),y_l(i,:));
ePICP(i)= PICP(Yval,y_u(i,:),y_l(i,:));
plot_Intervalos(Yval',y_u(i,:),y_l(i,:))
end

% % Minimos cuadrados
[g_u,g_l]=MinMax(Xtest,Ytest,model.a,model.b,model.g);
y_u1=ysim(Xval,model.a,model.b,g_u);
y_l1=ysim(Xval,model.a,model.b,g_l);

y_u8=ysim(x8,model.a,model.b,g_u);
y_l8=ysim(x8,model.a,model.b,g_l);

y_u16=ysim(x16,model.a,model.b,g_u);
y_l16=ysim(x16,model.a,model.b,g_l);

y_u=[y_u1 y_u8 y_u16];
y_l=[y_l1 y_l8 y_l16];

for i=1:c
ePINAW_mm(i)= PINAW(Yval,y_u(:,i),y_l(:,i));
ePICP_mm(i)= PICP(Yval,y_u(:,i),y_l(:,i));
plot_Intervalos(Yval',y_u',y_l')
end

