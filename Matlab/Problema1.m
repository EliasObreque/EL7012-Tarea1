clear all, clc
load('DatosProblema1b'); %Conjunto con 8 regresores

% % % %------Seleccion de Variables Relevantes. Análisis de sensibilidad---
% reglas=3; %Clusters
% [p, indice]=sensibilidad(Yent,Xent,reglas);
% 
% figure()
% c = categorical({'y(k-1)','y(k-2)','y(k-3)','y(k-4)','u(k-1)','u(k-2)','u(k-3)','u(k-4)'},{'y(k-1)','y(k-2)','y(k-3)','y(k-4)','u(k-1)','u(k-2)','u(k-3)','u(k-4)'});
% bar(c,indice,'b','LineWidth',2);
% xlabel('Variables de entrada')
% ylabel('I')

errS_8=errortest(Yent,Xent,Ytest,Xtest,reglas);

load('DatosProblema1a'); %Conjunto con 4 regresores
errS_4=errortest(Yent,Xent,Ytest,Xtest,reglas);

%-----Obtención modelo. Parametros antecedentes y consecuentes-------------
%Modelo con 4 regresores %y-1, y-2, u-1, u-2

% Seleccion del número óptimo de clusters
max_clusters=10;
[errtest,errent] = clusters_optimo(Ytest,Yent,Xtest,Xent,max_clusters);

% %Obtencion del modelo
reglas=3
[model, result]=TakagiSugeno(Yent,Xent,reglas,[1 2]);

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
% figure()
% plot(Xent(:,3),model.h(:,1),'b+',Xent(:,3),model.h (:,2),'r+', Xent(:,3),model.h (:,3),'g+')
% title('Clusters para  u(k-1)')
% xlabel('u(k-1)')
% ylabel('Grado de pertenencia')

% %Evaluación del modelo Original
y=ysim(Xval,model.a,model.b,model.g);
% 
figure ()
plot(y,'--')
hold on
plot(Yval,'red')
legend('Estimación','Modelo real')
xlabel('t')
ylabel('Salida del modelo')
% 
% e1_1=RMSE(Yval,y);
% e1_2=MAPE(Yval,y);
% e1_3=MAE(Yval,y);

%Prediccion a j-pasos del modelo Original
y1=ysim_p(Xval,model.a,model.b,model.g,1);

figure ()
plot(y1,'--')
hold on
plot(Yval,'red')
legend('Estimación a 1 pasos','Modelo real')
xlabel('t')
ylabel('Salida del modelo')

y8=ysim_p(Xval,model.a,model.b,model.g,8);

figure ()
plot(y8,'--')
hold on
plot(Yval,'red')
legend('Estimación a 8 pasos','Modelo real')
xlabel('t')
ylabel('Salida del modelo')

y16=ysim_p(Xval,model.a,model.b,model.g,16);

figure ()
plot(y16,'--')
hold on
plot(Yval,'red')
legend('Estimación a 16 pasos','Modelo real')
xlabel('t')
ylabel('Salida del modelo')

