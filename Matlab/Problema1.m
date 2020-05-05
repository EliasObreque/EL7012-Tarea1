clear all, clc
load('DatosProblema1');

% %------Seleccion de Variables Relevantes. Análisis de sensibilidad---
% %Seleccion del número óptimo de clusters
max_clusters=10;
[errtest,errent] = clusters_optimo(Ytest,Yent,Xtest,Xent,max_clusters);
reglas=3;

% %-------Seleccion de variables relevantes-----------------------------
% [~, v]=size(Xent);
% errS=zeros(v,1);
% VarDelete=zeros(v,1);
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
% end
% 
% plot(errS);
% 
% cantEntradas=3;  %y-1, y-2, u-1
% load('DatosProblema1','Xent', 'Xtest');
% for i=v:-1:cantEntradas+1
%     Xent(:,VarDelete(i))=[];
%     Xtest(:,VarDelete(i))=[];
%     Xval(:,VarDelete(i))=[];
% end
% 
% savefile = 'DatosProblema1Fuzzy.mat';
% save(savefile, 'X', 'Xent', 'Xtest','Xval','Y', 'Yent', 'Ytest', 'Yval','reglas');
 
%-----Obtención modelo. Parametros antecedentes y consecuentes-------------
load('DatosProblema1Fuzzy'); %Modelo con 3 reglas y 3 regresores %y-1, y-2, u-1

%Comprobación del número óptimo de clusters
%[errtest,errent] = clusters_optimo(Ytest,Yent,Xtest,Xent,10);

% %Obtencion del modelo
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

