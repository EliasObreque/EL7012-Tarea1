clear all, clc
load('DatosPV2015');
load('DatosPV2017');

%------------Contrucci�n del vector de datos
Dt=600;         %Tama�o del vector
f=length(y);
ry=10;
ru=5;
Y=zeros(Dt,1);
X=zeros(Dt,ry+ru);
for i=f:-1:f-Dt+1
    Y(Dt,1)=y(i);
    %Regresores de y
    for j=1:ry
        X(Dt,j)=y(i-j);
    end
    Dt=Dt-1;
end

%Selecci�n de datos Aleatoria
%55% para entrenamiento (330) 25% test (150) y 20% validaci�n (120)
rndIDX = randperm(600);

Xent = X(rndIDX(1:330), :);
Yent = Y(rndIDX(1:330), :);

Xtest = X(rndIDX(331:480), :);
Ytest = Y(rndIDX(331:480), :);

Xval = X(rndIDX(481:600), :);
Yval = Y(rndIDX(481:600), :);

savefile = 'DatosProblema1.mat';
save(savefile, 'X', 'Xent', 'Xtest','Xval','Y', 'Yent', 'Ytest', 'Yval');








% %------Seleccion de Variables Relevantes. An�lisis de sensibilidad---
% %Seleccion del n�mero �ptimo de clusters
% max_clusters=6;
% [errtest,errent] = clusters_optimo(Ytest,Yent,Xtest,Xent,max_clusters);
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
 
%-----Obtenci�n modelo. Parametros antecedentes y consecuentes-------------
load('DatosProblema1Fuzzy'); %Modelo con 3 reglas y 3 regresores %y-1, y-2, u-1

%Comprobaci�n del n�mero �ptimo de clusters
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

% %Evaluaci�n del modelo Original
y=ysim(Xval,model.a,model.b,model.g);
% 
figure ()
plot(y,'--')
hold on
plot(Yval,'red')
legend('Estimaci�n','Modelo real')
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
legend('Estimaci�n a 1 pasos','Modelo real')
xlabel('t')
ylabel('Salida del modelo')

y8=ysim_p(Xval,model.a,model.b,model.g,8);

figure ()
plot(y8,'--')
hold on
plot(Yval,'red')
legend('Estimaci�n a 8 pasos','Modelo real')
xlabel('t')
ylabel('Salida del modelo')

y16=ysim_p(Xval,model.a,model.b,model.g,16);

figure ()
plot(y16,'--')
hold on
plot(Yval,'red')
legend('Estimaci�n a 16 pasos','Modelo real')
xlabel('t')
ylabel('Salida del modelo')

