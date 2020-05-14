%Genera una señal PRBS
clear all;clc;
%Regresores z(k)=[z1(k) z2(k) ... zn(k)]=[y(k-1) y(k-2) u(k-1) u(k-2)]
%Modelo lineal: y(k)=a1*y(k-1)+a2*y(k-2)+b1*u(k-1)+b2*u(k-2)
%data=[Yent Xent];
load DatosProblema1a;%Se cargarn los Yent, Ytest, Yval, Xent,Xtest,Xval
%Conjunto de Prueba o test: 1500 datos
%Usaremos 1480 para luego realizar la predicion hasta 15 pasos
%y tener disponible los valores de entrada u(k+15-1) y u(k+15-2)
y_test=Ytest(1:1480);X_test=Xtest(1:1480,:);u_test=Xtest(1:1480,3);

%Conjunto de Validación: 1200 datos
%Usaremos 1180 para luego realizar la predicion hasta 15 pasos
%y tener disponible los valores de en u(k+15-1) y u(k+15-2)
y_val=Yval(1:1180);X_val=Xval(1:1180,:);u_val=Xval(1:1180,3);

%Conjunto de Entrenamiento: 3300 datos
%Usaremos 3280 para luego realizar la predicion hasta 15 pasos
%y tener disponible los valores de entrada u(k+15-1) y u(k+15-2)
y_ent=Yent(1:3280);X_ent=Xent(1:3280,:);u_ent=Xent(1:3280,3);%En este caso, u_ent es u(k-1)
k=[1:1:size(y_ent,1)]';%definimos las muestras
%**********************************************************
%Minimos Cuadrados sin constante:
%Parametros obtenidos con el conjunto de Entrenamiento
teta_hat =inv(X_ent'*X_ent)*X_ent'*y_ent;
%*****************************************
%****Conjunto de entrenamiento********
y_ent_hat_c=X_ent*teta_hat;
er_L1_RMSE_1p=RMSE(y_ent,y_ent_hat_c)
er_L1_MAPE_1p=MAPE(y_ent,y_ent_hat_c)
er_L1_MAE_1p=MAE(y_ent,y_ent_hat_c)

figure(1)
plot(k,y_ent,k,y_ent_hat_c,'r')
title('Predicicion a 1 paso y(k)_e_n_t hat en el Conjunto de entrenamiento')
xlabel('k [N° de muestras]');
legend('y(k)_e_n_t','y(k)_e_n_t hat')

%Prediccion a 1 paso en conjunto de entrenamiento.
y_ent_1p=teta_hat(1)*X_ent(:,1)+teta_hat(2)*X_ent(:,2)+teta_hat(3)*X_ent(:,3)...
    +teta_hat(4)*X_ent(:,4);
figure(2)
plot(k,y_ent,k,y_ent_1p,'r')
title('Predicicion a 1 paso y(k)_h_a_t en el Conjunto de entrenamiento')
xlabel('k [N° de muestras]');
legend('y(k)_e_n_t','y(k)_e_n_t hat')

%****Conjunto de prueba o test********
y_test_hat=X_test*teta_hat;
er_test_L1_RMSE_1p=RMSE(y_test,y_test_hat)
er_test_L1_MAPE_1p=MAPE(y_test,y_test_hat)
er_test_L1_MAE_1p=MAE(y_test,y_test_hat)
k=[1:1:size(y_test,1)]';
figure(3)
plot(k,y_test,k,y_test_hat,'r')
title('Predicicion a 1 paso y(k)_t_e_s_t hat en el Conjunto de prueba')
xlabel('k [N° de muestras]');
legend('y(k)_t_e_s_t','y(k)_t_e_s_t hat')

%****Conjunto de validacion********
y_val_hat=X_val*teta_hat;
er_val_L1_RMSE_1p=RMSE(y_val,y_val_hat)
er_val_L1_MAPE_1p=MAPE(y_val,y_val_hat)
er_val_L1_MAE_1p=MAE(y_val,y_val_hat)
k=[1:1:size(y_val,1)]';
figure(4)
plot(k,y_val,k,y_val_hat,'r')
title('Predicicion a 1 paso y(k)_v_a_l hat en el Conjunto de validación')
xlabel('k [N° de muestras]');
legend('y(k)_v_a_l','y(k)_v_a_l hat')
%********************************************************

%***Modelo idpoly para datos de entrenamiento 
A=[1 -teta_hat(1) -teta_hat(2)];
B=[0 teta_hat(3) teta_hat(4)];
modelo_lin=idpoly(A,B);
%y_ent_est=idsim(u_ent,modelo_lin);
y_ent_est=sim(u_ent,modelo_lin);
er_id_L1_RMSE_1p=RMSE(y_ent,y_ent_est)
er_id_L1_MAPE_1p=MAPE(y_ent,y_ent_est)
er_id_L1_MAE_1p=MAE(y_ent,y_ent_est)
k=[1:1:size(y_ent,1)]';
figure(5)
plot(k,y_ent,k,y_ent_est,'r')
title('Predicicion a 1 paso y(k)_e_n_t hat en el Conjunto de entrenamiento')
xlabel('k [N° de muestras]');
legend('y(k)_e_n_t','y(k)_e_n_t hat')

%*****Predicion a 8 pasos*****************
data_ent=[y_ent u_ent];
y_ent_8p= predict(modelo_lin,data_ent,7);
er_id_L1_RMSE_8p=RMSE(y_ent,y_ent_8p)
er_id_L1_MAPE_8p=MAPE(y_ent,y_ent_8p)
er_id_L1_MAE_8p=MAE(y_ent,y_ent_8p)

figure(6)
plot(k,y_ent,k,y_ent_8p,'r')
title('Predicicion a 8 pasos y(k)_h_a_t en el Conjunto de entrenamiento')
xlabel('k [N° de muestras]');
legend('y(k)_e_n_t','y(k+7)_e_n_t est')
%*****Predicion a 16 pasos*****************
y_ent_16p= predict(modelo_lin,data_ent,15);
er_id_L1_RMSE_16p=RMSE(y_ent,y_ent_16p)
er_id_L1_MAPE_8p=MAPE(y_ent,y_ent_16p)
er_id_L1_MAE_8p=MAE(y_ent,y_ent_16p)

figure(7)
plot(k,y_ent,k,y_ent_16p,'r')
title('Predicicion a 16 pasos y(k+15)_h_a_t en el Conjunto de entrenamiento')
xlabel('k [N° de muestras]');
legend('y(k)_e_n_t','y(k+15)_e_n_t est')

%***Modelo idpoly para datos de prueba*******************
% y_test_est=sim(u_test,modelo_lin);
% er_id_test_L1_RMSE_1p=RMSE(y_test,y_test_est)
% er_id_test_L1_MAPE_1p=MAPE(y_test,y_test_est)
% er_id_test_L1_MAE_1p=MAE(y_test,y_test_est)
% 
% k=[1:1:size(y_test,1)]';
% figure(8)
% plot(k,y_test,k,y_test_est,'r')
% title('Predicicion a 1 paso y(k)_t_e_s_t hat en el Conjunto de prueba')
% xlabel('k [N° de muestras]');
% legend('y(k)_t_e_s_t','y(k)_t_e_s_t hat')
%***Modelo idpoly para datos de validación*******************
y_val_est=sim(u_val,modelo_lin);
er_id_val_L1_RMSE_1p=RMSE(y_val,y_val_est)
er_id_val_L1_MAPE_1p=MAPE(y_val,y_val_est)
er_id_test_L1_MAE_1p=MAE(y_val,y_val_est)

k=[1:1:size(y_val,1)]';
figure(8)
plot(k,y_val,k,y_val_est,'r')
title('Predicicion a 1 paso y(k)_v_a_l hat en el Conjunto de prueba')
xlabel('k [N° de muestras]');
legend('y(k)_t_e_s_t','y(k)_v_a_l hat')

