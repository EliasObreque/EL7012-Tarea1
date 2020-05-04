%Genera una señal PRBS
clear all;clc;
%Regresores z(k)=[z1(k) z2(k) ... zn(k)]=[y(k-1) y(k-2) u(k-1) u(k-2)]
%Prueba de la señal sin ruido e(k) y entrada escalon unitario
%u(k)=1
N=600;y(1)=0;y(2)=1;
u1=ones(N,1);
for k=1:N-2
     y(k+2)=(0.8-0.5*exp(-(y(k+1)^2)))*y(k+1)-(0.3+0.9*exp(-(y(k+1)^2)))*y(k)+u1(k+1)+0.2*u1(k)+0.1*u1(k+1)*u1(k);
end
y=y';

k=1:1:15;
figure(1)
plot(k,y(1:15))
xlabel('instante o N° de la muestra k-ésima')
ylabel('y(k)')
%*************************************************
clear y;
N=600;
%***************************************
Band=[0 0.2];
Range=[-4,4];
u1 = idinput(N,'prbs',Band,Range);
k=1:1:N;
figure(2)
plot(k,u1)
%stairs(u1)
%Genera una señal APRBS (hay que mantener el ancho de los pulsos de la PBRS
u=u1.*rand(N,1);
%**Esta es la señal APRBS que hay que definir
y(1)=0;y(2)=1; r_b=0.25*randn(N,1);%ruido blanco de media=0 y sigma=0.25 
N=600;
for k=1:N-2
     e(k+2)=0.5*exp(-(y(k+1)^2))*r_b(k+2);
     y(k+2)=(0.8-0.5*exp(-(y(k+1)^2)))*y(k+1)-(0.3+0.9*exp(-(y(k+1)^2)))*y(k)+u(k+1)+0.2*u(k)+0.1*u(k+1)*u(k)+e(k+2);
end
e=e';y=y';

k=[1:1:N]';

figure(2)
plot(k,y)
xlabel('instante o N° de la muestra k-ésima')
ylabel('y(k)')
%**************data para entrenamiento****************************
u_train=u(1:330);y_train=y(3:330);
y_train=y(3:330);y1_train=y(2:329);y2_train=y(1:328);u1_train=u(2:329);u2_train=u(1:328);
z_train_reg=[y1_train y2_train u1_train u2_train];
save z_train_reg z_train_reg
data_train=[y_train z_train_reg];
save data_train data_train;
%**************data para prueba****************************
u_test=u(331:480);y_test=y(331:480);
y_test=y(333:480);y1_test=y(332:479);y2_test=y(331:478);u1_test=u(332:479);u2_test=u(331:478);
z_test_reg=[y1_test y2_test u1_test u2_test];
save z_test_reg z_test_reg
data_test=[y_test z_test_reg];
save data_test data_test;
%**************data para validacion****************************
u_val=u(481:600);y_val=y(481:600);
y_val=y(483:600);y1_val=y(482:599);y2_val=y(481:598);u1_val=u(482:599);u2_val=u(481:598);
z_val_reg=[y1_val y2_val u1_val u2_val];
save z_val_reg z_val_reg
data_val=[y_val z_val_reg];
save data_val data_val;
%Se intenta determinar el numero máximo de Clusters
max_num_clusters=12;%debo agregar 1 mas para que
%compare max_num_clusters-1 Clusters (es decir, prueba hasta con 11
%Clusters
[err_test,err_train]=clusters_optimo(y_test,y_train,z_test_reg,z_train_reg,max_num_clusters);
% l=size(err_test,1);
% k=[1:1:l]';
% plot(k,err_train,'b',k,err_test,'r')
% legend('error_e_n_t_r_e_n_a_m_i_e_n_t_o','error_p_r_u_e_b_a')
% xlabel('N° de Clusters')

[minimo_error_test,indice_min]=min(err_test)
reglas=indice_min;
%En indice_min se encuentra el numero optimo de clusters (o Reglas)
%correspondiente al menor error de prueba vs N° de Clusters
error_test=errortest(y_train,z_train_reg,y_test,z_test_reg,reglas);
[p, indice]=sensibilidad(y_train,z_train_reg,reglas);
m=1:1:size(z_train_reg,2);
figure(4)
stairs(m,indice,'*')
%Se podría eliminar el cuarto regresor o entrada u(k-2) pero como ya
%tenemos los regresores como parte del modelo, no tenemos que eliminar nada
%para la pregunta 2 si sera necesario
[model_train, result_train]=TakagiSugeno(y_train,z_train_reg,reglas,[1 2]);%[1 2] sirve
y_train_hat=ysim(z_train_reg,model_train.a,model_train.b,model_train.g);

k=[1:1:size(y_train,1)]';
figure (5)
plot(k,y_train,'b',k,y_train_hat,'r')
legend('y(k)_t_r_a_i_n','y(k)_t_r_a_i_n hat')
xlabel('k')

%**********************************************************
y_test_hat=ysim(z_test_reg,model_train.a,model_train.b,model_train.g);

k=[1:1:size(y_test,1)]';
figure (6)
plot(k,y_test,'b',k,y_test_hat,'r')
legend('y(k)_t_e_s_t','y(k)_t_e_s_t hat')
xlabel('k')

%****Pruebas en el Conjunto de Validacion*****
y_val_hat=ysim(z_val_reg,model_train.a,model_train.b,model_train.g);

k=[1:1:size(y_val,1)]';
figure (7)
plot(k,y_val,'b',k,y_val_hat,'r')
legend('y(k)_v_a_l','y(k)_v_a_l hat')
xlabel('k')

%Salida y_val anterior pero usando solo 3 reglas para efecto de comparacion simplemente 
reglas=3;
[model_ent, result_ent]=TakagiSugeno(y_train,z_train_reg,reglas,[1 2]);%[1 2] sirve
y_valida_hat=ysim(z_val_reg,model_ent.a,model_ent.b,model_ent.g);
figure (8)
plot(k,y_val,'b',k,y_valida_hat,'r')
title('Solo 3 reglas')
legend('y(k)_v_a_l','y(k)_v_a_l hat')
xlabel('k')

figure(9)
plot(k,y_val,'b',k,y_val_hat,'r',k,y_valida_hat,'g')
legend('y(k)_v_a_l','y(k)_v_a_l hat 7 reglas','y(k)_v_a_l hat 3 reglas')

%****Red Neuronal 4 entradas, 1 capa oculta de 8 neuronas y 1 neurona a la salida****
