"""
Created: 5/30/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

Prueba de concepto: Buscar el intervalo solo con los datos inciales de entrenamiento sin interar sobre el lambda.

"""
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from JointSupervision import PICP, PINAW, MAE

mat = scipy.io.loadmat('P_DatosProblema1.mat')
x_train = mat['Xent']
y_train = mat['Yent']
x_test = mat['Xtest']
y_test = mat['Ytest']
x_val = mat['Xval']
y_val = mat['Yval']
Y_real = mat['Y']
X_real = mat['X']


train_stats = np.load('best_train_stats_j_1.npy', allow_pickle=True)[()]

Weigths = train_stats['Weights']
Biases  = train_stats['Biases']
WI = Weigths[0].transpose()
WO = Weigths[1].transpose()
BI = Biases[0].reshape(-1, 1)
BO = Biases[1].reshape(-1, 1)


class MLPPredictive(object):
    def __init__(self, WI, WO, BI, BO):
        self.WI = WI
        self.WO = WO
        self.BI = BI
        self.BO = BO

    def evaluate(self, x_data):
        Z = np.tanh(self.WI.dot(x_data.transpose()) + self.BI)
        out  = self.WO.dot(Z) + self.BO
        return out

    def predictive_ahead(self, j_step, x_data, y_data):
        h = 2

        last_y_k_1_h = x_data[:, 0]
        last_y_k_2_h = x_data[:, 1]
        last_u_k_1_h = x_data[:, 2]
        last_u_k_2_h = x_data[:, 3]
        new_data_h = np.array([last_y_k_1_h,
                               last_y_k_2_h,
                               last_u_k_1_h,
                               last_u_k_2_h]).transpose()
        new_y_h = self.evaluate(new_data_h)
        new_y_k_1_h = new_y_h[1, :-1]
        new_y_k_2_h = last_y_k_1_h[:-1]
        new_u_k_1_h = last_u_k_1_h[1:]
        new_u_k_2_h = last_u_k_2_h[1:]
        while h <= j_step:
            last_y_k_1_h = new_y_k_1_h
            last_y_k_2_h = new_y_k_2_h
            last_u_k_1_h = new_u_k_1_h
            last_u_k_2_h = new_u_k_2_h
            new_data_h = np.array([last_y_k_1_h,
                                   last_y_k_2_h,
                                   last_u_k_1_h,
                                   last_u_k_2_h]).transpose()
            new_y_h = self.evaluate(new_data_h)
            new_y_k_1_h = new_y_h[1, :-1]
            new_y_k_2_h = last_y_k_1_h[:-1]
            new_u_k_1_h = last_u_k_1_h[1:]
            new_u_k_2_h = last_u_k_2_h[1:]
            h += 1
        y_data_pred = new_y_h.transpose()
        y_data_h = y_data[j_step - 1:, :]
        return y_data_h, y_data_pred


def plt_prediction_interval(y_real, y_interval, labels):
    fig = plt.figure(figsize=(6, 5))
    x = np.arange(0, len(y_real))
    y_upper = y_interval[:, 0]
    y_crisp = y_interval[:, 1]
    y_lower = y_interval[:, 2]
    plt.title(labels)
    plt.plot(x, y_real, 'o', label='Real', lw=0.4, markersize=1)
    plt.plot(x, y_crisp, label='Prediction', lw=1.0)
    plt.plot(x, y_upper, label='Upper', lw=0.4)
    plt.plot(x, y_lower, label='Lower', lw=0.4)
    plt.fill_between(x, y_lower, y_upper, alpha=0.4)
    plt.grid()
    plt.xlabel('Número de muestras')
    plt.ylabel('Salida del modelo')
    plt.xlim(0, 500)
    plt.legend()
#%%


model_mlp    = MLPPredictive(WI, WO, BI, BO)
y_val_pred   = model_mlp.evaluate(x_val)
y_train_pred = model_mlp.evaluate(x_train)
y_test_pred  = model_mlp.evaluate(x_test)

JSTEP = 16

y_train_h, y_train_pred_h = model_mlp.predictive_ahead(JSTEP, x_train, y_train)
y_test_h, y_test_pred_h = model_mlp.predictive_ahead(JSTEP, x_test, y_test)
y_val_h, y_val_pred_h = model_mlp.predictive_ahead(JSTEP, x_val, y_val)


error_train = y_train_h - y_train_pred_h[:,1].reshape(-1, 1)
error_test  = y_test_h - y_test_pred_h[:,1].reshape(-1, 1)
error_val   = y_val_h - y_val_pred_h[:,1].reshape(-1, 1)

MSE_train = np.mean(np.square(error_train))
MSE_test = np.mean(np.square(error_test))
MSE_val = np.mean(np.square(error_val))
RMSE_train = np.sqrt(MSE_train)
RMSE_test = np.sqrt(MSE_test)
RMSE_val = np.sqrt(MSE_val)
#%%
picp_train = PICP(y_train_h, y_train_pred_h)
picp_test = PICP(y_test_h, y_test_pred_h)
picp_val = PICP(y_val_h, y_val_pred_h)

pinaw_train = PINAW(y_train_h, y_train_pred_h)
pinaw_test = PINAW(y_test_h, y_test_pred_h)
pinaw_val = PINAW(y_val_h, y_val_pred_h)

print('Train:::::')
print('RMSE: ', RMSE_train, 'PICP: ', picp_train, 'PINAW: ', pinaw_train[0])
print('Test:::::')
print('RMSE: ', RMSE_test, 'PICP: ', picp_test, 'PINAW: ', pinaw_test[0])
print('Validation:::::')
print('RMSE: ', RMSE_val, 'PICP: ', picp_val, 'PINAW: ', pinaw_val[0])

#%%
plt_prediction_interval(y_train_h, y_train_pred_h, labels='Train')
plt_prediction_interval(y_test_h, y_test_pred_h, labels='Test')
plt_prediction_interval(y_val_h, y_val_pred_h, labels='Validation')
plt.show()