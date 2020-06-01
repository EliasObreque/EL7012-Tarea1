"""
Created: 5/30/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

Prueba de concepto: Buscar el intervalo solo con los datos inciales de entrenamiento sin interar sobre el lambda.

"""
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

datos2015 = scipy.io.loadmat('DatosPV2015.mat')['data2015']
datos2017 = scipy.io.loadmat('DatosPV2017.mat')['data2017']


def create_int_out_matrix(XDATA, nregrsors):
    Y = XDATA[nregrsors:]
    n = np.size(XDATA)
    X = np.zeros((np.size(Y), nregrsors))
    for k in range(nregrsors):
        X[:, nregrsors - k - 1] = XDATA[nregrsors - k - 1: n-1].transpose()[0]
        n -= 1
    return Y, X


def PINAW(y, y_pred):
    yu = y_pred[0]
    yl = y_pred[2]
    N = np.size(y)
    R = max(y) - min(y)
    return 1 / (N * R) * np.sum(yu - yl) * 100


def PICP(y, y_pred):
    yu = y_pred[0]
    yl = y_pred[2]
    N = np.size(y)
    c = 0
    print()
    for i in range(N):
        if yl[i] < y[i] < yu[i]:
            c = c + 1
    return 1 / N * c * 100


def MAE(y, yest):
    N = np.size(y)
    mae = (1 / N) * np.sum(np.abs(y - yest))
    return mae

NR = 6
y_train, x_train = create_int_out_matrix(datos2015, NR)
y_test, x_test = create_int_out_matrix(datos2017[0:int(len(datos2017)/2)], NR)
y_val, x_val = create_int_out_matrix(datos2017[int(len(datos2017)/2):], NR)


train_stats = np.load('best_train_stats_nn6.npy', allow_pickle=True)[()]

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

    def predictive_ahead(self, j_step, x_val, y_data, x_train, alpha):
        h = 2
        last_data = []
        y_crisp = []
        for x_data in [x_val, x_train]:
            n, m = np.shape(x_data)
            last_y_k_1_h = x_data[:, 0]
            last_y_k_2_h = x_data[:, 1]
            last_y_k_3_h = x_data[:, 2]
            last_y_k_4_h = x_data[:, 3]
            last_y_k_5_h = x_data[:, 4]
            last_y_k_6_h = x_data[:, 5]
            new_data_h = np.array([last_y_k_1_h,
                                   last_y_k_2_h,
                                   last_y_k_3_h,
                                   last_y_k_4_h,
                                   last_y_k_5_h,
                                   last_y_k_6_h]).transpose()
            new_y_h = self.evaluate(new_data_h)
            new_y_k_1_h = new_y_h[0][:-1]
            new_y_k_2_h = last_y_k_1_h[:-1]
            new_y_k_3_h = last_y_k_2_h[1:]
            new_y_k_4_h = last_y_k_3_h[1:]
            new_y_k_5_h = last_y_k_4_h[1:]
            new_y_k_6_h = last_y_k_5_h[1:]

            while h <= j_step:
                last_y_k_1_h = new_y_k_1_h
                last_y_k_2_h = new_y_k_2_h
                last_y_k_3_h = new_y_k_3_h
                last_y_k_4_h = new_y_k_4_h
                last_y_k_5_h = new_y_k_5_h
                last_y_k_6_h = new_y_k_6_h
                new_data_h = np.array([last_y_k_1_h,
                                       last_y_k_2_h,
                                       last_y_k_3_h,
                                       last_y_k_4_h,
                                       last_y_k_5_h,
                                       last_y_k_6_h]).transpose()
                new_y_h = self.evaluate(new_data_h)
                new_y_k_1_h = new_y_h[0][:-1]
                new_y_k_2_h = last_y_k_1_h[:-1]
                new_y_k_3_h = last_y_k_2_h[1:]
                new_y_k_4_h = last_y_k_3_h[1:]
                new_y_k_5_h = last_y_k_4_h[1:]
                new_y_k_6_h = last_y_k_5_h[1:]
                h += 1
            y_crisp.append(new_y_h.transpose())
            last_data.append(new_data_h)
        y_upper, y_lower = self.covariance(last_data[0], last_data[1], y_crisp[0], alpha)
        y_data_h = y_data[j_step-1:]
        return y_data_h, [y_upper, y_crisp[0], y_lower]

    def covariance(self, x_data, x_train, y_crisp, alpha):
        Nd = np.size(x_data, 0)

        Z_est = np.tanh(self.WI.dot(x_data.transpose()) + self.BI)
        Z_train = np.tanh(self.WI.dot(x_train.transpose()) + self.BI)

        z_train_2 = Z_train.dot(Z_train.transpose())
        z_train_inv = np.linalg.inv(z_train_2)

        arg = np.diag(np.ones(Nd)) + Z_est.transpose().dot(z_train_inv).dot(Z_est)

        y_u = y_crisp + alpha * np.sqrt(np.var(arg))
        y_l = np.maximum(y_crisp - alpha * np.sqrt(np.var(arg)), 0)
        return y_u, y_l


def plt_prediction_interval(y_real, y_interval, labels):
    fig = plt.figure(figsize=(6, 5))
    x = np.arange(0, len(y_real))
    y_upper = y_interval[0]
    y_crisp = y_interval[1]
    y_lower = y_interval[2]
    plt.title(labels)
    plt.plot(x, y_real, 'o', label='Real', lw=0.4, markersize=1)
    plt.plot(x, y_crisp, label='Prediction', lw=1.0)
    plt.plot(x, y_upper, label='Upper', lw=0.4)
    plt.plot(x, y_lower, label='Lower', lw=0.4)
    plt.fill_between(x, y_lower.transpose()[0], y_upper.transpose()[0], alpha=0.4)
    plt.grid()
    plt.xlim(0, 200)
    plt.xlabel('Número de muestras')
    plt.ylabel('Salida del modelo')
    plt.legend()


def plot_predictive_real(y_real, y_pred, labels):
    fig = plt.figure(figsize=(6, 5))
    x = np.arange(0, len(y_real))
    plt.title(labels)
    plt.step(x, y_real, label='Real')
    plt.step(x, y_pred, label='Prediction')
    plt.grid()
    plt.xlabel('Número de datos')
    plt.ylabel('y(k)')
    plt.legend()
#%%


model_mlp    = MLPPredictive(WI, WO, BI, BO)
y_val_pred   = model_mlp.evaluate(x_val)
y_train_pred = model_mlp.evaluate(x_train)
y_test_pred  = model_mlp.evaluate(x_test)


#plot_predictive_real(y_val, y_val_pred.transpose(), 'Validation')


JSTEP = 12
alpha = 250

y_test_h, y_test_pred_h = model_mlp.predictive_ahead(JSTEP, x_test, y_test, x_train, alpha)
y_val_h, y_val_pred_h = model_mlp.predictive_ahead(JSTEP, x_val, y_val, x_train, alpha)


error_test  = y_test_h - y_test_pred_h[1].reshape(-1, 1)
error_val   = y_val_h - y_val_pred_h[1].reshape(-1, 1)

MSE_test = np.mean(np.square(error_test))
MSE_val = np.mean(np.square(error_val))
RMSE_test = np.sqrt(MSE_test)
RMSE_val = np.sqrt(MSE_val)
#%%
picp_test = PICP(y_test_h, y_test_pred_h)
picp_val = PICP(y_val_h, y_val_pred_h)

pinaw_test = PINAW(y_test_h, y_test_pred_h)
pinaw_val = PINAW(y_val_h, y_val_pred_h)

mae_val = MAE(y_val_h,y_val_pred_h)

print('Test:::::')
print('RMSE: ', RMSE_test, 'PICP: ', picp_test, 'PINAW: ', pinaw_test[0])
print('Validation:::::')
print('RMSE: ', RMSE_val, 'PICP: ', picp_val, 'PINAW: ', pinaw_val[0], 'MAE:', mae_val)

#%%
#plt_prediction_interval(y_test_h, y_test_pred_h, labels='Test')
plt_prediction_interval(y_val_h, y_val_pred_h, labels='Validation')
plt.show()