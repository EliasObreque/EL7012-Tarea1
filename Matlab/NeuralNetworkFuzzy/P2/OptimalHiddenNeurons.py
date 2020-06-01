"""
Created: 5/28/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

"""
#%%
from JointSupervision import MLPClassifier
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

#%%

NR = 24

y_train, x_train = create_int_out_matrix(datos2015, NR)
y_test, x_test = create_int_out_matrix(datos2017[0:int(len(datos2017)/2)], NR)
y_val, x_val = create_int_out_matrix(datos2017[int(len(datos2017)/2):], NR)

#%
#%%
hide_neural = np.arange(5, 20)

train_stats = {}
error_train = {}
error_test = {}
error_val = {}
MSE_train = []
MSE_test = []
MSE_val = []
RMSE_train = []
RMSE_test = []
RMSE_val = []
best_train_stats = None
best_RMSE = 10
best_mlp = None
best_nn = 1
i = 0
# Busca el minimo RMSE con nn neuronas
for nn in hide_neural:
    mlp = MLPClassifier(
        n_features=NR,
        layer_sizes=[nn, 1],
        loss_function_name='mse',
        learning_rate=0.001,
        batch_size=2,
        max_epochs=2000,
        early_stopping=20,
        logdir='run_1',
        lambdas=0,
        joinLoss=True)

    # ----- Entrenamiento de MLP
    train_stats = mlp.fit(x_train, y_train, x_val, y_val, x_test, y_test)

    error_train, error_test, error_val = mlp.error_prediction(x_train, y_train,
                                                              x_val, y_val,
                                                              x_test, y_test)
    MSE_train.append(np.mean(np.square(error_train)))
    MSE_test.append(np.mean(np.square(error_test)))
    MSE_val.append(np.mean(np.square(error_val)))
    RMSE_train.append(np.sqrt(MSE_train[i]))
    RMSE_test.append(np.sqrt(MSE_test[i]))
    RMSE_val.append(np.sqrt(MSE_val[i]))

    if RMSE_test[i] < best_RMSE:
        best_mlp = mlp
        best_train_stats = train_stats
        best_RMSE = RMSE_test[i]
        best_nn = nn
        print('NN: ', best_nn, ', RMSE: ', best_RMSE)
    i += 1

print('NN: ', best_nn, ', RMSE: ', best_RMSE)

best_train_stats['best_nn'] = best_nn

np.save('best_train_stats_nn'+str(NR)+'.npy', best_train_stats)


def plot_loss_error(train_stats, error_train, error_test, error_val):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    ax[0].plot(train_stats['iteration_history'], train_stats['val_loss_history'], label='Validation')
    ax[0].plot(train_stats['iteration_history'], train_stats['train_loss_history'], label='Training')
    ax[0].plot(train_stats['iteration_history'], train_stats['test_loss_history'], label='Test')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Loss (MSE)')
    ax[0].set_title('Loss evolution during training')
    ax[0].grid()
    ax[0].legend()

    colors = ['blue', 'green', 'red']
    labels = ['Train', 'Test', 'Validation']
    ax[1].hist(np.array([error_train, error_test, error_val]), 20, histtype='bar',
               stacked=True, color=colors, label=labels, lw=2, edgecolor='black')
    ax[1].axvline(0, color='orange', linestyle='-', label='Zero error')
    ax[1].set_xlabel('Error')
    ax[1].legend()
    ax[1].grid()
    plt.draw()

#%%


def plot_predictive_real(y_real, y_pred, labels):
    fig = plt.figure(figsize=(6, 5))
    x = np.arange(0, len(y_real))
    plt.title(labels)
    plt.step(x, y_real, label='Real')
    plt.step(x, y_pred, label='Prediction')
    plt.grid()
    plt.xlim(0, 250)
    plt.xlabel('Número de datos')
    plt.ylabel('y(k)')
    plt.legend()

#%%


plt.figure()
plt.plot(hide_neural, RMSE_train, label='Train')
plt.plot(hide_neural, RMSE_test, label='Test')
plt.plot(hide_neural, RMSE_val, label='Validation')
plt.plot(best_nn, best_RMSE, 'o')
plt.grid()
plt.legend()
plt.xlabel('Número de neuronas')
plt.ylabel('RMSE')

y_val_pred = best_mlp.evaluate(x_val)

plot_predictive_real(y_val, y_val_pred, 'Validation')

plot_loss_error(train_stats, error_train, error_test, error_val)
plt.show()