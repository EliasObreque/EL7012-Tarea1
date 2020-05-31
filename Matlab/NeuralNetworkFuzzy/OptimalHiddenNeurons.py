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

mat = scipy.io.loadmat('P_DatosProblema1.mat')
x_train = mat['Xent']
y_train = mat['Yent']
x_test = mat['Xtest']
y_test = mat['Ytest']
x_val = mat['Xval']
y_val = mat['Yval']
Y_real = mat['Y']
X_real = mat['X']


h = 8


if h == 1:
    x_train_h = x_train
    x_test_h = x_test
    x_val_h = x_val
    y_train_h = y_train
    y_test_h = y_test
    y_val_h = y_val
else:
    x_train_h = x_train[:-h + 1, :]
    x_test_h = x_test[:-h + 1, :]
    x_val_h = x_val[:-h + 1, :]
    y_train_h = y_train[h - 1:]
    y_test_h = y_test[h-1:]
    y_val_h = y_val[h-1:]
#%%
hide_neural = np.arange(5, 15)

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

best_RMSE = 10
best_mlp = None
best_nn = 1
i = 0
# Busca el minimo RMSE con nn neuronas
for nn in hide_neural:
    mlp = MLPClassifier(
        n_features=4,
        layer_sizes=[nn, 1],
        loss_function_name='mse',
        learning_rate=0.005,
        batch_size=20,
        max_epochs=2000,
        early_stopping=20,
        logdir='run_1',
        lambdas=0,
        joinLoss=True)

    # ----- Entrenamiento de MLP
    train_stats = mlp.fit(x_train_h, y_train_h, x_val_h, y_val_h, x_test_h, y_test_h)

    error_train, error_test, error_val = mlp.error_prediction(
        x_train_h, y_train_h, x_val_h, y_val_h, x_test_h, y_test_h)
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
    plt.xlabel('Número de datos')
    plt.ylabel('y(k)')
    plt.xlim(0, 500)
    plt.legend()

#%%


plt.figure()
plt.plot(hide_neural, RMSE_train)
plt.plot(hide_neural, RMSE_test)
plt.plot(hide_neural, RMSE_val)
plt.plot(best_nn, best_RMSE, 'o')
plt.grid()
plt.xlabel(r'$\lambda$')
plt.ylabel('RMSE')

y_val_pred = best_mlp.evaluate(x_val_h)

plot_predictive_real(y_val_h, y_val_pred, 'Validation')

plot_loss_error(train_stats, error_train, error_test, error_val)
plt.show()