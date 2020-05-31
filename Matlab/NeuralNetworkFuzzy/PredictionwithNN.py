"""
Created: 5/30/2020
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


h = 11


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
hide_neural = 12

# Busca el minimo RMSE con nn neuronas
mlp = MLPClassifier(
    n_features=4,
    layer_sizes=[hide_neural, 1],
    loss_function_name='mse',
    learning_rate=0.005,
    batch_size=20,
    max_epochs=2000,
    early_stopping=40,
    logdir='run_1',
    lambdas=0,
    joinLoss=True)

train_stats = mlp.fit(x_train_h, y_train_h, x_val_h, y_val_h, x_test_h, y_test_h)

error_train, error_test, error_val = mlp.error_prediction(x_train_h,
                                                          y_train_h,
                                                          x_val_h,
                                                          y_val_h,
                                                          x_test_h,
                                                          y_test_h)

MSE_train = np.mean(np.square(error_train))
MSE_test = np.mean(np.square(error_test))
MSE_val = np.mean(np.square(error_val))
RMSE_train = np.sqrt(MSE_train)
RMSE_test = np.sqrt(MSE_test)
RMSE_val = np.sqrt(MSE_val)


print('RMSE: ', RMSE_val)


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


y_val_pred = mlp.evaluate(x_val_h)
y_train_pred = mlp.evaluate(x_train_h)
print('RMSE: ', RMSE_train)
plot_predictive_real(y_val_h, y_val_pred, 'Validation')
plot_predictive_real(y_train_h, y_train_pred, 'Train')
x = np.arange(0, len(y_train))
plt.step(x, y_train)
plot_loss_error(train_stats, error_train, error_test, error_val)
plt.show()