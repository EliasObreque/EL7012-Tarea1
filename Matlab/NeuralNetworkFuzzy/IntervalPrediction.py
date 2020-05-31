"""
Created: 5/28/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

Funcion que busca el intervalo de prediccion iterativamente

"""
import json
from JointSupervision import MLPClassifier, PINAW, PICP
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

mat = scipy.io.loadmat('P_DatosProblema1.mat')
x_train = mat['Xent']
y_train = mat['Yent']
x_test = mat['Xtest']
y_test = mat['Ytest']
x_val = mat['Xval']
y_val = mat['Yval']
Y_real = mat['Y']
X_real = mat['X']


#%%

y_train_js = np.hstack([y_train] * 3)
y_test_js = np.hstack([y_test] * 3)
y_val_js = np.hstack([y_val] * 3)
mu_PICP = 90
best_nn = 12

#%%

lambd = 30
dlambdas = 1
best_PINAW = 1
current_PICP = 0
current_PINAW = 1
best_PICP = current_PICP
best_lambda = 1
counter = 0
max_counter = 20
best_train_stats = None
best_mlp = None
JSTEP = 16
# Busca el valor de lambda
while current_PICP <= mu_PICP:
    mlp = MLPClassifier(
        n_features=4,
        layer_sizes=[best_nn, 3],
        loss_function_name='mse',
        learning_rate=0.005,
        batch_size=20,
        max_epochs=2000,
        early_stopping=20,
        logdir='run_1',
        lambdas=lambd,
        joinLoss=False,
        IntervalPred=True,
        JSTEP=JSTEP)

    # ----- Entrenamiento de MLP
    train_stats = mlp.fit(x_train, y_train_js, x_val, y_val_js, x_test, y_test_js)

    y_test_h, y_test_pred_h = mlp.predictive_ahead(JSTEP, x_test, y_test)

    current_PICP = PICP(y_test_h, y_test_pred_h)
    current_PINAW = PINAW(y_test_h, y_test_pred_h)
    print('-----------------------------------------------------------------------------------')
    print('PICP:', current_PICP, ', Lambda: ', lambd)

    if current_PICP > best_PICP:
        best_lambda = lambd
        best_PICP = current_PICP
        best_train_stats = train_stats
        best_PINAW = current_PINAW
        best_mlp = mlp
        print(best_PICP)
        print(best_PICP)
        print(best_PICP)
    lambd += dlambdas

# %%

while counter <= max_counter:
    mlp = MLPClassifier(
        n_features=4,
        layer_sizes=[best_nn, 3],
        loss_function_name='mse',
        learning_rate=0.005,
        batch_size=20,
        max_epochs=2000,
        early_stopping=20,
        logdir='run_1',
        lambdas=best_lambda,
        joinLoss=False)
    train_stats = mlp.fit(x_train, y_train_js, x_val, y_val_js, x_test, y_test_js)
    y_test_h, y_test_pred_h = mlp.predictive_ahead(JSTEP, x_test, y_test)
    current_PICP = PICP(y_test_h, y_test_pred_h)
    current_PINAW = PINAW(y_test_h, y_test_pred_h)
    print('counter: ', counter)
    if current_PICP >= mu_PICP:
        if current_PINAW < best_PINAW:
            best_PINAW = current_PINAW
            best_mlp = mlp
            best_PICP = current_PICP
            best_train_stats = train_stats
    counter += 1

print('=====================================================')
print(best_PICP, best_PINAW, best_lambda, best_nn)
print('=====================================================')
best_train_stats['best_lambda'] = best_lambda
best_train_stats['best_PICP'] = best_PICP
best_train_stats['best_PINAW'] = best_PINAW

np.save('best_train_stats_j_'+str(JSTEP)+'.npy', best_train_stats)
#%%


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
def MAE(y, yest):
    N = np.size(y)
    mae = (1 / N) * np.sum(np.abs(y - yest))
    return mae


#%%


y_train_h, y_train_pred_h = best_mlp.predictive_ahead(JSTEP, x_train, y_train)
y_test_h, y_test_pred_h = best_mlp.predictive_ahead(JSTEP, x_test, y_test)
y_val_h, y_val_pred_h = best_mlp.predictive_ahead(JSTEP, x_val, y_val)

error_train = y_train_h - y_train_pred_h[:, 1].reshape(-1,1)
error_test = y_test_h - y_test_pred_h[:, 1].reshape(-1,1)
error_val = y_val_h - y_val_pred_h[:, 1].reshape(-1,1)

mae_val = MAE(y_val_h, y_val_pred_h[:, 1].reshape(-1, 1))
mse_val = np.mean(np.square(error_val))

picp_val = PICP(y_val_h, y_val_pred_h)

pinaw_val = PINAW(y_val_h, y_val_pred_h)

rmse_val = np.sqrt(mse_val)
print('MAE: ', mae_val, 'RMSE: ', rmse_val, 'MSE: ', mse_val, 'PICP: ', picp_val, 'PINAW: ', pinaw_val[0])
print('=====================================================')
plot_loss_error(best_train_stats, error_train, error_test, error_val)

#plt_prediction_interval(y_train_h, y_train_pred, labels='Train')
#plt_prediction_interval(y_test_h, y_test_pred, labels='Test')
plt_prediction_interval(y_val_h, y_val_pred_h, labels='Validation')
plt.show()




