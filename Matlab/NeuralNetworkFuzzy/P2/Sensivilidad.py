"""
Created: 5/31/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

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
Nh = len(BI)

#%%

I = np.zeros(NR)

for k in range(0, NR):
    sum_out = 0
    for i in range(0, Nh):
        w_ki = WI[i, k]
        w_i = WO[:, i]
        b_i = BI[i]
        sum_int = 0
        for j in range(0, NR):
            w_ji = WI[i, j]
            x_j = x_train[:, j]
            sum_int = sum_int + w_ji * x_j
        sum_out = sum_out + w_i * (1 - np.square(np.tanh(sum_int + b_i))) * w_ki
    xi_k = sum_out
    mu2_xi = np.square(np.mean(xi_k))
    var2_xi = np.var(xi_k)
    I[k] = mu2_xi + var2_xi

#%%
name_reg = []
for i in range(0, NR):
    var = "y(k-" + str(i + 1) + ")"
    print(var)
    name_reg.append(var)

print(name_reg)
plt.figure()
plt.bar(name_reg, I)
plt.ylabel('Indicador I')
plt.xlabel('Regresor')
plt.grid()
plt.show()