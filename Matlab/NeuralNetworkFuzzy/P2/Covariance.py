"""
Created: 5/31/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

datos2015 = scipy.io.loadmat('DatosPV2015.mat')['data2015']
datos2017 = scipy.io.loadmat('DatosPV2017.mat')['data2017']


def create_int_out_matrix(XDATA, nregrsors):
    X = XDATA.reshape(nregrsors, -1).transpose()
    Y = XDATA[nregrsors:np.size(X, 0) + nregrsors]
    return Y, X


NR = 12
y_train, x_train = create_int_out_matrix(datos2015, NR)
y_test, x_test = create_int_out_matrix(datos2017[0:int(len(datos2017)/2)], NR)
y_val, x_val = create_int_out_matrix(datos2017[int(len(datos2017)/2):], NR)

train_stats = np.load('best_train_stats_nn.npy', allow_pickle=True)[()]

Weigths = train_stats['Weights']
Biases  = train_stats['Biases']
WI = Weigths[0].transpose()
WO = Weigths[1].transpose()
BI = Biases[0].reshape(-1, 1)
BO = Biases[1].reshape(-1, 1)
