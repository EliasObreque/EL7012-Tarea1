"""
Created: 5/23/2020
Autor: Elias Obreque Sepulveda
email: els.obrq@gmail.com

"""
#%%
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json


def optimizer_js(inp_loss_upper, inp_loss_crisp, inp_loss_lower, learning_rate):
    with tf.compat.v1.variable_scope('optimizer'):
        # Gradient Descent
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
        # Minimizacion de la funcion de costo con el optimizador elegido
        train_upper = optimizer.minimize(inp_loss_upper)
        train_crisp = optimizer.minimize(inp_loss_crisp)
        train_lower = optimizer.minimize(inp_loss_lower)
        #train_step = tf.group(train_upper, train_crisp, train_lower)
        train_step = [train_upper, train_crisp, train_lower]
    return train_step


def optimizer_fn(loss, learning_rate):
    with tf.compat.v1.variable_scope('optimizer'):
        # Gradient Descent
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
        # Minimizacion de la funcion de costo con el optimizador elegido
        train_step = optimizer.minimize(loss)
    return train_step


def loss_upper(logits, labels, loss_function_name, lambdas):
    # Codificacion 'one hot' para las etiquetas de clase
    with tf.compat.v1.variable_scope('loss_upper'):
        # Mean Squared Error loss
        #e_upper = tf.math.subtract(labels[:, 0], tf.transpose(logits[:, 0]))
        e_upper = tf.math.subtract(labels[:, 0], tf.transpose(logits))
        # --------------------------
        loss_upper_s = tf.reduce_mean(tf.square(e_upper), name='mse')

        # Condición de JOINT SUPERVISION
        e_upper_new = tf.nn.relu(e_upper)

        loss_upper_i = tf.reduce_mean(tf.square(e_upper_new), name='mse')
        loss_upper_total = loss_upper_s + lambdas * loss_upper_i

        loss = loss_upper_total
    return loss, e_upper


def loss_lower(logits, labels, loss_function_name, lambdas):
    with tf.compat.v1.variable_scope('loss_lower'):
        #e_lower = tf.math.subtract(labels[:, 2], tf.transpose(logits[:, 2]))
        e_lower = tf.math.subtract(labels[:, 2], tf.transpose(logits))
        # --------------------------
        loss_lower_s = tf.reduce_mean(tf.square(e_lower), name='mse')

        # Condición de JOINT SUPERVISION
        e_lower_new = tf.nn.relu(-e_lower)
        loss_lower_i = tf.reduce_mean(tf.square(e_lower_new), name='mse')
        loss_lower_total = loss_lower_s + lambdas * loss_lower_i
        loss = loss_lower_total
    return loss, e_lower


def loss_crisp(logits, labels, loss_function_name, lambdas):
    # Codificacion 'one hot' para las etiquetas de clase
    with tf.compat.v1.variable_scope('loss'):
        #e_crisp = tf.math.subtract(labels[:, 1], tf.transpose(logits[:, 1]))
        e_crisp = tf.math.subtract(labels[:, 1], tf.transpose(logits))
        # --------------------------
        loss_crisp = tf.reduce_mean(tf.square(e_crisp), name='mse')
        loss = loss_crisp
    return loss, e_crisp


def loss_fn(logits, labels, loss_function_name):
    with tf.compat.v1.variable_scope('loss'):
        # Mean Squared Error loss
        error = tf.math.subtract(labels, logits)
        loss = tf.reduce_mean(tf.square(error), name='mse')
    return loss, error


def loss_total(logits, labels, loss_function_name, lambd):
    loss_u, e_u = loss_upper(logits, labels, loss_function_name, lambd)
    loss_l, e_l = loss_lower(logits, labels, loss_function_name, lambd)
    loss_c, e_c = loss_crisp(logits, labels, loss_function_name, lambd)
    return tf.reduce_mean(loss_c + loss_u + loss_l), e_c + e_u + e_l


def model_fn(inputs, layer_sizes):
    layer = inputs
    n_layers = len(layer_sizes)
    # Capas neuronales
    for i in range(n_layers):
        with tf.compat.v1.variable_scope('layer_' + str(i)):
            previous_size = layer.shape[1]  # Tamaño de entrada a la capa
            # Pesos de la capa oculta i-esima
            weights = tf.compat.v1.get_variable(
                name='weights_' + str(i),
                shape=[previous_size, layer_sizes[i]],
                initializer=tf.compat.v1.glorot_uniform_initializer())

            # Sesgos de la capa oculta i-esima
            biases = tf.compat.v1.get_variable(
                name='biases_' + str(i),
                shape=[layer_sizes[i]],
                initializer=tf.zeros_initializer())
            layer = tf.matmul(layer, weights) + biases
            if i < n_layers - 1:
                # Aplicacion de funcion de activacion de capa oculta
                layer = tf.nn.tanh(layer)
            else:
                # Aplicacion de funcion de activacion de capa de salida
                logits = layer
    return logits


def model_fnInt(inputs, layer_sizes):
    layer = inputs
    n_layers = len(layer_sizes)
    layers_out = []
    # Capas neuronales
    for i in range(n_layers):
        with tf.compat.v1.variable_scope('layer_' + str(i)):
            previous_size = layer.shape[1]  # Tamaño de entrada a la capa
            if i < n_layers - 1:
                # Pesos de la capa oculta i-esima
                weights = tf.compat.v1.get_variable(
                    name='weights_' + str(i),
                    shape=[previous_size, layer_sizes[i]],
                    initializer=tf.compat.v1.glorot_uniform_initializer())

                # Sesgos de la capa oculta i-esima
                biases = tf.compat.v1.get_variable(
                    name='biases_' + str(i),
                    shape=[layer_sizes[i]],
                    initializer=tf.zeros_initializer())
                layer = tf.matmul(layer, weights) + biases
            else:
                for out in range(layer_sizes[i]):
                    # Pesos de la capa oculta i-esima
                    weights = tf.compat.v1.get_variable(
                        name='weights_' + str(i)+str(out),
                        shape=[previous_size, 1],
                        initializer=tf.compat.v1.glorot_uniform_initializer())
                    # Sesgos de la capa oculta i-esima
                    biases = tf.compat.v1.get_variable(
                        name='biases_' + str(i)+str(out),
                        shape=[1],
                        initializer=tf.zeros_initializer())
                    layers_out.append(tf.matmul(layer, weights) + biases)
            if i < n_layers - 1:
                # Aplicacion de funcion de activacion de capa oculta
                layer = tf.nn.tanh(layer)
            else:
                # Aplicacion de funcion de activacion de capa de salida
                logits = layers_out
    return logits

#%%


class MLPClassifier(object):
    """Implementacion de clasificador Perceptron Multicapa.
    """

    def __init__(
            self,
            n_features,
            layer_sizes,
            loss_function_name='cross_entropy',
            learning_rate=0.1,
            batch_size=32,
            max_epochs=100,
            early_stopping=None,
            logdir='logs',
            lambdas=0.5,
            joinLoss=True,
            IntervalPred=False
    ):
        # Limpiar grafo computacional
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.disable_eager_execution()
        # Agregar parametros al objeto
        self.n_features = n_features
        self.layer_sizes = layer_sizes
        self.loss_function_name = loss_function_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.logdir = logdir
        self.gamma = lambdas
        self.IntervalPred = IntervalPred
        self.joinLoss = joinLoss

        self.inputs_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, n_features],
                                                  name='image_placeholder')
        self.labels_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, layer_sizes[-1]],
                                                  name='label_placeholder')
        # Construccion del grafo computacional
        if IntervalPred:
            if joinLoss:
                self.logits_crisp = model_fn(self.inputs_ph, layer_sizes)
                self.loss, self.error_crisp = loss_total(self.logits_crisp, self.labels_ph, loss_function_name, self.gamma)
                self.train_step_crisp = optimizer_fn(self.loss, learning_rate)
            else:
                [self.logits_upper, self.logits_crisp, self.logits_lower] = model_fnInt(self.inputs_ph, layer_sizes)
                self.loss_upper, self.error_upper = loss_upper(self.logits_upper, self.labels_ph, loss_function_name,
                                                               self.gamma)
                self.loss, self.error_crisp = loss_crisp(self.logits_crisp, self.labels_ph, loss_function_name,
                                                         self.gamma)
                self.loss_lower, self.error_lower = loss_lower(self.logits_lower, self.labels_ph, loss_function_name,
                                                               self.gamma)
                [self.train_step_upper, self.train_step_crisp, self.train_step_lower] = optimizer_js(self.loss_upper,
                                                                                                     self.loss,
                                                                                                     self.loss_lower,
                                                                                                     learning_rate)
        else:
            self.logits_crisp = model_fn(self.inputs_ph, layer_sizes)
            self.loss, self.error_crisp = loss_fn(self.logits_crisp, self.labels_ph, loss_function_name)
            self.train_step_crisp = optimizer_fn(self.loss, learning_rate)

        # Fusion de todos los summaries
        self.summ = tf.compat.v1.summary.merge_all()
        # Crear sesion de tensorflow para administrar grafo
        self.sess = tf.compat.v1.Session()

    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Entrenamiento del clasificador con los hiperparametros escogidos.

        Args:
            X_train: Entradas del entrenamiento con dimensiones (n_ejemplos, n_features).
            y_train: Etiquetas del entrenamiento con dimensiones (n_ejemplos,)
            X_train: Entradas de la validacion con dimensiones (n_ejemplos, n_features).
            y_train: Etiquetas de la validacion con dimensiones (n_ejemplos,)

        Returns:
            train_stats: Diccionario con datos historicos del entrenamiento.
        """

        # Creacion de 'writers' que guardan datos para Tensorboard
        print('\n\n[Beginning training of MLP at logdir "%s"]\n' % (self.logdir,))
        # Inicializacion de todas las variables
        self.sess.run(tf.compat.v1.global_variables_initializer())
        # Definicion de variables utiles para el entrenamiento
        n_batches = int(X_train.shape[0] / self.batch_size)
        prev_validation_loss = np.array([100.0, 100, 100])
        validation_period = 10
        val_loss_upper = 0
        train_loss_upper = 0
        val_loss_lower = 0
        train_loss_lower = 0
        early_stop_flag = False
        start_time = time.time()
        iteration_history = []
        train_loss_history = []
        val_loss_history = []
        test_loss_history = []
        validation_checks = 0
        Weights = []
        Bias = []

        # Ciclo que recorre una epoca completa de los datos cada vez
        for epoch in range(self.max_epochs):
            if early_stop_flag:
                # Si early stopping se activo, detener el entrenamiento
                break

            # Ciclo que recorre los mini batches del set de train
            for i in range(n_batches):
                if early_stop_flag:
                    # Si early stopping se activo, detener el entrenamiento
                    break
                iteration = epoch * n_batches + i
                # Obtencion del minibatch actual
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                X_batch = X_train[start:end, :]
                y_batch = y_train[start:end, :]
                # Ejecutar una iteracion de gradiente
                feed_dict = {self.inputs_ph: X_batch, self.labels_ph: y_batch}
                if self.IntervalPred:
                    if self.joinLoss:
                        self.sess.run(self.train_step_crisp, feed_dict=feed_dict)
                    else:
                        self.sess.run(self.train_step_upper, feed_dict=feed_dict)
                        self.sess.run(self.train_step_crisp, feed_dict=feed_dict)
                        self.sess.run(self.train_step_lower, feed_dict=feed_dict)
                else:
                    self.sess.run(self.train_step_crisp, feed_dict=feed_dict)
                # Obtener estadisticas del entrenamiento
                if iteration % validation_period == 0:
                    iteration_history.append(iteration)
                    # Estadisticas en el set de validacion
                    feed_dict = {self.inputs_ph: X_val, self.labels_ph: y_val}
                    val_loss = self.sess.run(self.loss, feed_dict=feed_dict)
                    error = self.sess.run(self.error_crisp, feed_dict=feed_dict)
                    if self.IntervalPred:
                        if not self.joinLoss:
                            val_loss_upper = self.sess.run(self.loss_upper, feed_dict=feed_dict)
                            val_loss_lower = self.sess.run(self.loss_lower, feed_dict=feed_dict)
                    val_loss_history.append(val_loss)
                    # Estadisticas en el set de entrenamiento
                    feed_dict = {self.inputs_ph: X_train, self.labels_ph: y_train}
                    train_loss = self.sess.run(self.loss, feed_dict=feed_dict)
                    if self.IntervalPred:
                        if not self.joinLoss:
                            train_loss_upper = self.sess.run(self.loss_upper, feed_dict=feed_dict)
                            train_loss_lower = self.sess.run(self.loss_lower, feed_dict=feed_dict)
                    train_loss_history.append(train_loss)
                    # Estadisticas en el set de prueba
                    feed_dict = {self.inputs_ph: X_test, self.labels_ph: y_test}
                    test_loss = self.sess.run(self.loss, feed_dict=feed_dict)
                    test_loss_history.append(test_loss)
                    print('Epoch: %d/%d, iter: %d. ' %
                          (epoch + 1, self.max_epochs, iteration), end='')
                    print('Loss C(train/val): %.3f / %.3f.' %
                          (train_loss, val_loss), end='')
                    if self.IntervalPred:
                        if ~self.joinLoss:
                            print('Loss U(train/val): %.3f / %.3f.' %
                                  (train_loss_upper, val_loss_upper), end='')
                            print('Loss L(train/val): %.3f / %.3f.' %
                                  (train_loss_lower, val_loss_lower), end='')

                    # Chequear condicion de early_stopping
                    if self.early_stopping is not None:
                        if np.any(np.array([val_loss, val_loss_upper, val_loss_upper]) > prev_validation_loss):
                            validation_checks += 1
                        else:
                            validation_checks = 0
                            prev_validation_loss = np.array([val_loss, val_loss_upper, val_loss_upper])
                        print(', Val. checks: %d/%d' %
                              (validation_checks, self.early_stopping))
                        if validation_checks >= self.early_stopping:
                            early_stop_flag = True
                            print('Early stopping')
                    else:
                        print('')

            elap_time = time.time() - start_time
            print("Epoch finished. Elapsed time %1.4f [s]\n" % (elap_time,))

        for i in range(len(self.layer_sizes)):
            with tf.compat.v1.variable_scope('layer_' + str(i), reuse=True):
                if i < len(self.layer_sizes) - 1:
                        Wi = self.sess.run(tf.compat.v1.get_variable('weights_'+str(i)))
                        Bi = self.sess.run(tf.compat.v1.get_variable('biases_'+str(i)))
                        Weights.append(Wi)
                        Bias.append(Bi)
                else:
                    if self.IntervalPred and self.joinLoss == False:
                        for out in range(self.layer_sizes[i]):
                            Wi = self.sess.run(tf.compat.v1.get_variable('weights_' + str(i) + str(out)))
                            Bi = self.sess.run(tf.compat.v1.get_variable('biases_' + str(i) + str(out)))
                            Weights.append(Wi)
                            Bias.append(Bi)
                    else:
                        Wi = self.sess.run(tf.compat.v1.get_variable('weights_'+str(i)))
                        Bi = self.sess.run(tf.compat.v1.get_variable('biases_'+str(i)))
                        Weights.append(Wi)
                        Bias.append(Bi)

        # Guardar estadisticas en un diccionario
        train_stats = {'iteration_history': np.array(iteration_history),
                       'train_loss_history': np.array(train_loss_history),
                       'val_loss_history': np.array(val_loss_history),
                       'test_loss_history': np.array(test_loss_history),
                       'Weights': Weights,
                       'Biases': Bias}

        return train_stats

    def error_prediction(self, X_train, y_train, X_val, y_val, X_test, y_test):
        # Estadisticas en el set de prueba
        feed_dict = {self.inputs_ph: X_test}
        test_prediction = self.sess.run(self.logits_crisp, feed_dict=feed_dict)
        # Estadisticas en el set de validacion
        feed_dict = {self.inputs_ph: X_val}
        val_prediction = self.sess.run(self.logits_crisp, feed_dict=feed_dict)
        # Estadisticas en el set de train
        feed_dict = {self.inputs_ph: X_train}
        train_prediction = self.sess.run(self.logits_crisp, feed_dict=feed_dict)
        if self.layer_sizes[-1] != 1:
            error_test = y_test - test_prediction[:, 1].reshape(-1, 1)
            error_val = y_val - val_prediction[:, 1].reshape(-1, 1)
            error_train = y_train - train_prediction[:, 1].reshape(-1, 1)
        else:
            error_test = y_test - test_prediction.reshape(-1, 1)
            error_val = y_val - val_prediction.reshape(-1, 1)
            error_train = y_train - train_prediction.reshape(-1, 1)
        return error_train, error_test, error_val

    def evaluate(self, X_data):
        feed_dict = {self.inputs_ph: X_data}
        y_crisp = self.sess.run(self.logits_crisp, feed_dict=feed_dict)
        out = y_crisp
        if self.IntervalPred and self.joinLoss == False:
            y_upper = self.sess.run(self.logits_upper, feed_dict=feed_dict)
            y_lower = self.sess.run(self.logits_lower, feed_dict=feed_dict)
            out = np.concatenate((y_upper, y_crisp, y_lower), axis=1)
        return out

    def predictive_ahead(self, j_step, x_data, y_data=None):
        h = 2
        n, m = np.shape(x_data)
        new_y_k_j_h = np.zeros(np.shape(x_data))
        last_y_k_j_h = x_data
        new_y_h = self.evaluate(last_y_k_j_h)
        new_y_k_j_h[:, 0] = new_y_h[:, 1]
        new_y_k_j_h[:, 1:] = last_y_k_j_h[:, :m-1]

        while h <= j_step:
            last_y_k_j_h = new_y_k_j_h
            new_y_h = self.evaluate(last_y_k_j_h)
            new_y_k_j_h[:, 0] = new_y_h[:, 1]
            new_y_k_j_h[:, 1:] = last_y_k_j_h[:, :m-1]
            h += 1
        y_data_pred = new_y_h
        if y_data is not None:
            y_data_h = y_data[j_step - 1:, :]
        else:
            y_data_h = None
        return y_data_h, y_data_pred

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


def plot_predictive_real(y_real, y_pred, labels):
    fig = plt.figure(figsize=(13, 5))
    x = np.arange(0, len(y_real))
    plt.title(labels)
    plt.step(x, y_real, label='Real')
    plt.step(x, y_pred, label='Prediction')
    plt.grid()
    plt.xlabel('Número de datos')
    plt.ylabel('y(k)')
    plt.xlim(0, 500)
    plt.legend()

def PINAW(y, y_pred):
    yu = y_pred[:,0]
    yl = y_pred[:,2]
    N = np.size(y)
    R = max(y) - min(y)
    return 1 / (N * R) * np.sum(yu - yl) * 100


def PICP(y, y_pred):
    yu = y_pred[:,0]
    yl = y_pred[:,2]
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