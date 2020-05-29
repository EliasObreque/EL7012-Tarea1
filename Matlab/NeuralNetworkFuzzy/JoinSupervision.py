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
import itertools


def optimizer_fn(loss, learning_rate):
    """Construye el grafo computacional de la actualizacion por gradiente.

    Se aplica el algoritmo de optimizacion 'sgd' para ejecutar una
    iteracion de minimizacion por gradiente sobre el loss entregado.

    Args:
        loss: Tensor escalar que corresponde al costo calculado.
        learning_rate: Escalar que indica la tasa de aprendizaje. Al seleccionar
            Adam este parametro es ignorado.
        optimizer_name: 'sgd' o 'adam', selecciona el optimizador.

    Returns:
        train_step: Operacion que ejecuta una iteracion de gradiente.
    """

    with tf.compat.v1.variable_scope('optimizer'):
        # Gradient Descent
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
        # Minimizacion de la funcion de costo con el optimizador elegido
        train_step = optimizer.minimize(loss)
    return train_step


def loss_fn(logits, labels, loss_function_name, gamma):
    """Construye el grafo computacional del calculo de la funcion de costo.

    Se aplica el loss 'loss_function_name' entre los labels reales y la salida
    de la MLP. Ademas, se calcula el accuracy.
    Summaries son agregados para su visualizacion en Tensorboard.

    Args:
        logits: Tensor de dimensiones (batch_size, n_classes) con los logits
        de la salida de la MLP
        labels: Tensor de dimensiones (batch_size,) con las etiquetas reales.
        loss_function_name: 'cross_entropy' o 'mse', selecciona el costo.

    Returns:
        loss: Tensor escalar que corresponde al costo calculado.
        accuracy: Tensor escalar que corresponde al accuracy calculado.
        val_summaries: summaries que son de interes al predecir en la validacion
    """

    # Codificacion 'one hot' para las etiquetas de clase
    n_classes = logits.shape[1]
    val_summaries = []
    with tf.compat.v1.variable_scope('loss'):
        if loss_function_name == 'cross_entropy':
            # Cross Entropy loss
            loss = tf.reduce_mean(
                tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits,
                    labels=labels),
                name='xentropy'
            )
            # Summary de loss
            loss_sum = tf.summary.scalar('xentropy_loss', loss)
        elif loss_function_name == 'mse':
            # Mean Squared Error loss
            error = labels - logits
            e_upper = error[:, 0]
            e_crisp = error[:, 1]
            e_lower = error[:, 2]
            # --------------------------
            loss_crisp = tf.reduce_mean(tf.square(e_crisp), name='mse')
            loss_upper_s = tf.reduce_mean(tf.square(e_upper), name='mse')
            loss_lower_s = tf.reduce_mean(tf.square(e_lower), name='mse')

            # Condición de JOIN SUPERVITION
            e_upper_new = tf.nn.relu(e_upper)
            e_lower_new = tf.nn.relu(-e_lower)

            loss_upper_i = tf.reduce_mean(tf.square(e_upper_new), name='mse')
            loss_lower_i = tf.reduce_mean(tf.square(e_lower_new), name='mse')
            loss_upper_total = loss_upper_s + gamma * loss_upper_i
            loss_lower_total = loss_lower_s + gamma * loss_lower_i

            loss = loss_crisp + loss_upper_total + loss_lower_total
            # Summary de loss
            loss_sum = tf.summary.scalar('mse_loss', loss)
        else:
            raise ValueError('Wrong value for loss_function_name')
    val_summaries.append(loss_sum)
    return loss, val_summaries


def model_fn(inputs, layer_sizes):
    """Construye el grafo computacional de la red MLP.

    Se procesa 'inputs' a través de un perceptron multicapa, cuya salida se
    retorna en forma de logits y probabilidades de cada clase.
    Summaries son agregados para su visualizacion en Tensorboard.

    Args:
        inputs: Tensor de entrada de dimensiones (batch_size, n_features).
        layer_sizes: Lista de enteros que indica el tamaño de cada capa de
            neuronas. La salida de la capa i-esima posee dimensiones
            (batch_size, layer_sizes[i]). El ultimo numero de la lista indica el
            tamaño de la capa de salida, que debe ser igual al numero de clases.

    Returns:
        logits: Tensor de salida lineal de dimensiones (batch_size, n_classes).
        probabilities: Tensor de salida con activacion softmax.
    """

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
            # Summary de la distribucion de los pesos
            tf.summary.histogram('weights_' + str(i), weights)

            # Sesgos de la capa oculta i-esima
            biases = tf.compat.v1.get_variable(
                name='biases_' + str(i),
                shape=[layer_sizes[i]],
                initializer=tf.zeros_initializer())
            # Summary de la distribucion de los sesgos
            tf.summary.histogram('biases_' + str(i), biases)
            # Aplicacion de pesos y sesgos
            layer = tf.matmul(layer, weights) + biases
            if i < n_layers - 1:
                # Aplicacion de funcion de activacion de capa oculta
                layer = tf.nn.tanh(layer)
            else:
                # Aplicacion de funcion de activacion de capa de salida
                logits = layer
    return logits


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
            gamma=0.5
    ):
        """Construye un clasificador Perceptron Multicapa.

        Args:
            n_features: Entero que indica el numero de caracteristicas de las entradas.
            layer_sizes: Lista de enteros que indica el tamaño de cada capa de
            neuronas. La salida de la capa i-esima posee dimensiones
            (batch_size, layer_sizes[i]). El ultimo numero de la lista indica el
            tamaño de la capa de salida, que debe ser igual al numero de clases.
            loss_function_name: 'cross_entropy' o 'mse', selecciona el costo.
                Por defecto es 'cross_entropy'.
            learning_rate: Escalar que indica la tasa de aprendizaje. Al
                seleccionar Adam este parametro es ignorado. Por defecto es 0.1
            batch_size: Entero que indica el tamaño de los mini-batches para
                el entrenamiento de la red.
            max_epochs: Entero que indica el maximo numero de epocas de
                entrenamiento (pasadas completas por los datos de entrada)
            early_stopping: Indica cuantas veces las verificaciones en la
                validacion deben indicar que el costo esta aumentando para
                realizar una detencion temprana. Por defecto es None, lo cual
                desactiva la detencion temprana.
            logdir: String que indica el directorio en donde guardar los
                archivos del entrenamiento. Por defecto es 'logs'.
        """
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
        self.gamma = gamma
        # Tensor que reserva espacio para las imagenes de entrada a la red
        self.inputs_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, n_features],
                                                  name='image_placeholder')
        # Tensor que reserva espacio para las etiquetas de la entrada
        self.labels_ph = tf.compat.v1.placeholder(tf.float32, shape=None,
                                                  name='label_placeholder')
        # Construccion del grafo computacional
        self.logits = model_fn(self.inputs_ph, layer_sizes)
        self.loss, self.val_summ = loss_fn(self.logits, self.labels_ph, loss_function_name, self.gamma)
        self.train_step = optimizer_fn(self.loss, learning_rate)
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
        prev_validation_loss = 100.0
        validation_period = 10
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
                y_batch = y_train[start:end]
                # Ejecutar una iteracion de gradiente
                feed_dict = {self.inputs_ph: X_batch, self.labels_ph: y_batch}
                self.sess.run(self.train_step, feed_dict=feed_dict)
                # Obtener estadisticas del entrenamiento
                if iteration % validation_period == 0:
                    iteration_history.append(iteration)
                    # Estadisticas en el set de validacion
                    feed_dict = {self.inputs_ph: X_val, self.labels_ph: y_val}
                    val_loss = self.sess.run(self.loss, feed_dict=feed_dict)
                    val_loss_history.append(val_loss)
                    # Estadisticas en el set de entrenamiento
                    feed_dict = {self.inputs_ph: X_train, self.labels_ph: y_train}
                    train_loss = self.sess.run(self.loss, feed_dict=feed_dict)
                    train_loss_history.append(train_loss)
                    # Estadisticas en el set de prueba
                    feed_dict = {self.inputs_ph: X_test, self.labels_ph: y_test}
                    test_loss = self.sess.run(self.loss, feed_dict=feed_dict)
                    test_loss_history.append(test_loss)

                    print('Epoch: %d/%d, iter: %d. ' %
                          (epoch + 1, self.max_epochs, iteration), end='')
                    print('Loss (train/val): %.3f / %.3f.' %
                          (train_loss, val_loss), end='')

                    # Chequear condicion de early_stopping
                    if self.early_stopping is not None:
                        if val_loss > prev_validation_loss:
                            validation_checks += 1
                        else:
                            validation_checks = 0
                            prev_validation_loss = val_loss
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
            with tf.compat.v1.variable_scope('layer_'+str(i), reuse=True):
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
        test_prediction = self.sess.run(self.logits, feed_dict=feed_dict)
        # Estadisticas en el set de validacion
        feed_dict = {self.inputs_ph: X_val}
        val_prediction = self.sess.run(self.logits, feed_dict=feed_dict)
        # Estadisticas en el set de train
        feed_dict = {self.inputs_ph: X_train}
        train_prediction = self.sess.run(self.logits, feed_dict=feed_dict)
        error_test = y_test - test_prediction[:, 1].reshape(-1, 1)
        error_val = y_val - val_prediction[:, 1].reshape(-1, 1)
        error_train = y_train - train_prediction[:, 1].reshape(-1, 1)
        return error_train, error_test, error_val

    def evaluate(self, X_data):
        feed_dict = {self.inputs_ph: X_data}
        return self.sess.run(self.logits, feed_dict=feed_dict)


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


def plot_predictive_real(y_real, error, labels):
    fig = plt.figure(figsize=(13, 5))
    x = np.arange(0, len(y_real))
    plt.title(labels)
    plt.step(x, y_real, label='Real')
    plt.step(x, (y_real - error), label='Prediction')
    plt.grid()
    plt.xlabel('Nd []')
    plt.ylabel('y(k)')
    plt.xlim(0, 500)
    plt.legend()


def plt_prediction_interval(y_real, y_interval, labels):
    fig = plt.figure(figsize=(13, 5))
    x = np.arange(0, len(y_real))
    y_upper = y_interval[:, 0]
    y_crisp = y_interval[:, 1]
    y_lower = y_interval[:, 2]
    plt.title(labels)
    plt.plot(x, y_real, '.',label='Real')
    plt.plot(x, y_crisp, label='Prediction')
    plt.plot(x, y_upper, label='Upper')
    plt.plot(x, y_lower, label='Lower')
    plt.fill_between(x, y_lower, y_upper, alpha=0.4)
    plt.grid()
    plt.xlabel('Nd []')
    plt.ylabel('y(k)')
    plt.xlim(0, 300)
    plt.legend()


if __name__ == '__main__':

    # ----- Creacion de MLP
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

    out = 3
    y_train_js = np.hstack([y_train] * 3)
    y_test_js = np.hstack([y_test] * 3)
    y_val_js = np.hstack([y_val] * 3)

    mlp = MLPClassifier(
        n_features=4,
        layer_sizes=[10, out],
        loss_function_name='mse',
        learning_rate=0.05,
        batch_size=200,
        max_epochs=1000,
        early_stopping=50,
        logdir='run_1',
        gamma=20.)

    # ----- Entrenamiento de MLP
    train_stats = mlp.fit(x_train, y_train_js, x_val, y_val_js, x_test, y_test_js)

    #%%
    error_train, error_test, error_val = mlp.error_prediction(x_train, y_train, x_val, y_val, x_test, y_test)

    plot_loss_error(train_stats, error_train, error_test, error_val)
    plot_predictive_real(y_train, error_train, labels='Train')
    plot_predictive_real(y_test, error_test, labels='Test')
    plot_predictive_real(y_val, error_val, labels='Validation')

    y_train_pred = mlp.evaluate(x_train)
    y_test_pred = mlp.evaluate(x_test)
    y_val_pred = mlp.evaluate(x_val)

    plt_prediction_interval(y_train_js, y_train_pred, labels='Validation')
    plt_prediction_interval(y_train_js, y_train_pred, labels='Validation')
    plt_prediction_interval(y_train_js, y_train_pred, labels='Validation')
    plt.show()

