from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import tensorflow as tf
from copy import copy
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, Flatten
import h5py


def norm1(data):
    norms = np.abs(data.sum(axis=1))
    norms[norms == 0] = 1
    return data / norms[:, None]


def totalEnergy(data):
    norms = data.sum(axis=1)
    norms[norms == 0] = 1

    return data / norms[:, None]


def meanStd(data):
    mean_data = np.mean(data, axis=1)
    std_data = np.std(data, axis=1)
    data_norm = (np.transpose(data) - mean_data) / std_data
    data_norm = np.transpose(data_norm)
    return data_norm


# Topological Preprocessing
class RpLayer(Layer):

    def __init__(self, **kwargs):
        kwargs.setdefault('dtype', 'float32')
        super(RpLayer, self).__init__(**kwargs)
        self.rvec = np.concatenate((
                                   np.arange(1, 9), np.arange(1, 65), np.arange(1, 9), np.arange(1, 9), np.arange(1, 5),
                                   np.arange(1, 5), np.arange(1, 5)))
        self.output_dim = (len(self.rvec),)

    def build(self, input_shape):
        # Create the alpha trainable tf.variable
        self.__alpha = self.add_weight(name='alpha',
                                       shape=(1, 1),
                                       initializer=tf.keras.initializers.RandomNormal(mean=1, stddev=0.00005),
                                       trainable=True)

        # Create the beta trainable tf.variable
        self.__beta = self.add_weight(name='beta',
                                      shape=(1, 1),
                                      initializer=tf.keras.initializers.RandomNormal(mean=1, stddev=0.00005),
                                      trainable=True)

        # Create the rvec tf.constant
        self.__rvec = K.constant(copy(self.rvec))
        super(RpLayer, self).build(input_shape)

    def call(self, input):
        Ea = K.sign(input) * K.pow(K.abs(input), self.__alpha)
        rb = K.pow(self.__rvec, self.__beta)
        Ea_sum = tf.reshape(K.sum(Ea, axis=1), (-1, 1))
        out = (Ea * rb) / Ea_sum
        return out

    def get_output_shape_for(self, input_shape):
        return self.output_dim


from tensorflow.keras.callbacks import Callback


class sp(Callback):

    def __init__(self, verbose=True, save_the_best=True, rp_layer=True, patience=True, **kw):
        super(Callback, self).__init__()
        self.__verbose = verbose
        self.__patience = 10
        self.__ipatience = 0
        self.__best_sp = 0.0
        self.__save_the_best = save_the_best
        self.__best_weights = None
        self.__best_epoch = 0
        self._validation_data = None
        self.__rp_layer = rp_layer

        if (self.__rp_layer):
            self.__alpha = 1
            self.__beta = 1

    def set_validation_data(self, v):
        self._validation_data = v

    # This computes dSP/dFA (partial derivative of SP respect to FA)
    def __get_partial_derivative_fa(self, fa, pd):
        c = 0.353553
        up = -(pd * (pd - fa + 1)) / (2 * np.sqrt(pd * (1 - fa))) - np.sqrt(pd * (1 - fa))
        down = np.sqrt(np.sqrt(pd * (1 - fa)) * (pd - fa + 1))
        return c * up / down

    # This computes dSP/dPD (partial derivative of SP respect to PD)
    def __get_partial_derivative_pd(self, fa, pd):
        c = 0.353553
        up = ((1 - fa) * (pd - fa + 1)) / (2 * np.sqrt(pd * (1 - fa))) + np.sqrt(pd * (1 - fa))
        down = np.sqrt(np.sqrt(pd * (1 - fa)) * (pd - fa + 1))
        return c * up / down

    def on_epoch_end(self, epoch, logs={}):

        if self._validation_data is not None:  # Check if _validation_data is set
            y_true = self._validation_data[1]
            y_pred = self.model.predict(self._validation_data[0], batch_size=1024).ravel()
        else:
            raise ValueError(
                "Validation data not set. Please call set_validation_data() before training.")  # Raise error if not set

        # Computes SP
        fa, pd, thresholds = roc_curve(y_true, y_pred)
        sp = np.sqrt(np.sqrt(pd * (1 - fa)) * (0.5 * (pd + (1 - fa))))

        knee = np.argmax(sp)

        # Computes partial derivatives
        partial_pd = self.__get_partial_derivative_pd(fa[knee], pd[knee])
        partial_fa = self.__get_partial_derivative_fa(fa[knee], pd[knee])

        logs['max_sp_val'] = sp[knee]
        logs['max_sp_fa_val'] = fa[knee]
        logs['max_sp_pd_val'] = pd[knee]
        logs['max_sp_partial_derivative_fa_val'] = partial_fa
        logs['max_sp_partial_derivative_pd_val'] = partial_pd

        if (self.__rp_layer):
            self.alpha_beta_history()
            logs['alpha_training'] = self.__alpha
            logs['beta_training'] = self.__beta

        if self.__verbose:
            if (self.__rp_layer):
                print(
                    " - val_sp: {:.4f} (fa:{:.4f},pd:{:.4f}), patience: {}, dSP/dFA: {:.4f}, dSP/dPD: {:.4f}, alpha: {:.4f}, beta: {:.4f} ".format(
                        sp[knee],
                        fa[knee], pd[knee], self.__ipatience, partial_fa, partial_pd, self.__alpha, self.__beta))

            else:
                print(" - val_sp: {:.4f} (fa:{:.4f},pd:{:.4f}), patience: {}, dSP/dFA: {:.4f}, dSP/dPD: {:.4f} ".format(
                    sp[knee],
                    fa[knee], pd[knee], self.__ipatience, partial_fa, partial_pd))

        if sp[knee] > self.__best_sp:
            self.__best_sp = sp[knee]
            if self.__save_the_best:
                self.__best_weights = self.model.get_weights()
                logs['max_sp_best_epoch_val'] = epoch
            self.__ipatience = 0
        else:
            self.__ipatience += 1

        if self.__ipatience > self.__patience:
            self.model.stop_training = True
            self.__ipatience = 0
            self.__best_sp = 0.0
            self.__best_epoch = 0

    def alpha_beta_history(self, logs={}):
        self.__alpha = self.model.trainable_weights[0][0][0].numpy()
        self.__beta = self.model.trainable_weights[1][0][0].numpy()


# Tensor flow based ML models

def get_model_mlp_rp(neuron_min, neuron_max, numInit):
    modelCol = []
    for n in range(neuron_min, neuron_max + 1):
        for init in range(numInit):
            inputs = layers.Input(shape=(100,), name='Input_rings')
            input_rp = RpLayer()(inputs)
            dense_rp = layers.Dense(n, activation='sigmoid', name='dense_rp_layer')(input_rp)
            dense = layers.Dense(1, activation='linear', name='output_for_inference')(dense_rp)
            outputs = layers.Activation('sigmoid', name='output_for_training')(dense)
            model = tf.keras.Model(inputs, outputs, name="model")
            modelCol.append(model)

    return modelCol


def get_model_mlp(neuron_min, neuron_max, numInit):
    modelCol = []
    for n in range(neuron_min, neuron_max + 1):
        for init in range(numInit):
            model = Sequential()
            model.add(Dense(n, input_shape=(100,), activation='tanh'))
            model.add(Dense(1, activation='linear'))
            model.add(Activation('sigmoid'))
            modelCol.append(model)

    return modelCol


def get_model_conv(inputShape, kernel_size=2, use_l2=False, numInit=10):
    # expect an input with 100 domensions (features)
    modelCol = []
    for init in range(numInit):
        input = layers.Input(shape=(inputShape,), name='Input')  # 0
        input_reshape = layers.Reshape((inputShape, 1), name='Reshape_layer')(input)
        conv = layers.Conv1D(4, kernel_size=kernel_size, activation='relu', name='conv1d_layer_1')(input_reshape)  # 1
        conv = layers.Conv1D(8, kernel_size=kernel_size, activation='relu', name='conv1d_layer_2')(conv)  # 2
        conv = layers.Flatten(name='flatten')(conv)  # 3
        dense = layers.Dense(16, activation='relu', name='dense_layer')(conv)  # 4
        if use_l2:
            dense = layers.Dense(1, activation='linear', name='output_for_inference', kernel_regularizer='l2',
                                 bias_regularizer='l2')(dense)  # 5
        else:
            dense = layers.Dense(1, activation='linear', name='output_for_inference')(dense)  # 5
        output = layers.Activation('sigmoid', name='output_for_training')(dense)  # 6
        model = tf.keras.Model(input, output, name="model")
        modelCol.append(model)

    return modelCol


def get_model_conv_rp(inputShape, kernel_size=2, use_l2=False, numInit=10):
    # expect an input with 100 domensions (features)
    modelCol = []

    for init in range(numInit):
        inputs = layers.Input(shape=(inputShape,), name='Input')  # 0
        input_rp = RpLayer()(inputs)
        input_reshape = layers.Reshape((inputShape, 1), name='Reshape_layer')(input_rp)
        conv = layers.Conv1D(4, kernel_size=kernel_size, activation='relu', name='conv1d_layer_1')(input_reshape)  # 1
        conv = layers.Conv1D(8, kernel_size=kernel_size, activation='relu', name='conv1d_layer_2')(conv)  # 2
        conv = layers.Flatten(name='flatten')(conv)  # 3
        dense = layers.Dense(16, activation='relu', name='dense_layer')(conv)  # 4
        if use_l2:
            dense = layers.Dense(1, activation='linear', name='output_for_inference', kernel_regularizer='l2',
                                 bias_regularizer='l2')(dense)  # 5
        else:
            dense = layers.Dense(1, activation='linear', name='output_for_inference')(dense)  # 5
        output = layers.Activation('sigmoid', name='output_for_training')(dense)  # 6
        model = tf.keras.Model(inputs, output, name="model_init%i_neuron%i" % (init, 16))
        modelCol.append(model)

    return modelCol


# evaluation

def sp_metrics(pf, pd):
    sp = np.sqrt(np.sqrt(pd * (1 - fa)) * (0.5 * (pd + (1 - fa))))
    sp_max = np.max(sp)

    return sp_max, sp


# Rp topological pre processing analysis

def getBest_alphaBeta(history):
    alpha = history['alpha_training'][-1]
    beta = history['beta_training'][-1]

    return alpha, beta


def getBest_alphaBeta_history(history):
    alpha_hist = history['alpha_training']
    beta_hist = history['beta_training']

    return alpha_hist, beta_hist


def getPerfRef(model_perf, pd):
    pf_ref = []
    sp_ref = []
    for i, k in enumerate(model_perf['roc_pd_test']):
        if k >= pd:
            pf_ref.append(model_perf['roc_pf_test'][i])
            sp_ref.append(model_perf['sp_test'][i])

            return model_perf['sp_test'][i]


# dumping information

def dumpModel(models, history, skf, x, y, seed_cv, numInit):
    vars = ['model', 'fold', 'init', 'history', 'seed_cv', 'output_test', 'output_train',
            'sp_test', 'sp_train', 'pd_test', 'pd_train', 'pf_test', 'pf_train',
            'thr_train', 'thr_test', 'roc_sp_test', 'roc_sp_train', 'roc_pd_test',
            'roc_pf_test', 'roc_pd_train', 'roc_pf_train', 'roc_thr_test', 'roc_thr_train',
            'mse_train', 'mse_test', 'size_sig_test', 'size_bkg_test']

    d = {key: [] for key in vars}
    init = 0
    for i, (train, test) in enumerate(skf.split(x, y)):

        for model_idx, model in enumerate(models):

            print(f"Dumping Fold {i}:")
            print(f"Dumping Init {init}:")

            y_pred_test = model.predict(x[test])
            y_pred_train = model.predict(x[train])

            d['model'].append(model.to_json())  # Store model configuration as JSON string
            d['fold'].append(i)

            d['history'].append(
                history[model_idx + i * len(models)].history)  # Access history from the correct History object
            d['seed_cv'].append(seed_cv)

            fa, pd, thr = roc_curve(y[train], y_pred_train)
            sp = np.sqrt(np.sqrt(pd * (1 - fa)) * (0.5 * (pd + (1 - fa))))
            knee = np.argmax(sp)

            d['output_train'].append(y_pred_train.tolist())  # Convert to list
            d['mse_train'].append(mean_squared_error(y[train], y_pred_train))
            d['sp_train'].append(sp[knee])
            d['pd_train'].append(pd[knee])
            d['pf_train'].append(fa[knee])
            d['thr_train'].append(thr[knee])
            d['roc_sp_train'].append(sp.tolist())  # Convert to list
            d['roc_pd_train'].append(pd.tolist())  # Convert to list
            d['roc_pf_train'].append(fa.tolist())  # Convert to list
            d['roc_thr_train'].append(thr.tolist())  # Convert to list

            fa, pd, thr = roc_curve(y[test], y_pred_test)
            sp = np.sqrt(np.sqrt(pd * (1 - fa)) * (0.5 * (pd + (1 - fa))))
            knee = np.argmax(sp)

            d['output_test'].append(y_pred_test.tolist())  # Convert to list
            d['mse_test'].append(mean_squared_error(y[test], y_pred_test))
            d['sp_test'].append(sp[knee])
            d['pd_test'].append(pd[knee])
            d['pf_test'].append(fa[knee])
            d['thr_test'].append(thr[knee])
            d['roc_sp_test'].append(sp.tolist())  #
            d['roc_pd_test'].append(pd.tolist())  #
            d['roc_pf_test'].append(fa.tolist())  #
            d['roc_thr_test'].append(thr.tolist())  #

            size_sig_test = np.max(np.argwhere(y[test] == 1)) + 1
            size_bkg_test = np.size(y[test]) - size_sig_test

            d['size_sig_test'].append(size_sig_test)
            d['size_bkg_test'].append(size_bkg_test)

            d['init'].append(init)
            init = init + 1

            if init > numInit - 1:
                init = 0

    for key in d:
        if isinstance(d[key], list) and key != 'model':  # Exclude the 'model' key as it contains JSON strings
            try:
                d[key] = np.array(d[key], dtype=object)
            except ValueError:
                print(f"Warning: Could not convert '{key}' to NumPy array due to inconsistent shapes.")

    return d


def dumpModelH5(models, history, skf, x, y, seed_cv, numInit, output_file):
    init = 0
    with h5py.File(output_file, 'w') as f:
        for i, (train, test) in enumerate(skf.split(x, y)):
            for model_idx, model in enumerate(models):
                print(f"Dumping Fold {i}, Init {init}")
                group_name = f"fold_{i}_init_{init}"
                grp = f.create_group(group_name)

                # Predict
                y_pred_train = model.predict(x[train], batch_size=10240)
                y_pred_test = model.predict(x[test], batch_size=10240)

                # Save predictions
                grp.create_dataset('output_train', data=y_pred_train)
                grp.create_dataset('output_test', data=y_pred_test)

                # Save MSE
                grp.create_dataset('mse_train', data=mean_squared_error(y[train], y_pred_train))
                grp.create_dataset('mse_test', data=mean_squared_error(y[test], y_pred_test))

                # ROC + SP for train
                fa, pd, thr = roc_curve(y[train], y_pred_train)
                sp = np.sqrt(np.sqrt(pd * (1 - fa)) * (0.5 * (pd + (1 - fa))))
                knee = np.argmax(sp)

                grp.create_dataset('roc_pd_train', data=pd)
                grp.create_dataset('roc_pf_train', data=fa)
                grp.create_dataset('roc_thr_train', data=thr)
                grp.create_dataset('roc_sp_train', data=sp)
                grp.create_dataset('pd_train', data=pd[knee])
                grp.create_dataset('pf_train', data=fa[knee])
                grp.create_dataset('thr_train', data=thr[knee])
                grp.create_dataset('sp_train', data=sp[knee])

                # ROC + SP for test
                fa, pd, thr = roc_curve(y[test], y_pred_test)
                sp = np.sqrt(np.sqrt(pd * (1 - fa)) * (0.5 * (pd + (1 - fa))))
                knee = np.argmax(sp)

                grp.create_dataset('roc_pd_test', data=pd)
                grp.create_dataset('roc_pf_test', data=fa)
                grp.create_dataset('roc_thr_test', data=thr)
                grp.create_dataset('roc_sp_test', data=sp)
                grp.create_dataset('pd_test', data=pd[knee])
                grp.create_dataset('pf_test', data=fa[knee])
                grp.create_dataset('thr_test', data=thr[knee])
                grp.create_dataset('sp_test', data=sp[knee])

                # Sizes
                size_sig_test = np.max(np.argwhere(y[test] == 1)) + 1
                size_bkg_test = np.size(y[test]) - size_sig_test
                grp.create_dataset('size_sig_test', data=size_sig_test)
                grp.create_dataset('size_bkg_test', data=size_bkg_test)

                # Save fold/init/seed
                grp.attrs['fold'] = i
                grp.attrs['init'] = init
                grp.attrs['seed_cv'] = seed_cv

                # Save model as JSON string
                grp.attrs['model'] = model.to_json()

                # Save history
                hist = history[model_idx + i * len(models)].history
                hist_grp = grp.create_group('history')
                for key, val in hist.items():
                    hist_grp.create_dataset(key, data=val)

                init += 1
                if init >= numInit:
                    init = 0

