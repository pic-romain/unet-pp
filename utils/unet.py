# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #    

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

tfd = tfp.distributions
from keras import layers
import numpy as np

from typing import Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ---------------------------------------------------------------------------- #
#                                   FUNCTIONS                                  #
# ---------------------------------------------------------------------------- #

def ConvBatchRelu(x, filters=32, kernel_size=3, strides=1, padding="same", activation="relu", separable=False):
    if not separable:
        x = layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding=padding
        )(x)
    else:
        x = layers.SeparableConv2D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding=padding
        )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x


def UNet2D_Groenquist(
    img_size, num_params, in_channels, filters_list=[64, 128], separable=True):
    """U-Net architecture inspired by Grönquist et al. (2021)"""

    inputs = keras.Input(shape=img_size + (in_channels,))
    # Storing the residuals
    res_list = []

    # -------------- First half of the network: downsampling inputs -------------- #

    # Entry block
    x = ConvBatchRelu(
        inputs,
        filters=32,
        kernel_size=3,
        strides=1,
        padding="same",
        separable=separable,
    )
    x = ConvBatchRelu(
        x, filters=32, kernel_size=3, strides=1, padding="same", separable=separable
    )
    # Set aside residual
    res_list.append(x)
    for filters in filters_list:
        x = layers.MaxPool2D(pool_size=2)(x)
        x = ConvBatchRelu(
            x,
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            separable=separable,
        )
        x = ConvBatchRelu(
            x,
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            separable=separable,
        )
        # Set aside residual
        if filters != filters_list[-1]:
            res_list.append(x)

   # --------------- Second half of the network: upsampling inputs -------------- #
    p = 0
    for filters in filters_list[::-1] + [32]:
        if p != 0:
            x = layers.Concatenate(axis=-1)([x, res_list[-p]])
        x = ConvBatchRelu(
            x,
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            separable=separable,
        )
        x = ConvBatchRelu(
            x,
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            separable=separable,
        )
        if filters != 32:  # ie the last layer
            x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        p += 1
        outputs = layers.Conv2D(num_params, (3, 3), activation=None, padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


def rescale_data(X_train, X_test, type="standard"):
    d1, d2 = X_train.shape[1], X_train.shape[2]
    X_train_tp = X_train.reshape((X_train.shape[0], d1 * d2, X_train.shape[-1]))
    X_test_tp = X_test.reshape((X_test.shape[0], d1 * d2, X_test.shape[-1]))

    # Rescaling
    scalers = {}
    for i in range(X_train_tp.shape[-1]):
        if type == "standard":
            scalers[i] = StandardScaler()
        elif type == "minmax":
            scalers[i] = MinMaxScaler()
        else:
            raise ValueError(f"{type} scaler not available")
        X_train_tp[:, :, i] = scalers[i].fit_transform(X_train_tp[:, :, i])

    for i in range(X_test.shape[-1]):
        X_test_tp[:, :, i] = scalers[i].transform(
            X_test_tp[:, :, i]
        )  # Using the same scaler as for the training set

    X_train = X_train_tp.reshape((X_train.shape[0], d1, d2, X_train.shape[-1]))
    X_test = X_test_tp.reshape((X_test.shape[0], d1, d2, X_test.shape[-1]))

    return X_train, X_test


# ------------------------------- SCORING RULES ------------------------------ #

_normal_dist = tfd.Normal(loc=0.0, scale=1.0)
_frac_sqrt_pi = 1 / np.sqrt(np.pi)
_sqrt_2 = np.sqrt(2.0)

@tf.function
def CRPS_gtcnd(rho_L: tf.Tensor,
               mu: tf.Tensor,
               rho_sigma: tf.Tensor,
               y: tf.Tensor,
               mean: bool = True,
               eps: Union[int, float] = 1e-12,
               fit: bool = True):
    """
    CRPS of the Generalized Truncated-Censored Normal distribution from Jordan et al. (2019)
    """
    # Transformation of parameters
    L = tf.math.sigmoid(tf.squeeze(rho_L))
    mu = tf.squeeze(mu)
    sigma = tf.math.exp(tf.squeeze(rho_sigma))

    # Intermediate variables
    z = tf.where(y <= 0.0, tf.zeros_like(y), y)
    w = (z - mu + eps) / (sigma + eps)
    cdf_l = _normal_dist.cdf((-mu + eps) / (sigma + eps))
    cdf_w = _normal_dist.cdf(w)
    pdf_l = tf.exp(_normal_dist.log_prob((-mu + eps) / (sigma + eps)))
    pdf_w = tf.exp(_normal_dist.log_prob(w))

    # CRPS
    CRPS1 = tf.abs((y - z + eps) / (sigma + eps))
    CRPS2 = (mu + eps) / (sigma + eps) * tf.square(L)
    T1 = (1 - L + eps) / (1 - cdf_l + eps)
    CRPS3 = T1 * w * (2 * cdf_w - (1 - 2 * L + cdf_l + eps) / (1 - L + eps))
    CRPS4 = T1 * 2 * (pdf_w - pdf_l * L)
    CRPS5 = - tf.square(T1) * _frac_sqrt_pi * (1 - _normal_dist.cdf((-mu + eps) * _sqrt_2 / (sigma + eps)))
    CRPS = sigma * (CRPS1 + CRPS2 + CRPS3 + CRPS4 + CRPS5)

    # Avoid NaN or Infinity by replacing them with a large value
    CRPS = tf.where(tf.math.is_nan(CRPS) | tf.math.is_inf(CRPS), tf.ones_like(CRPS) * 1e6, CRPS)

    if fit:
        return tf.reduce_mean(CRPS)
    elif mean:
        return tf.reduce_mean(CRPS, axis=0)
    else:
        return CRPS


@tf.function
def CRPS_csgd(
    rho_k: tf.Tensor,
    rho_theta: tf.Tensor,
    rho_delta: tf.Tensor,
    y: tf.Tensor,
    mean: bool = True,
    eps: Union[int, float] = 1e-12,
    fit: bool = True):
    """
    CRPS of the Censored-Shifted Gamma distribution from Scheuerer & Hamill (2015)
    """
    # Transformation of parameters
    k = tf.math.exp(tf.squeeze(rho_k))
    theta = tf.math.exp(tf.squeeze(rho_theta))
    delta = - k*theta * tf.math.sigmoid(tf.squeeze(rho_delta))

    # Distributions
    _gamma_dist_k = tfd.Gamma(concentration=k, rate=tf.ones_like(k))
    _gamma_dist_kp1 = tfd.Gamma(concentration=k+1, rate=tf.ones_like(k))
    _gamma_dist_2k = tfd.Gamma(concentration=2*k, rate=tf.ones_like(k))

    # Intermediate variables
    c_tilde = -(delta + eps) / (theta + eps)
    y_tilde = (y - delta + eps) / (theta + eps)
    cdf_k_ytilde = _gamma_dist_k.cdf(y_tilde)
    cdf_k_ctilde = _gamma_dist_k.cdf(c_tilde)
    cdf_kp1_ytilde = _gamma_dist_kp1.cdf(y_tilde)
    cdf_kp1_ctilde = _gamma_dist_kp1.cdf(c_tilde)
    cdf_2k_2ctilde = _gamma_dist_2k.cdf(2*c_tilde)
    gamma_half = tf.math.exp(tf.math.lgamma(tf.ones_like(k) / 2))
    gamma_kphalf = tf.math.exp(tf.math.lgamma(k + .5))
    gamma_kp1 = tf.math.exp(tf.math.lgamma(k + 1))
    beta = gamma_half * gamma_kphalf / gamma_kp1

    # CRPS
    CRPS1 = y_tilde * (2 * cdf_k_ytilde - 1)
    CRPS2 = - c_tilde * tf.square(cdf_k_ctilde)
    CRPS3 = k * (1 + 2* cdf_k_ctilde * cdf_kp1_ctilde
                 - tf.square(cdf_k_ctilde)
                 - 2 * cdf_kp1_ytilde)
    CRPS4 = - k / np.pi * beta * (1 - cdf_2k_2ctilde)
    CRPS = theta * (CRPS1 + CRPS2 + CRPS3 + CRPS4)

    if fit:
        return tf.reduce_mean(CRPS)
    elif mean:
        return tf.reduce_mean(CRPS, axis=0)
    else:
        return CRPS


# ---------------------------------------------------------------------------- #
#                                    CLASSES                                   #
# ---------------------------------------------------------------------------- #

class NN_gtcnd_model(keras.Model):
    def __init__(self, NN, *args, **kwargs):
        super(NN_gtcnd_model, self).__init__(*args, **kwargs)
        self.NN = NN

    @tf.function
    def call(self, X):
        params_pred = self.NN(X)
        return params_pred

    def train_step(self, data):
        X, Y = data
        with tf.GradientTape() as tape:
            params_pred = self.NN(X)
            rho_L, mu, rho_sigma = tf.split(params_pred, num_or_size_splits=3, axis=-1)
            loss = self.loss(rho_L=rho_L, mu=mu, rho_sigma=rho_sigma, y=Y, fit=True)
        trainable_vars = self.NN.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {"loss": loss}

    def test_step(self, data):
        X, Y = data
        params_pred = self.NN(X)
        rho_L, mu, rho_sigma = tf.split(params_pred, num_or_size_splits=3, axis=-1)
        loss = self.loss(rho_L=rho_L, mu=mu, rho_sigma=rho_sigma, y=Y, fit=True)
        return {"loss": loss}
    
class NN_csgd_model(keras.Model):
    def __init__(self, NN, *args, **kwargs):
        super(NN_csgd_model, self).__init__(*args, **kwargs)
        self.NN = NN

    @tf.function
    def call(self, X):
        params_pred = self.NN(X)
        return params_pred

    def train_step(self, data):
        X, Y = data
        with tf.GradientTape() as tape:
            params_pred = self.NN(X)
            rho_k, rho_theta, rho_delta = tf.split(params_pred, num_or_size_splits=3, axis=-1)
            loss = self.loss(rho_k=rho_k, rho_theta=rho_theta, rho_delta=rho_delta, y=Y, fit=True)
        trainable_vars = self.NN.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {"loss": loss}

    def test_step(self, data):
        X, Y = data
        params_pred = self.NN(X)
        rho_k, rho_theta, rho_delta = tf.split(params_pred, num_or_size_splits=3, axis=-1)
        loss = self.loss(rho_k=rho_k, rho_theta=rho_theta, rho_delta=rho_delta, y=Y, fit=True)
        return {"loss": loss}
    
