import os
import sys
import torch
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import sklearn as skl
import tensorflow as tf

from scipy.stats import gamma
from callbacks import RegressionCallback
from regression_data import generate_toy_data
from regression_models import GammaNormalRegression


class MeanVarianceLogger(object):
    def __init__(self):
        self.cols_data = ['Algorithm', 'Prior', 'x', 'y']
        self.df_data = pd.DataFrame(columns=self.cols_data)
        self.cols_eval = ['Algorithm', 'Prior', 'x', 'mean(y|x)', 'std(y|x)']
        self.df_eval = pd.DataFrame(columns=self.cols_eval)

    @staticmethod
    def __to_list(val):
        if isinstance(val, tf.Tensor):
            val = val.numpy()
        assert isinstance(val, np.ndarray)
        val = np.squeeze(val)
        return val.tolist()

    def update(self, algorithm, prior, x_train, y_train, x_eval, mean, std, trial):

        # update training points data frame
        algorithm_list = [algorithm] * len(x_train)
        prior_list = [prior] * len(x_train)
        x_train = self.__to_list(x_train)
        y_train = self.__to_list(y_train)
        df_new = pd.DataFrame(dict(zip(self.cols_data, (algorithm_list, prior_list, x_train, y_train))),
                              index=[trial] * len(x_train))
        self.df_data = self.df_data.append(df_new)

        # update evaluation data frame
        algorithm_list = [algorithm] * len(x_eval)
        prior_list = [prior] * len(x_eval)
        x_eval = self.__to_list(x_eval)
        mean = self.__to_list(mean)
        std = self.__to_list(std)
        df_new = pd.DataFrame(dict(zip(self.cols_eval, (algorithm_list, prior_list, x_eval, mean, std))),
                              index=[trial] * len(x_eval))
        self.df_eval = self.df_eval.append(df_new)


def train_and_eval_toy_data(mdl, learning_rate, x_train, y_train, epochs, x_eval):

    # construct TF data set
    ds_train = tf.data.Dataset.from_tensor_slices({'x': x_train, 'y': y_train}).batch(x_train.shape[0])

    # compile and train the model
    mdl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=[None])
    mdl.fit(ds_train, epochs=epochs, verbose=0, callbacks=[RegressionCallback(epochs)])

    # evaluate the model
    mdl.num_mc_samples = 2000
    ll_exact = mdl.posterior_predictive_log_likelihood(x_train, y_train, exact=True).numpy()
    ll_estimate = mdl.posterior_predictive_log_likelihood(x_train, y_train, exact=False).numpy()
    mdl_mean, mdl_std = mdl.posterior_predictive_mean(x_eval).numpy(), mdl.posterior_predictive_std(x_eval).numpy()

    # print update
    print('LL Exact:', ll_exact, ', LL Estimate:', ll_estimate)

    return ll_exact, mdl_mean, mdl_std


def run_toy_experiments():
    # set common configurations for our models (these all match Detlefsen)
    d_hidden = 50
    learning_rate = 1e-2
    epochs = int(6e3)
    n_trials = 20

    # initialize result loggers
    ll_logger = pd.DataFrame(columns=['Algorithm', 'Prior', 'LL'])
    mv_logger = MeanVarianceLogger()

    # loop over the number of trials
    for t in range(n_trials):

        # generate toy data
        x_train, y_train, x_eval, true_mean, true_std = generate_toy_data()
        mv_logger.update('truth', 'N/A', x_train, y_train, x_eval, true_mean, true_std, trial=t)

        prior = 'standard'

        # run our Gamma-Normal model
        print('\n***** Trial {:d}/{:d}:'.format(t + 1, n_trials), 'Gamma-Normal:', prior, '*****')
        a, _, b_inv = gamma.fit(1 / true_std[(np.min(x_train) <= x_eval) * (x_eval <= np.max(x_train))] ** 2, floc=0)
        mdl = GammaNormalRegression(d_in=x_train.shape[1],
                                    d_hidden=d_hidden,
                                    f_hidden='sigmoid',
                                    d_out=y_train.shape[1],
                                    prior=prior,
                                    y_mean=0.0,
                                    y_var=1.0,
                                    a=a,
                                    b=1 / b_inv,
                                    k=20,
                                    n_mc=50)
        ll, mdl_mean, mdl_std = train_and_eval_toy_data(mdl, learning_rate, x_train, y_train, epochs, x_eval)
        alg = 'Gamma-Normal'
        ll_logger = ll_logger.append(pd.DataFrame({'Algorithm': alg, 'Prior': prior, 'LL': ll}, index=[t]))
        mv_logger.update(alg, prior, x_train, y_train, x_eval, mdl_mean, mdl_std, trial=t)
    
        # save results after each trial
        ll_logger.to_pickle(os.path.join('results', 'regression_toy_ll.pkl'))
        mv_logger.df_data.to_pickle(os.path.join('results', 'regression_toy_data.pkl'))
        mv_logger.df_eval.to_pickle(os.path.join('results', 'regression_toy_mean_variance.pkl'))

    print('\nDone!')

run_toy_experiments()