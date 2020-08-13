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

# import data loaders and callbacks
from regression_data import generate_toy_data
from callbacks import RegressionCallback

# import Detlefsen baseline model
sys.path.append(os.path.join(os.getcwd(), 'john-master'))
from toy_regression import detlefsen_toy_baseline
from experiment_regression import detlefsen_uci_baseline

# import our regression models
from regression_models import GammaNormalRegression, LogNormalNormalRegression

# list of priors to test
PRIORS = ['mle', 'standard', 'vamp', 'vamp_trainable', 'vbem', 'standard_alt']


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

        # run baseline w/o our bug fixes
        print('\n***** Trial {:d}/{:d}:'.format(t + 1, n_trials), 'Detlefsen (orig)', '*****')
        mdl_mean, mdl_std, ll_estimate = detlefsen_toy_baseline(x_train, y_train, x_eval, bug_fix=False)
        print('LL Estimate:', ll_estimate)
        alg, prior = 'Detlefsen (orig)', 'N/A'
        ll_logger = ll_logger.append(pd.DataFrame({'Algorithm': alg, 'Prior': prior, 'LL': ll_estimate}, index=[t]))
        mv_logger.update(alg, prior, x_train, y_train, x_eval, mdl_mean, mdl_std, trial=t)

        # run baseline w/ our bug fixes
        print('\n***** Trial {:d}/{:d}:'.format(t + 1, n_trials), 'Detlefsen (fixed)', '*****')
        mdl_mean, mdl_std, ll_estimate = detlefsen_toy_baseline(x_train, y_train, x_eval, bug_fix=True)
        print('LL Estimate:', ll_estimate)
        alg, prior = 'Detlefsen (fixed)', 'N/A'
        ll_logger = ll_logger.append(pd.DataFrame({'Algorithm': alg, 'Prior': prior, 'LL': ll_estimate}, index=[t]))
        mv_logger.update(alg, prior, x_train, y_train, x_eval, mdl_mean, mdl_std, trial=t)

        # loop over the configurations
        for prior in ['mle', 'standard', 'vamp', 'vamp_trainable', 'vbem']:

            # VAMP prior pseudo-input initializers
            u = np.expand_dims(np.linspace(np.min(x_eval), np.max(x_eval), 20), axis=-1)

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
                                        u=u,
                                        n_mc=50)
            ll, mdl_mean, mdl_std = train_and_eval_toy_data(mdl, learning_rate, x_train, y_train, epochs, x_eval)
            alg = 'Gamma-Normal'
            ll_logger = ll_logger.append(pd.DataFrame({'Algorithm': alg, 'Prior': prior, 'LL': ll}, index=[t]))
            mv_logger.update(alg, prior, x_train, y_train, x_eval, mdl_mean, mdl_std, trial=t)

            # run our LogNormal-Normal model
            print('\n***** Trial {:d}/{:d}:'.format(t + 1, n_trials), 'LogNormal-Normal:', prior, '*****')
            precisions = 1 / true_std[(np.min(x_train) <= x_eval) * (x_eval <= np.max(x_train))] ** 2
            a, b = np.mean(np.log(precisions)), np.std(np.log(precisions))
            mdl = LogNormalNormalRegression(d_in=x_train.shape[1],
                                            d_hidden=d_hidden,
                                            f_hidden='sigmoid',
                                            d_out=y_train.shape[1],
                                            prior=prior,
                                            y_mean=0.0,
                                            y_var=1.0,
                                            a=a,
                                            b=b,
                                            k=20,
                                            u=u,
                                            n_mc=50)
            ll, mdl_mean, mdl_std = train_and_eval_toy_data(mdl, learning_rate, x_train, y_train, epochs, x_eval)
            alg = 'LogNormal-Normal'
            ll_logger = ll_logger.append(pd.DataFrame({'Algorithm': alg, 'Prior': prior, 'LL': ll}, index=[t]))
            mv_logger.update(alg, prior, x_train, y_train, x_eval, mdl_mean, mdl_std, trial=t)

        # save results after each trial
        ll_logger.to_pickle(os.path.join('results', 'regression_toy_ll.pkl'))
        mv_logger.df_data.to_pickle(os.path.join('results', 'regression_toy_data.pkl'))
        mv_logger.df_eval.to_pickle(os.path.join('results', 'regression_toy_mean_variance.pkl'))

    print('\nDone!')


def train_and_eval_uci_data(mdl, learning_rate, epochs, ds_train, ds_test):
    # compile and train the model
    mdl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=[None])
    hist = mdl.fit(ds_train, validation_data=ds_test, epochs=epochs, verbose=0, callbacks=[
        RegressionCallback(epochs),
        tf.keras.callbacks.EarlyStopping(monitor='val_LL', min_delta=1e-4, patience=500, mode='max')])

    # get index of best validation log likelihood
    i_best = np.argmax(hist.history['val_LL'])
    if i_best != np.argmax(hist.history['val_LL (adjusted)']):
        delta = np.abs(np.max(hist.history['val_LL (adjusted)']) - hist.history['val_LL (adjusted)'][i_best])
        warnings.warn('LL != LL adjusted by {:f}'.format(delta))
    if np.argmax(hist.history['val_LL (adjusted)']) >= 0.9 * epochs:
        warnings.warn('LL not converged!')

    # get test set results
    ll = hist.history['val_LL (adjusted)'][i_best]
    mae = hist.history['val_MAE'][i_best]
    rmse = np.sqrt(hist.history['val_MSE'][i_best])

    # print update
    print('LL Exact:', ll, ', MAE:', mae, ', RMSE:', rmse)

    return ll, mae, rmse, mdl


def run_uci_experiments(data_set_name, mode, resume, augment):
    assert mode in {'10000', 'full'}
    assert isinstance(resume, bool)
    assert isinstance(augment, bool)
    assert not (resume and augment)

    # load results if we are append mode and they exist
    results = os.path.join('results', 'regression_uci_' + data_set_name + '.pkl')
    if (resume or augment) and os.path.exists(results):
        logger = pd.read_pickle(results)
        if resume:
            t_start = max(logger.index)
            priors = PRIORS
            print('Resuming at trial {:d}'.format(t_start + 2))
        else:
            t_start = -1
            priors = [p for p in PRIORS if p not in logger.Prior.unique()]
            print('Augmenting missing priors: ' + ', '.join(priors))

    # otherwise, initialize the logger
    else:
        logger = pd.DataFrame(columns=['Algorithm', 'Prior', 'LL', 'MAE', 'RMSE'])
        t_start = -1
        priors = PRIORS

    # common configurations
    batch_size = 512
    detlefsen_iterations = int(20e4) if mode == 'full' else int(10e3)
    n_trials, d_hidden = (5, 100) if data_set_name in {'protein', 'year'} else (20, 50)

    # load data
    with open(os.path.join('data', data_set_name, data_set_name + '.pkl'), 'rb') as f:
        data_dict = pickle.load(f)
    x, y = data_dict['data'], data_dict['target']

    # loop over the trials
    for t in range(t_start + 1, n_trials):

        # set random number seeds
        np.random.seed(t)
        tf.random.set_seed(t)
        torch.manual_seed(t)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # split the data
        x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(x, y, test_size=0.1)

        # scale x
        scaler_x = skl.preprocessing.StandardScaler().fit(x_train)
        x_train = scaler_x.transform(x_train)
        x_test = scaler_x.transform(x_test)

        # run baseline
        if not augment:
            ll, mae, rmse = detlefsen_uci_baseline(x_train, y_train, x_test, y_test, detlefsen_iterations, batch_size)
            print('LL Estimate:', ll, ', MAE:', mae, ', RMSE:', rmse)
            new_df = pd.DataFrame({'Algorithm': 'Detlefsen', 'Prior': 'N/A', 'LL': ll, 'MAE': mae, 'RMSE': rmse}, index=[t])
            logger = logger.append(new_df)

        # create TF data loaders
        ds_train = tf.data.Dataset.from_tensor_slices({'x': x_train, 'y': y_train})
        ds_train = ds_train.shuffle(10000, reshuffle_each_iteration=True).batch(batch_size)
        ds_test = tf.data.Dataset.from_tensor_slices({'x': x_test, 'y': y_test})
        ds_test = ds_test.shuffle(10000, reshuffle_each_iteration=True).batch(batch_size)

        # get pseudo-input initializer
        u_init = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]

        # loop over the configurations
        for model in [GammaNormalRegression]: #, LogNormalNormalRegression]:
            for prior in priors:

                # set learning rate according to prior
                if prior == 'mle':
                    learning_rate = 1e-4
                elif 'standard' in prior:
                    learning_rate = 1e-3
                elif prior == 'vbem':
                    learning_rate = 5e-4
                else:  # vamp
                    learning_rate = 1e-4

                # ensure number of epochs corresponds to the number Detlefsen batch iterations
                epochs = round(detlefsen_iterations / int(np.ceil(x_train.shape[0] / batch_size)))

                if model == GammaNormalRegression:
                    a = 1.0
                    b = 0.1 if prior == 'standard_alt' else 1.0
                else:
                    a = -np.log(2) / 2
                    b = np.sqrt(np.log(2))

                # configure model
                mdl = model(d_in=x_train.shape[1],
                            d_hidden=d_hidden,
                            f_hidden='relu',
                            d_out=y_train.shape[1],
                            prior='standard' if prior == 'standard_alt' else prior,
                            y_mean=np.mean(y_train),
                            y_var=np.var(y_train),
                            a=a,
                            b=b,
                            k=100,
                            u=u_init,
                            n_mc=20)

                # train and evaluate
                print('\n***** Trial {:d}/{:d}:'.format(t + 1, n_trials), mdl.type + ':', prior, '*****')
                ll, mae, rmse, _ = train_and_eval_uci_data(mdl, learning_rate, epochs, ds_train, ds_test)

                # update results
                new_df = pd.DataFrame({'Algorithm': mdl.type, 'Prior': prior, 'LL': ll, 'MAE': mae, 'RMSE': rmse},
                                      index=[t])
                logger = logger.append(new_df)

        # save results after each trial
        post_fix = '_10000' if mode == '10000' else ''
        logger.to_pickle(os.path.join('results', 'regression_uci_' + data_set_name + post_fix + '.pkl'))

    print('\nDone!')


if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='boston', help='data set name = {toy} union UCI sets')
    parser.add_argument('--mode', type=str, default='full', help='mode in {10000, full}')
    parser.add_argument('--resume', type=int, default=0, help='resumes where we left off')
    parser.add_argument('--augment', type=int, default=1, help='adds missing priors to results')
    args = parser.parse_args()

    # check inputs
    assert args.data in {'toy'}.union(set(os.listdir('data')))
    assert isinstance(args.resume, int)
    assert isinstance(args.augment, int)

    # make result directory if it doesn't already exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # run experiments accordingly
    if args.data == 'toy':
        run_toy_experiments()
    else:
        run_uci_experiments(data_set_name=args.data, mode=args.mode, resume=bool(args.resume), augment=bool(args.augment))
