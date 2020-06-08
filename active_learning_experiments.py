import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import sklearn as skl
import torch as torch
import tensorflow as tf

# import data loaders and callbacks
from regression_experiments import train_and_eval_uci_data

# import Detlefsen baseline model
sys.path.append(os.path.join(os.getcwd(), 'john-master'))
from experiment_active_learning import detlefsen_uci_baseline

# import our regression models
from regression_models import GammaNormalRegression, LogNormalNormalRegression


def update_training_set(x_train, y_train, x_pool, y_pool, var, num_to_add):
    i_sort = np.argsort(var)
    x_train = np.concatenate((x_train, x_pool[i_sort[-num_to_add:]]), axis=0)
    y_train = np.concatenate((y_train, y_pool[i_sort[-num_to_add:]]), axis=0)
    x_pool = x_pool[i_sort[:-num_to_add]]
    y_pool = y_pool[i_sort[:-num_to_add]]
    return x_train, y_train, x_pool, y_pool


def run_uci_experiments(data_set_name, resume):
    assert isinstance(resume, bool)

    # load results if we are append mode and they exist
    results = os.path.join('results', 'active_learning_uci_' + data_set_name + '.pkl')
    if resume and os.path.exists(results):
        logger = pd.read_pickle(results)
        t_start = max(logger.index)
        print('Resuming at trial {:d}'.format(t_start + 2))

    # otherwise, initialize the logger
    else:
        logger = pd.DataFrame(columns=['Algorithm', 'Prior', 'Percent', 'LL', 'MAE', 'RMSE'])
        t_start = -1

    # common configurations
    batch_size = 512
    detlefsen_iterations = int(20e4)
    n_trials, d_hidden = (5, 100) if data_set_name in {'protein', 'year'} else (10, 50)
    n_al_steps = 10

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
        x_train, x_test, y_train, y_test = skl.model_selection.train_test_split(x, y, test_size=0.2)
        x_train, x_pool, y_train, y_pool = skl.model_selection.train_test_split(x, y, test_size=0.75)
        num_to_add = round(x_pool.shape[0] * 0.05)

        # run baseline
        xt, yt, xp, yp = x_train, y_train, x_pool, y_pool
        for i in range(n_al_steps):
            print('----- AL Step {:d}/{:d}:'.format(i + 1, n_al_steps), '-----')

            # configure, train, and evaluate model
            scale = skl.preprocessing.StandardScaler().fit(xt)
            iterations = round(detlefsen_iterations * xt.shape[0] / (xt.shape[0] + xp.shape[0]))
            ll, mae, rmse, var = detlefsen_uci_baseline(x_train=scale.transform(xt), y_train=yt,
                                                        x_pool=scale.transform(xp), y_pool=yp,
                                                        x_test=scale.transform(x_test), y_test=y_test,
                                                        iterations=iterations, batch_size=batch_size)
            print('LL Estimate:', ll, ', MAE:', mae, ', RMSE:', rmse)

            # update data frame
            new_df = pd.DataFrame({'Algorithm': 'Detlefsen', 'Prior': 'N/A',
                                   'Percent': xt.shape[0] / (xt.shape[0] + xp.shape[0]),
                                   'LL': ll, 'MAE': mae, 'RMSE': rmse}, index=[t])
            logger = logger.append(new_df)

            # add highest variance points to training set
            var = np.prod(var, axis=-1)
            xt, yt, xp, yp = update_training_set(xt, yt, xp, yp, var, num_to_add)

        # loop over the models and prior configurations
        for model in [GammaNormalRegression]: #, LogNormalNormalRegression]:
            for prior in ['mle', 'standard', 'vamp', 'vamp_trainable', 'vbem']:
                model_name = 'Gamma-Normal' if model == GammaNormalRegression else 'LogNormal-Normal'
                print('\n***** Trial {:d}/{:d}:'.format(t + 1, n_trials), model_name + ':', prior, '*****')

                # initialize training/pool splits
                xt, yt, xp, yp = x_train, y_train, x_pool, y_pool

                # set learning rate according to prior
                if prior == 'mle':
                    learning_rate = 1e-4
                elif prior == 'standard':
                    learning_rate = 1e-3
                elif 'vbem' in prior:
                    learning_rate = 5e-4
                else:  # vamp
                    learning_rate = 1e-4

                # loop over the active learning steps
                for i in range(n_al_steps):
                    print('----- AL Step {:d}/{:d}:'.format(i + 1, n_al_steps), '-----')

                    # create TF data loaders
                    scale = skl.preprocessing.StandardScaler().fit(xt)
                    ds_train = tf.data.Dataset.from_tensor_slices({'x': scale.transform(xt), 'y': yt})
                    ds_train = ds_train.shuffle(10000, reshuffle_each_iteration=True).batch(batch_size)
                    ds_test = tf.data.Dataset.from_tensor_slices({'x': scale.transform(x_test), 'y': y_test})
                    ds_test = ds_test.shuffle(10000, reshuffle_each_iteration=True).batch(batch_size)

                    # ensure number of epochs corresponds to the number Detlefsen batch iterations
                    n_samples = xt.shape[0] + xp.shape[0]
                    epochs = detlefsen_iterations / int(np.ceil(n_samples / batch_size))
                    epochs = round(epochs * xt.shape[0] / n_samples)

                    # configure, train, and evaluate model
                    k = round(100 * xt.shape[0] / (xt.shape[0] + xp.shape[0]))
                    mdl = model(d_in=x_train.shape[1],
                                d_hidden=d_hidden,
                                f_hidden='relu',
                                d_out=y_train.shape[1],
                                prior=prior,
                                y_mean=np.mean(yt),
                                y_var=np.var(yt),
                                a=1.0 if model == GammaNormalRegression else -np.log(2) / 2,
                                b=1.0 if model == GammaNormalRegression else np.sqrt(np.log(2)),
                                k=k,
                                u=scale.transform(xt[np.random.choice(xt.shape[0], k, replace=False)]),
                                n_mc=20)
                    ll, mae, rmse, mdl = train_and_eval_uci_data(mdl, learning_rate, epochs, ds_train, ds_test)

                    # update data frame
                    new_df = pd.DataFrame({'Algorithm': mdl.type, 'Prior': prior,
                                           'Percent': xt.shape[0] / (xt.shape[0] + xp.shape[0]),
                                           'LL': ll, 'MAE': mae, 'RMSE': rmse}, index=[t])
                    logger = logger.append(new_df)

                    # add highest variance points to training set
                    mdl.num_mc_samples = 2000
                    var = np.prod(mdl.posterior_predictive_std(scale.transform(xp)) ** 2, axis=-1)
                    xt, yt, xp, yp = update_training_set(xt, yt, xp, yp, var, num_to_add)

        # save results after each trial
        logger.to_pickle(os.path.join('results', 'active_learning_uci_' + data_set_name + '.pkl'))

    print('\nDone!')


if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='boston', help='UCI set name')
    parser.add_argument('--resume', type=int, default=0, help='resumes where we left off')
    args = parser.parse_args()

    # check inputs
    assert args.data in {'toy'}.union(set(os.listdir('data')))
    assert isinstance(args.resume, int)

    # make result directory if it doesn't already exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # run experiments accordingly
    run_uci_experiments(data_set_name=args.data, resume=bool(args.resume))
