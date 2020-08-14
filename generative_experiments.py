import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import torch as torch
import tensorflow as tf

from generative_data import load_data_set
from generative_models import NormalVAE, StudentVAE, VariationalVarianceVAE, precision_prior_params

# import Detlefsen baseline model
sys.path.append(os.path.join(os.getcwd(), 'john-master'))
from experiment_vae import detlefsen_vae_baseline


# dictionary of methods to test
METHODS = [
    {'Method': 'VAE', 'mdl': NormalVAE, 'kwargs': {'split_decoder': False}},
    {'Method': 'VAE-Split', 'mdl': NormalVAE, 'kwargs': {'split_decoder': True}},
    {'Method': 'MAP-VAE', 'mdl': NormalVAE, 'kwargs': {'split_decoder': True, 'b': 1e-3}},
    {'Method': 'Student-VAE', 'mdl': StudentVAE, 'kwargs': dict()},
    {'Method': 'EB-MAP-VAE', 'mdl': NormalVAE, 'kwargs': {'split_decoder': True}},
    {'Method': 'V3AE-Uniform', 'mdl': VariationalVarianceVAE, 'kwargs': {'prior': 'mle'}},
    {'Method': 'V3AE-Gamma', 'mdl': VariationalVarianceVAE, 'kwargs': {'prior': 'standard', 'a': 1.0, 'b': 1e-3}},
    {'Method': 'V3AE-Gamma-x', 'mdl': VariationalVarianceVAE, 'kwargs': {'prior': 'standard', 'a': 1.0, 'b': 1e-3, 'qp_dependence': 'x'}},
    {'Method': 'EB-V3AE-Gamma', 'mdl': VariationalVarianceVAE, 'kwargs': {'prior': 'standard'}},
    {'Method': 'V3AE-VAMP', 'mdl': VariationalVarianceVAE, 'kwargs': {'prior': 'vamp', 'a': 1.0, 'b': 1e-3}},
    {'Method': 'V3AE-VAMP-x', 'mdl': VariationalVarianceVAE, 'kwargs': {'prior': 'vamp', 'a': 1.0, 'b': 1e-3, 'qp_dependence': 'x'}},
    {'Method': 'V3AE-VBEM', 'mdl': VariationalVarianceVAE, 'kwargs': {'prior': 'vbem', 'a': 1.0, 'b': 1e-3, 'k': 10}},
    {'Method': 'V3AE-VBEM-x', 'mdl': VariationalVarianceVAE, 'kwargs': {'prior': 'vbem', 'a': 1.0, 'b': 1e-3, 'k': 10, 'qp_dependence': 'x'}},
]

# latent dimension per data set
DIM_Z = {'mnist': 10, 'fashion_mnist': 25, 'svhn_cropped': 50}
ARCHITECTURE = {'mnist': 'dense', 'fashion_mnist': 'dense', 'svhn_cropped': 'convolution'}
BATCH_NORM = {'mnist': [False, True], 'fashion_mnist': [False, True], 'svhn_cropped': [False]}


def run_vae_experiments(data_set_name, resume, augment):
    assert data_set_name in DIM_Z.keys()
    assert isinstance(resume, bool)
    assert isinstance(augment, bool)
    assert not (resume and augment)

    # load results if we are append mode and they exist
    results = os.path.join('results', 'generative_' + data_set_name + '_metrics.pkl')
    plots = os.path.join('results', 'generative_' + data_set_name + '_plots.pkl')
    if (resume or augment) and os.path.exists(results) and os.path.exists(plots):
        logger = pd.read_pickle(results)
        with open(plots, 'rb') as f:
            plotter = pickle.load(f)
        if resume:
            t_start = max(logger.index)
            methods = METHODS
            print('Resuming at trial {:d}'.format(t_start + 2))
        else:
            t_start = -1
            methods = [m for m in METHODS if m['Method'] not in logger.Method.unique()]
            for m in methods:
                plotter.update({m['Method']: [{'learning': [], 'reconstruction': []},
                                              {'learning': [], 'reconstruction': []}]})
            print('Augmenting missing methods: ' + ', '.join([m['Method'] for m in methods]))

    # otherwise, initialize the logger
    else:
        logger = pd.DataFrame(columns=['Method', 'BatchNorm', 'LL', 'RMSE', 'Entropy'])
        plotter = {'x': None}
        for m in METHODS:
            plotter.update({m['Method']: [{'learning': [], 'reconstruction': []},
                                          {'learning': [], 'reconstruction': []}]})
        t_start = -1
        methods = METHODS

    # common configurations
    n_trials = 5
    batch_size = 250
    epochs = 1000

    # load data
    train_set, test_set, info = load_data_set(data_set_name=data_set_name, px_family='Normal', batch_size=batch_size)

    # loop over the trials
    for t in range(t_start + 1, n_trials):

        # set random number seeds
        np.random.seed(t)
        tf.random.set_seed(t)
        torch.manual_seed(t)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # get prior parameters for precision
        a, b, u = precision_prior_params(data=train_set,
                                         num_classes=info.features['label'].num_classes,
                                         pseudo_inputs_per_class=10)

        # get sub-set of test set for results plotting
        if plotter['x'] is None:
            plotter['x'] = precision_prior_params(data=test_set,
                                                  num_classes=info.features['label'].num_classes,
                                                  pseudo_inputs_per_class=10)[-1]

        # run Detlefsen baseline
        x_train = np.concatenate([x['image'] for x in train_set.as_numpy_iterator()], axis=0)
        x_test = np.concatenate([x['image'] for x in test_set.as_numpy_iterator()], axis=0)
        # ll, rmse, h, x_mean, x_std, x_new = detlefsen_vae_baseline(x_train, x_test, plotter['x'],
        #                                                            DIM_Z[data_set_name], epochs, batch_size)

        # loop over the configurations
        for m in methods:
            for batch_norm in BATCH_NORM[data_set_name]:
                status = '(batch norm)' if batch_norm else ''
                print('\n***** Trial {:d}/{:d}:'.format(t + 1, n_trials), m['Method'], status, '*****')

                # update kwargs accordingly
                kwargs = m['kwargs']
                kwargs.update({'dim_x': info.features['image'].shape, 'dim_z': DIM_Z[data_set_name],
                               'architecture': ARCHITECTURE[data_set_name], 'batch_norm': batch_norm,
                               'latex_metrics': False})
                if 'EB-' in m['Method']:
                    kwargs.update({'a': a, 'b': b})
                if 'VAMP' in m['Method']:
                    kwargs.update({'u': u})

                # configure and compile
                mdl = m['mdl'](**kwargs)
                mdl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=[None])

                # train
                hist = mdl.fit(train_set, validation_data=test_set, epochs=epochs, verbose=1,
                               validation_steps=np.ceil(info.splits['test'].num_examples // batch_size),
                               callbacks=[tf.keras.callbacks.TerminateOnNaN(),
                                          tf.keras.callbacks.EarlyStopping(monitor='val_LL',
                                                                           patience=50,
                                                                           mode='max',
                                                                           restore_best_weights=True)])

                # log scalar metrics
                ll = []
                x_new = []
                h = []
                for i in range(int(np.ceil(x_test.shape[0] / batch_size))):
                    i_start = i * batch_size
                    i_end = min((i + 1) * batch_size, x_test.shape[0])
                    ll.append(mdl.variational_objective(x_test[i_start:i_end])[1])
                    _, _, x_new_batch, h_batch = mdl.posterior_predictive(x=x_test[i_start:i_end])
                    x_new.append(x_new_batch)
                    h.append(h_batch)
                ll = np.mean(tf.concat(ll, axis=0))
                rmse = np.sqrt(np.mean((tf.concat(x_new, axis=0) - x_test) ** 2))
                h = np.mean(tf.concat(h, axis=0))
            print('LL = {:.2f}, RMSE = {:.4f}, H = {:.2f}'.format(ll, rmse, h))

            # get plot data
            x_mean, x_std, x_new, _ = mdl.posterior_predictive(x=plotter['x'])

                # update results
                new_df = pd.DataFrame({'Method': m['Method'], 'BatchNorm': batch_norm,
                                       'LL': ll, 'RMSE': rmse, 'Entropy': h}, index=[t])
                logger = logger.append(new_df)
                plotter[m['Method']][batch_norm]['learning'].append(hist.history)
                plotter[m['Method']][batch_norm]['reconstruction'].append({'mean': x_mean, 'std': x_std, 'sample': x_new})

        # save results after each trial
        logger.to_pickle(results)
        with open(plots, 'wb') as f:
            pickle.dump(plotter, f)

    print('\nDone!')


if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='mnist', help='https://www.tensorflow.org/datasets/catalog/overview')
    parser.add_argument('--resume', type=int, default=0, help='resumes where we left off')
    parser.add_argument('--augment', type=int, default=0, help='adds missing methods to results')
    args = parser.parse_args()

    # check inputs
    assert isinstance(args.resume, int)
    assert isinstance(args.augment, int)

    # make result directory if it doesn't already exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # run experiments accordingly
    run_vae_experiments(data_set_name=args.data, resume=bool(args.resume), augment=bool(args.augment))
