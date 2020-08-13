import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind_from_stats


def raw_result_table(pickle_files, main_body):

    # aggregate raw results into a table
    raw_table = []
    for result in pickle_files:

        # load logger
        log = pd.read_pickle(result)

        # assign experiment name
        log['Data'] = result.split('generative_')[-1].split('_metrics')[0]

        # append experiment to results table
        raw_table.append(log)

    # concatenate and clean up table
    raw_table = pd.concat(raw_table)
    raw_table = raw_table.drop(['Entropy'], axis=1)
    raw_table = raw_table[raw_table.Method != 'EB-MAP-VAE']
    raw_table = raw_table[raw_table.Method != 'EB-V3AE-Gamma']
    if main_body:
        raw_table = raw_table[raw_table.BatchNorm == False]
        raw_table = raw_table.drop(['BatchNorm'], axis=1)

    return raw_table


def string_table(mean, std):
    mean['LL'] = mean['LL'].round(2).astype('str')
    std['LL'] = std['LL'].round(2).astype('str')
    mean['RMSE'] = mean['RMSE'].round(3).astype('str')
    std['RMSE'] = std['RMSE'].round(3).astype('str')
    return mean + '$\\pm$' + std


def generate_tables(pickle_files, main_body):

    # get raw results
    raw_table = raw_result_table(pickle_files, main_body)
    raw_table = raw_table.replace('V3AE-Uniform', 'V3AE-MLE')

    # aggregate processed results into a table
    table = None
    for data in raw_table['Data'].unique():

        # clean up the name
        new_name = data.replace('_', ' ')
        raw_table.loc[raw_table.Data == data, 'Data'] = new_name
        data = new_name

        # compute means and standard deviations over methods
        experiment = raw_table[raw_table.Data == data]
        experiment = experiment[experiment.Method != 'V3AE-Gamma-x']
        experiment = experiment[experiment.Method != 'V3AE-VAMP-x']
        experiment = experiment[experiment.Method != 'V3AE-VBEM-x']
        groups = ['Data', 'Method'] if main_body else ['Data', 'Method', 'BatchNorm']
        mean = pd.DataFrame(experiment.groupby(groups, sort=False).mean())
        std = pd.DataFrame(experiment.groupby(groups, sort=False).std(ddof=1))

        # build string table
        df = string_table(mean.copy(deep=True), std.copy(deep=True))

        # bold winners if sufficient trials
        n_trials = max(experiment.index) + 1
        if n_trials >= 2:

            # loop over the metrics
            for (metric, order) in [('LL', 'max'), ('RMSE', 'min')]:  #, ('Entropy', 'min')]:

                # get top performer
                i_best = np.argmax(mean[metric]) if order == 'max' else np.argmin(mean[metric])

                # get null hypothesis
                null_mean = mean[metric].to_numpy()[i_best]
                null_std = std[metric].to_numpy()[i_best]

                # compute p-values
                ms = zip([m for m in mean[metric].to_numpy().tolist()], [s for s in std[metric].to_numpy().tolist()])
                p = [ttest_ind_from_stats(null_mean, null_std, n_trials, m, s, n_trials, False)[-1] for (m, s) in ms]

                # bold statistical ties for best
                for i in range(df.shape[0]):
                    if i == i_best or p[i] >= 0.05:
                        df.loc[mean[metric].index[i], metric] = '\\textbf{' + df.loc[mean[metric].index[i], metric] + '}'

        # concatenate experiment to results table
        if main_body:
            table = pd.concat([table, df.unstack(level=0).T.swaplevel(0, 1)])
        else:
            table = pd.concat([table, df])

    return table.to_latex(escape=False)


def image_reshape(x):
    return np.reshape(tf.transpose(x, [1, 0, 2, 3]), [x.shape[1], -1, x.shape[-1]])


def generate_plots(pickle_files):

    # get raw results
    raw_table = raw_result_table(pickle_files, main_body=True)

    # loop over the experiments and methods
    for data in raw_table['Data'].unique():

        # load plot data
        with open(os.path.join('results', 'generative_' + data + '_plots.pkl'), 'rb') as f:
            plots = pickle.load(f)

        # get table and methods for this data set
        t_data = raw_table[raw_table.Data == data]
        methods = [m for m in t_data['Method'].unique() if m not in {'EB-MAP-VAE', 'EB-V3AE-Gamma'}]

        # grab original data
        x = np.squeeze(image_reshape(plots['x'][0::2]))

        # initialize figure
        fig, ax = plt.subplots(len(methods), 1, figsize=(16, 1.3 * len(methods)))
        plt.subplots_adjust(left=0.03, bottom=0.01, right=0.99, top=0.99, wspace=0.0, hspace=0.0)

        # loop over the methods for this data set
        for i, method in enumerate(methods):

            # select between the better of batch normalization being on or off
            t_method = t_data[t_data.Method == method]
            best_row = np.argmin(t_method['RMSE'])
            batch_norm = False  # t_method.iloc[best_row]['BatchNorm']

            # grab the mean, std, and samples
            best_trial = t_method['RMSE'].idxmin()
            mean = np.squeeze(image_reshape(plots[method][batch_norm]['reconstruction'][best_trial]['mean'][0::2]))
            std = np.squeeze(image_reshape(plots[method][batch_norm]['reconstruction'][best_trial]['std'][0::2]))
            if len(std.shape) == 3:
                std = 1 - std
            sample = np.squeeze(image_reshape(plots[method][batch_norm]['reconstruction'][best_trial]['sample'][0::2]))
            ax[i].imshow(np.concatenate((x, mean, std, sample), axis=0), vmin=0, vmax=1, cmap='Greys')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_ylabel(method.replace('Uniform', 'MLE'), fontsize=13)

        # save figure
        fig.savefig(os.path.join('assets', 'fig_vae_samples_' + data + '.pdf'))


def generative_analysis():
    # get list of VAE experiments
    results = glob.glob(os.path.join('results', 'generative_*_metrics.pkl'))
    results.sort()

    # build tables
    with open(os.path.join('assets', 'generative_table.tex'), 'w') as f:
        print(generate_tables(results, main_body=False), file=f)
    with open(os.path.join('assets', 'generative_table_short.tex'), 'w') as f:
        print(generate_tables(results, main_body=True), file=f)

    # generate plots
    generate_plots(results)


if __name__ == '__main__':
    # script arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # make assets folder if it doesn't already exist
    if not os.path.exists('assets'):
        os.mkdir('assets')

    # run analysis accordingly
    generative_analysis()

    # hold the plots
    plt.show()
