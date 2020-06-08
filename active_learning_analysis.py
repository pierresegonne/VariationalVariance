import os
import glob
import math
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from analysis_utils import clean_prior_names, make_clean_method_names, build_table

# enable background tiles on plots
sns.set(color_codes=True)


def fix_early_runs(pickle_files):
    for result in pickle_files:
        log = pd.read_pickle(result)

        # eliminate POOPS priors
        log = log[log.Prior != 'vamp_poops']
        log = log[log.Prior != 'vamp_trainable_poops']
        log = log[log.Prior != 'vbem_poops']

        # fix percentage offset for our models
        for prior in log['Prior'].unique():
            if prior != 'N/A':
                log.loc[log.Prior == prior, 'Percent'] = log[log.Prior == 'N/A']

        # save the result
        log.to_pickle(result)


def generate_plots(pickle_files, metric):
    assert metric in {'LL', 'MAE', 'RMSE'}

    # generate subplots
    n_rows = math.ceil(len(pickle_files) / 5)
    n_cols = min(5, len(pickle_files))
    fig, ax = plt.subplots(n_rows, n_cols, **{'figsize': (4 * n_cols, 4 * n_rows)})
    ax = np.reshape(ax, -1)

    # make it tight
    plt.subplots_adjust(left=0.05, bottom=0.15, right=0.98, top=0.95, wspace=0.20, hspace=0.20)

    # plot results
    for i, result in enumerate(pickle_files):
        log = pd.read_pickle(result)
        log = make_clean_method_names(log)

        # plot results
        ax[i].set_title(result.split('_')[-1].split('.')[0])
        sns.lineplot(x='Ratio', y=metric, hue='Method', ci='sd', data=log.rename(columns={'Percent': 'Ratio'}), ax=ax[i])

        # y label once per row
        if i % n_cols != 0:
            ax[i].set_ylabel('')

        # x label only bottom row
        if i // n_cols < n_rows - 1:
            ax[i].set_xlabel('')

        # shared legend
        if i == (n_rows - 1) * n_cols:
            ax[i].legend(ncol=len(log['Method'].unique()) + 1, loc='lower left', bbox_to_anchor=(0, -0.36), fontsize=13)
        else:
            ax[i].legend().remove()

    return fig


def integrate_active_learning_curves(log, **kwargs):
    return pd.DataFrame(log.groupby(['Algorithm', 'Prior', log.index], sort=False)[kwargs['metric']].sum())


def active_learning_analysis():

    # get list of UCI experiments
    results = glob.glob(os.path.join('results', 'active_learning_uci_*.pkl'))
    results.sort()

    # generate plots
    for metric in ['LL', 'MAE', 'RMSE']:
        generate_plots(results, metric).savefig(os.path.join('assets', 'fig_al_' + metric.lower() + '.pdf'))

    # print result tables
    max_cols = 5
    process_fn = [clean_prior_names, integrate_active_learning_curves]
    with open(os.path.join('assets', 'active_learning_uci_ll.tex'), 'w') as f:
        print(build_table(results, 'LL', order='max', max_cols=max_cols, process_fn=process_fn), file=f)
    with open(os.path.join('assets', 'active_learning_uci_mae.tex'), 'w') as f:
        print(build_table(results, 'MAE', order='min', max_cols=max_cols, process_fn=process_fn), file=f)
    with open(os.path.join('assets', 'active_learning_uci_rmse.tex'), 'w') as f:
        print(build_table(results, 'RMSE', order='min', max_cols=max_cols, process_fn=process_fn), file=f)


if __name__ == '__main__':
    # script arguments
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # make assets folder if it doesn't already exist
    if not os.path.exists('assets'):
        os.mkdir('assets')

    # run analysis accordingly
    active_learning_analysis()

    # hold the plots
    plt.show()
