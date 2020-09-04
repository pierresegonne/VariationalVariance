import os
import glob
import argparse
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from regression_data import generate_toy_data
from analysis_utils import clean_prior_names, make_clean_method_names, build_table

# enable background tiles on plots
sns.set(color_codes=True)


def regression_subplot(alg, prior, ll_logger, data_logger, mv_logger, ax, color):

    # get best performance for this algorithm/prior combination
    i_best = ll_logger.query("Algorithm == '" + alg + "' and Prior == '" + prior + "'")['LL'].idxmax()

    # plot the training data
    data = data_logger.query("Algorithm == '" + alg + "' and Prior == '" + prior + "'").loc[i_best]
    sns.scatterplot(data['x'], data['y'], ax=ax, color=color)

    # plot the model's mean and standard deviation
    model = mv_logger.query("Algorithm == '" + alg + "' and Prior == '" + prior + "'").loc[i_best]
    ax.plot(model['x'], model['mean(y|x)'], color=color)
    ax.fill_between(model['x'],
                    model['mean(y|x)'] - 2 * model['std(y|x)'],
                    model['mean(y|x)'] + 2 * model['std(y|x)'],
                    color=color, alpha=0.5)

    # plot the true mean and standard deviation
    _, _, x_eval, true_mean, true_std = generate_toy_data()
    ax.plot(x_eval, true_mean, '--k')
    ax.plot(x_eval, true_mean + 2 * true_std, ':k')
    ax.plot(x_eval, true_mean - 2 * true_std, ':k')

    # make it pretty
    ax.set_title(model['Method'].unique()[0])


def toy_regression_plot(ll_logger, data_logger, mv_logger):
    # make clean method names for report
    ll_logger = make_clean_method_names(ll_logger)
    data_logger = make_clean_method_names(data_logger)
    mv_logger = make_clean_method_names(mv_logger)

    # get priors
    priors = ll_logger['Prior'].unique()

    # size toy data figure
    n_rows, n_cols = 3, len(priors)
    fig = plt.figure(figsize=(2.9 * n_cols, 2.9 * n_rows), constrained_layout=False)
    gs = fig.add_gridspec(n_rows, n_cols)
    for i, prior in enumerate(priors):
        fig.add_subplot(gs[0, i])
        fig.add_subplot(gs[1, i])
        fig.add_subplot(gs[2, i])

    # make it tight
    plt.subplots_adjust(left=0.05, bottom=0.07, right=0.98, top=0.95, wspace=0.15, hspace=0.15)

    # get color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # plot toy regression subplots
    for i, prior in enumerate(['N/A', 'MLE', 'Standard', 'VAMP', 'VAMP*', 'xVAMP', 'xVAMP*', 'VBEM', 'VBEM*']):
        if prior not in priors:
            continue

        # first row subplots
        ax = fig.axes[n_rows * i]
        alg1 = 'Detlefsen' if prior == 'N/A' else 'Gamma-Normal'
        regression_subplot(alg1, prior, ll_logger, data_logger, mv_logger, ax, colors[0])
        ax.set_xlim([-5, 15])
        ax.set_ylim([-25, 25])
        ax.set_xlabel('')
        ax.set_xticklabels([])
        if i > 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])

        # second row subplots
        ax = fig.axes[n_rows * i + 1]
        alg2 = 'Detlefsen (fixed)' if prior == 'N/A' else 'LogNormal-Normal'
        regression_subplot(alg2, prior, ll_logger, data_logger, mv_logger, ax, colors[1])
        ax.set_xlim([-5, 15])
        ax.set_ylim([-25, 25])
        ax.set_xlabel('')
        ax.set_xticklabels([])
        if i > 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])

        # third row subplots
        ax = fig.axes[n_rows * i + 2]
        _, _, x_eval, _, true_std = generate_toy_data()
        ax.plot(x_eval, true_std, 'k', label='truth')
        query = "(Algorithm == '" + alg1 + "' or Algorithm == '" + alg2 + "') and Prior == '" + prior + "'"
        sns.lineplot(x='x', y='std(y|x)', hue='Method', ci='sd', data=mv_logger.query(query), ax=ax)
        ax.legend().remove()
        ax.set_xlim([-5, 15])
        ax.set_ylim([0, 6])
        if i > 0:
            ax.set_ylabel('')
            ax.set_yticklabels([])

    return fig


def toy_regression_analysis():

    # get all the pickle files
    data_pickles = set(glob.glob(os.path.join('resultsV2', 'toy', '*_data.pkl')))
    mv_pickles = set(glob.glob(os.path.join('resultsV2', 'toy', '*_mv.pkl')))
    ll_pickles = (set(glob.glob(os.path.join('resultsV2', 'toy', '*.pkl'))) - data_pickles) - mv_pickles

    # aggregate results into single data frame
    ll_logger = pd.DataFrame()
    for p in ll_pickles:
        ll_logger = ll_logger.append(pd.read_pickle(p))
    data_logger = pd.DataFrame()
    for p in data_pickles:
        data_logger = data_logger.append(pd.read_pickle(p))
    mv_logger = pd.DataFrame()
    for p in mv_pickles:
        mv_logger = mv_logger.append(pd.read_pickle(p))

    # generate plot
    fig = toy_regression_plot(ll_logger, data_logger, mv_logger)
    fig.savefig(os.path.join('assets', 'fig_toy.pdf'))


def exclude_log_normal(df, **kwargs):
    return df[df.Algorithm != 'LogNormal-Normal']


def only_standards(df, **kwargs):
    return df[(df.Prior == 'Standard') | (df.Prior == 'Standard*')]


def uci_regression_analysis():

    # get list of UCI experiments
    pickles = glob.glob(os.path.join('results', 'regression_uci_*.pkl'))
    pickles.sort()

    # print result tables
    for mode in ['10000', 'full']:
        max_cols = 5
        process_fn = [clean_prior_names]
        if mode == '10000':
            post_fix = '_10000'
            results = [result for result in pickles if mode in result]
        else:
            post_fix = ''
            results = [result for result in pickles if '10000' not in result and 'full' not in result]
        with open(os.path.join('assets', 'regression_uci_ll' + post_fix + '.tex'), 'w') as f:
            print(build_table(results, 'LL', 'max', max_cols, post_fix=post_fix, process_fn=process_fn), file=f)
        with open(os.path.join('assets', 'regression_uci_mae' + post_fix + '.tex'), 'w') as f:
            print(build_table(results, 'MAE', 'min', max_cols, post_fix=post_fix, process_fn=process_fn), file=f)
        with open(os.path.join('assets', 'regression_uci_rmse' + post_fix + '.tex'), 'w') as f:
            print(build_table(results, 'RMSE', 'min', max_cols, post_fix=post_fix, process_fn=process_fn), file=f)

    # print small table for main body
    process_fn = [clean_prior_names, exclude_log_normal]
    with open(os.path.join('assets', 'regression_uci_ll_short.tex'), 'w') as f:
        print(build_table(results, 'LL', 'max', max_cols, post_fix=post_fix, process_fn=process_fn), file=f)

    # print small table for main body
    process_fn = [clean_prior_names, exclude_log_normal, only_standards]
    with open(os.path.join('assets', 'regression_standard_priors.tex'), 'w') as f:
        print(build_table(results, 'LL', None, max_cols, post_fix=post_fix, process_fn=process_fn, transpose=True), file=f)


if __name__ == '__main__':
    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='uci', help='experiment in {toy, uci}')
    args = parser.parse_args()

    # make assets folder if it doesn't already exist
    if not os.path.exists('assets'):
        os.mkdir('assets')

    # run experiments accordingly
    if args.experiment == 'toy':
        toy_regression_analysis()
    else:
        uci_regression_analysis()

    # hold the plots
    plt.show()
