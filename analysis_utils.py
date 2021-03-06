import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind_from_stats


def clean_prior_names(df, **kwargs):
    """
    :param df: a pandas data frame containing experimental results
    :return: a pandas data frame containing the same results but with cleaner prior names
    """
    df = df.replace('mle', 'MLE')
    df = df.replace('standard', 'Standard')
    df = df.replace('vamp', 'VAMP')
    if kwargs['mode'] == 'tex':
        df = df.replace('vamp_trainable', '$\\text{VAMP}^*$')
    else:
        df = df.replace('vamp_trainable', '$\\mathregular{VAMP}^*$')
    df = df.replace('vbem', 'VBEM')
    return df


def make_clean_method_names(df):
    """
    Cleans prior names and adds a Method column from which plot labels can be created
    :param df: a pandas data frame containing experimental results
    :return: a pandas data frame containing the same results but with cleaner prior names and new methods column
    """
    # make clean method names for report
    df = clean_prior_names(df, **{'mode': 'matplotlib'})
    df['Method'] = df['Algorithm'] + ' (' + df['Prior'] + ')'
    df.loc[df.Algorithm == 'Detlefsen', 'Method'] = 'Detlefsen'
    df.loc[df.Algorithm == 'Detlefsen (orig)', 'Method'] = 'Detlefsen'
    df.loc[df.Algorithm == 'Detlefsen (fixed)', 'Method'] = 'Detlefsen (fixed)'
    return df


def build_table(pickle_files, metric, order, max_cols, post_fix='', process_fn=None):
    """
    :param pickle_files: list of pickle files to include in table
    :param metric: name of desired metric (must be column in pandas data frame)
    :param order: how to order best results
    :param max_cols: max columns per row
    :param post_fix: needed to handle files of the form *_uci_[name]_[post-fix].pkl
    :param process_fn: optional processing functions
    :return: 
    """
    if process_fn is None:
        process_fn = []
    assert order in {'max', 'min'}

    # aggregate results into table
    table = None
    test_table = None
    for result in pickle_files:

        # load logger
        log = pd.read_pickle(result)
        n_trials = max(log.index) + 1
        if n_trials < 2:
            continue

        # apply processing functions
        for fn in process_fn:
            log = fn(log, **{'mode': 'tex', 'metric': metric})

        # compute means and standard deviations over methods
        mean = pd.DataFrame(log.groupby(['Algorithm', 'Prior'], sort=False)[metric].mean())
        mean = mean.rename(columns={metric: 'mean'}).sort_values('Algorithm')
        std = pd.DataFrame(log.groupby(['Algorithm', 'Prior'], sort=False)[metric].std(ddof=1))
        std = std.rename(columns={metric: 'std'}).sort_values('Algorithm')

        # build table
        exp = result.split('uci_')[-1].split(post_fix + '.pkl')[0]
        df = pd.DataFrame(mean['mean'].round(3).astype('str') + '$\\pm$' + std['std'].round(3).astype('str'), columns=[exp])

        # get top performer
        i_best = np.argmax(mean) if order == 'max' else np.argmin(mean)

        # get null hypothesis
        null_mean = mean.T[mean.T.columns[i_best]][0]
        null_std = std.T[std.T.columns[i_best]][0]

        # compute p-values
        ms = zip([m[0] for m in mean.to_numpy().tolist()], [s[0] for s in std.to_numpy().tolist()])
        p = [ttest_ind_from_stats(null_mean, null_std, n_trials, m, s, n_trials, False)[-1] for (m, s) in ms]

        # bold statistical ties for best
        for i in range(df.shape[0]):
            if i == i_best or p[i] >= 0.05:
                df.loc[mean.index[i]] = '\\textbf{' + df.loc[mean.index[i]] + '}'

        # append experiment to results table
        table = df if table is None else table.join(df)

        # build test table for viewing with PyCharm SciView
        mean = mean.rename(columns={'mean': exp})
        test_table = mean if test_table is None else test_table.join(mean)

    # split tables into a maximum number of cols
    i = 0
    tables = []
    experiments = []
    while i < table.shape[1]:
        experiments.append(table.columns[i:i + max_cols])
        tables.append(table[experiments[-1]])
        i += max_cols
    tables = [t.to_latex(escape=False) for t in tables]

    # add experimental details
    for i in range(len(tables)):
        target = 'Algorithm & Prior'
        i_start = tables[i].find(target)
        i_stop = tables[i][i_start:].find('\\')
        assert len(tables[i][i_start + len(target):i_start + i_stop].split('&')) == len(experiments[i]) + 1
        details = ''
        for experiment in experiments[i]:
            experiment = experiment.split('_')[0]
            with open(os.path.join('data', experiment, experiment + '.pkl'), 'rb') as f:
                dd = pickle.load(f)
            details += '& ({:d}, {:d}, {:d})'.format(dd['data'].shape[0], dd['data'].shape[1], dd['target'].shape[1])
        tables[i] = tables[i][:i_start + len(target)] + details + tables[i][i_start + i_stop:]

    # merge the tables into a single table
    if len(tables) > 1:
        tables[0] = tables[0].split('\\bottomrule')[0]
    for i in range(1, len(tables)):
        tables[i] = '\\midrule' + tables[i].split('\\toprule')[-1]

    return ''.join(tables)
