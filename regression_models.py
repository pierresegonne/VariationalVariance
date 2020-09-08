import os
import argparse
import warnings
import numpy as np
import tensorflow as tf
import scipy.stats as sps
import tensorflow_probability as tfp

import seaborn as sns
from matplotlib import pyplot as plt

from callbacks import RegressionCallback
from regression_data import generate_toy_data

# workaround: https://github.com/tensorflow/tensorflow/issues/34888
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)


def softplus_inverse(x):
    return tf.math.log(tf.exp(x) - 1)


def neural_network(d_in, d_hidden, f_hidden, d_out, f_out=None, name=None):
    nn = tf.keras.Sequential(name=name)
    nn.add(tf.keras.layers.InputLayer(d_in))
    nn.add(tf.keras.layers.Dense(d_hidden, f_hidden))
    nn.add(tf.keras.layers.Dense(d_out, f_out))
    return nn


class VariationalNormalRegression(tf.keras.Model):

    def __init__(self, prior_type, y_mean, y_var, num_mc_samples):
        super(VariationalNormalRegression, self).__init__()
        assert isinstance(prior_type, str)
        poops = len(prior_type.split('_poops')) > 1
        prior_type = prior_type.split('_poops')[0]
        assert prior_type in {'mle', 'standard', 'vamp', 'vamp_uniform', 'vamp_trainable', 'vbem'}
        assert not poops or prior_type in {'vamp', 'vamp_trainable', 'vbem'}
        assert isinstance(num_mc_samples, int) and num_mc_samples > 0

        # save configuration
        self.prior_type = prior_type
        self.poops = poops
        self.y_mean = tf.constant(y_mean, dtype=tf.float32)
        self.y_var = tf.constant(y_var, dtype=tf.float32)
        self.y_std = tf.sqrt(self.y_var)
        self.num_mc_samples = num_mc_samples

        self.epsilon_p = tf.constant(0.0, dtype=tf.float32)
        self.epsilon_q = tf.constant(0.0, dtype=tf.float32)

    def px(self, mean, precision):
        px = tfp.distributions.Normal(loc=mean, scale=1 / (tf.sqrt(precision) + self.epsilon_p))
        return tfp.distributions.Independent(px)

    @ staticmethod
    def ll(y, mu, expected_lambda, expected_log_lambda):
        ll = 0.5 * (expected_log_lambda - tf.math.log(2 * np.pi) - (y - mu) ** 2 * expected_lambda)
        return tf.reduce_sum(ll, axis=-1)

    def whiten(self, y, mu, expected_lambda, expected_log_lambda):
        y = (y - self.y_mean) / self.y_std
        return y, mu, expected_lambda, expected_log_lambda

    def de_whiten(self, y, mu, expected_lambda, expected_log_lambda):
        mu = mu * self.y_std + self.y_mean
        expected_lambda = expected_lambda / self.y_var
        expected_log_lambda = expected_log_lambda - tf.math.log(self.y_var)
        return y, mu, expected_lambda, expected_log_lambda

    def variational_objective(self, x, y):

        # run mean network
        mu = self.mu(x)

        # run variational family
        qp, dkl = self.variational_family(x)

        # variational variance log likelihood E_{q(lambda|alpha(x), beta(x))}[log p(y|mu(x), lambda)]
        expected_log_lambda = self.expected_log_lambda(x)
        ll = self.ll(*self.whiten(y, mu, qp.mean(), expected_log_lambda))

        # # fixed-variance log likelihood
        # ll_fv = self.px(mu, 0.25).log_prob((y - self.y_mean) / self.y_std)

        # evidence lower bound
        elbo = ll - dkl

        # compute adjusted log likelihood of non-scaled y using de-whitened model parameter
        ll_adjusted = self.ll(*self.de_whiten(y, mu, qp.mean(), expected_log_lambda))

        # compute squared error for reporting purposes
        error_dist = tf.norm(y - (mu * self.y_std + self.y_mean), axis=-1)
        squared_error = error_dist ** 2

        # add metrics for call backs
        self.add_metric(elbo, name='ELBO', aggregation='mean')
        self.add_metric(ll, name='LL', aggregation='mean')
        self.add_metric(dkl, name='KL', aggregation='mean')
        self.add_metric(ll_adjusted, name='LL (adjusted)', aggregation='mean')
        self.add_metric(error_dist, name='MAE', aggregation='mean')
        self.add_metric(squared_error, name='MSE', aggregation='mean')

        # add minimization objective
        self.add_loss(-tf.reduce_mean(elbo))

    def posterior_predictive_mean(self, x):
        return self.mu(x) * self.y_std + self.y_mean

    def posterior_predictive_std(self, x, num_mc_samples=2000):
        return tf.reduce_mean(1 / (tf.sqrt(self.qp(x).sample(num_mc_samples)) + self.epsilon_p), axis=0) * self.y_std

    def posterior_predictive_sample(self, x):
        return self.posterior_predictive_mean(x) + self.posterior_predictive_std(x) * tf.random.normal(tf.shape(x))

    def posterior_predictive_log_likelihood(self, x, y, exact=True):
        qp = self.qp(x)
        if exact:
            if self.epsilon_p != 0:
                warnings.warn('exact method is approximate since it doesnt account for the eps > 0 in p(x|mu,lambda)')
            ll = tf.reduce_mean(self.ll(*self.de_whiten(y, self.mu(x), qp.mean(), self.expected_log_lambda(x))))
        else:
            precision_samples = qp.sample(sample_shape=self.num_mc_samples) / self.y_var
            mu = self.mu(x) * self.y_std + self.y_mean
            ll = tf.reduce_mean(tf.map_fn(lambda p: self.px(mu, p).log_prob(y), precision_samples))
        return ll

    def call(self, inputs, **kwargs):
        self.variational_objective(x=inputs['x'], y=inputs['y'])
        return tf.constant(0.0, dtype=tf.float32)


class GammaNormalRegression(VariationalNormalRegression):

    def __init__(self, d_in, d_hidden, f_hidden, d_out, prior, y_mean, y_var, a=None, b=None, u=None, k=None, n_mc=1):
        super(GammaNormalRegression, self).__init__(prior, y_mean, y_var, n_mc)
        assert isinstance(d_in, int) and d_in > 0
        assert isinstance(d_hidden, int) and d_hidden > 0
        assert isinstance(d_out, int) and d_out > 0

        # give the model a name
        self.type = 'Gamma-Normal'

        # save fixed prior parameters
        self.a = tf.constant([a] * d_out, dtype=tf.float32)
        self.b = tf.constant([b] * d_out, dtype=tf.float32)

        if self.prior_type == 'standard':
            # set prior for precision
            self.pp = tfp.distributions.Gamma(concentration=self.a, rate=self.b)
            self.pp = tfp.distributions.Independent(self.pp, reinterpreted_batch_ndims=1)
        elif 'vamp' in self.prior_type:
            # pseudo-inputs
            trainable = 'trainable' in self.prior_type
            self.u = tf.Variable(initial_value=u, dtype=tf.float32, trainable=trainable, name='u')
        elif self.prior_type == 'vbem':
            # trainable prior parameters for precision
            u = tf.random.uniform(shape=(k, d_out), minval=-3, maxval=3, dtype=tf.float32)
            v = tf.random.uniform(shape=(k, d_out), minval=-3, maxval=3, dtype=tf.float32)
            self.u = tf.Variable(initial_value=u, dtype=tf.float32, trainable=True, name='u')
            self.v = tf.Variable(initial_value=v, dtype=tf.float32, trainable=True, name='v')

        # build parameter networks
        self.mu = neural_network(d_in, d_hidden, f_hidden, d_out, f_out=None, name='mu')
        self.alpha = neural_network(d_in, d_hidden, f_hidden, d_out, f_out='softplus', name='alpha')
        self.beta = neural_network(d_in, d_hidden, f_hidden, d_out, f_out='softplus', name='beta')
        if self.prior_type in {'vamp', 'vamp_trainable', 'vbem'}:
            d_out = self.u.shape[0] + int(self.poops)
            self.pi = neural_network(d_in, d_hidden, f_hidden, d_out, f_out='softmax', name='pi')
            self.pc = tfp.distributions.Categorical(logits=[1] * self.u.shape[0] + [self.u.shape[0]] * self.poops)

    def qp(self, x):
        qp = tfp.distributions.Gamma(self.alpha(x) + self.epsilon_q, self.beta(x) + self.epsilon_q)
        return tfp.distributions.Independent(qp)

    def variational_family(self, x):
        # variational family q(precision|x)
        qp = self.qp(x)

        # compute kl-divergence depending on prior type
        if self.prior_type == 'standard':
            dkl = qp.kl_divergence(self.pp)
        elif self.prior_type in {'vamp', 'vamp_trainable', 'vamp_uniform', 'vbem'}:
            if self.prior_type in {'vamp', 'vamp_trainable', 'vamp_uniform'}:
                alpha = self.alpha(self.u)
                beta = self.beta(self.u)
            else:
                alpha = tf.nn.softplus(self.u)
                beta = tf.nn.softplus(self.v)
            if self.poops:
                alpha = tf.concat((alpha, tf.expand_dims(self.a, axis=0)), axis=0)
                beta = tf.concat((beta, tf.expand_dims(self.b, axis=0)), axis=0)

            # compute VAMP prior's mixing densities
            priors = tfp.distributions.Gamma(alpha, beta)
            priors = tfp.distributions.Independent(priors, reinterpreted_batch_ndims=1)

            # MC estimate kl-divergence due to pesky log-sum
            if self.prior_type == 'vamp_uniform':
                pi_x = tf.ones(self.u.shape[0])
            else:
                pi_x = tf.clip_by_value(self.pi(x), clip_value_min=1e-6, clip_value_max=tf.float32.max)
            p = qp.sample(self.num_mc_samples)
            log_qp = qp.log_prob(p)
            p = tf.tile(tf.expand_dims(p, axis=-2), [1, 1] + priors.batch_shape.as_list() + [1])
            log_pp = tf.reduce_logsumexp(priors.log_prob(p) + tf.math.log(tf.expand_dims(pi_x, axis=0)), axis=-1)
            dkl = tf.reduce_mean(log_qp - log_pp, axis=0)
            if self.prior_type != 'vamp_uniform':
                dkl += tfp.distributions.Categorical(logits=pi_x).kl_divergence(self.pc)

        else:
            dkl = tf.constant(0.0)

        return qp, dkl

    def expected_log_lambda(self, x):
        return tf.math.digamma(self.alpha(x) + self.epsilon_q) - tf.math.log(self.beta(x) + self.epsilon_q)


class LogNormalNormalRegression(VariationalNormalRegression):

    def __init__(self, d_in, d_hidden, f_hidden, d_out, prior, y_mean, y_var, a=None, b=None, u=None, k=None, n_mc=1):
        super(LogNormalNormalRegression, self).__init__(prior, y_mean, y_var, n_mc)
        assert isinstance(d_in, int) and d_in > 0
        assert isinstance(d_hidden, int) and d_hidden > 0
        assert isinstance(d_out, int) and d_out > 0

        # give the model a name
        self.type = 'LogNormal-Normal'

        # save fixed prior parameters
        self.a = tf.constant([a] * d_out, dtype=tf.float32)
        self.b = tf.constant([b] * d_out, dtype=tf.float32)

        if self.prior_type == 'standard':
            # set prior for precision
            self.pp = tfp.distributions.LogNormal(loc=self.a, scale=self.b)
            self.pp = tfp.distributions.Independent(self.pp, reinterpreted_batch_ndims=1)
        elif 'vamp' in self.prior_type:
            # pseudo-inputs
            trainable = 'trainable' in self.prior_type
            self.u = tf.Variable(initial_value=u, dtype=tf.float32, trainable=trainable, name='u')
        elif self.prior_type == 'vbem':
            # trainable prior parameters for precision
            u = tf.random.uniform(shape=(k, d_out), minval=-3, maxval=3, dtype=tf.float32)
            v = tf.random.uniform(shape=(k, d_out), minval=-3, maxval=3, dtype=tf.float32)
            self.u = tf.Variable(initial_value=u, dtype=tf.float32, trainable=True, name='u')
            self.v = tf.Variable(initial_value=v, dtype=tf.float32, trainable=True, name='v')

        # build parameter networks
        self.mu = neural_network(d_in, d_hidden, f_hidden, d_out, f_out=None, name='mu')
        self.alpha = neural_network(d_in, d_hidden, f_hidden, d_out, f_out=None, name='alpha')
        self.beta = neural_network(d_in, d_hidden, f_hidden, d_out, f_out='softplus', name='beta')
        if self.prior_type in {'vamp', 'vamp_trainable', 'vbem'}:
            d_out = self.u.shape[0] + int(self.poops)
            self.pi = neural_network(d_in, d_hidden, f_hidden, d_out, f_out='softmax', name='pi')
            self.pc = tfp.distributions.Categorical(logits=[1] * self.u.shape[0] + [self.u.shape[0]] * self.poops)

    def qp(self, x):
        qp = tfp.distributions.LogNormal(self.alpha(x), self.beta(x) + self.epsilon_q)
        return tfp.distributions.Independent(qp)

    def variational_family(self, x):
        # variational family q(precision|x)
        qp = self.qp(x)

        # compute kl-divergence depending on prior type
        if self.prior_type == 'standard':
            dkl = qp.kl_divergence(self.pp)
        elif self.prior_type in {'vamp', 'vamp_trainable', 'vamp_uniform', 'vbem'}:
            if self.prior_type in {'vamp', 'vamp_trainable', 'vamp_uniform'}:
                alpha = self.alpha(self.u)
                beta = self.beta(self.u)
            else:
                alpha = self.u
                beta = tf.nn.softplus(self.v)
            if self.poops:
                alpha = tf.concat((alpha, tf.expand_dims(self.a, axis=0)), axis=0)
                beta = tf.concat((beta, tf.expand_dims(self.b, axis=0)), axis=0)

            # compute VAMP prior's mixing densities
            priors = tfp.distributions.LogNormal(alpha, beta)
            priors = tfp.distributions.Independent(priors, reinterpreted_batch_ndims=1)

            # MC estimate kl-divergence due to pesky log-sum
            if self.prior_type == 'vamp_uniform':
                pi_x = tf.ones(self.u.shape[0])
            else:
                pi_x = tf.clip_by_value(self.pi(x), clip_value_min=1e-6, clip_value_max=tf.float32.max)
            p = qp.sample(self.num_mc_samples)
            log_qp = qp.log_prob(p)
            p = tf.tile(tf.expand_dims(p, axis=-2), [1, 1] + priors.batch_shape.as_list() + [1])
            log_pp = tf.reduce_logsumexp(priors.log_prob(p) + tf.math.log(tf.expand_dims(pi_x, axis=0)), axis=-1)
            dkl = tf.reduce_mean(log_qp - log_pp, axis=0)
            if self.prior_type != 'vamp_uniform':
                dkl += tfp.distributions.Categorical(logits=pi_x).kl_divergence(self.pc)

        else:
            dkl = tf.constant(0.0)

        return qp, dkl

    def expected_log_lambda(self, x):
        return self.alpha(x)


def fancy_plot(x_train, y_train, x_eval, true_mean, true_std, mdl_mean, mdl_std, title):
    # squeeze everything
    x_train = np.squeeze(x_train)
    y_train = np.squeeze(y_train)
    x_eval = np.squeeze(x_eval)
    true_mean = np.squeeze(true_mean)
    true_std = np.squeeze(true_std)
    mdl_mean = np.squeeze(mdl_mean)
    mdl_std = np.squeeze(mdl_std)

    # get a new figure
    fig, ax = plt.subplots(2, 1)
    fig.suptitle(title)

    # plot the data
    sns.scatterplot(x_train, y_train, ax=ax[0])

    # plot the true mean and standard deviation
    ax[0].plot(x_eval, true_mean, '--k')
    ax[0].plot(x_eval, true_mean + true_std, ':k')
    ax[0].plot(x_eval, true_mean - true_std, ':k')

    # plot the model's mean and standard deviation
    l = ax[0].plot(x_eval, mdl_mean)[0]
    ax[0].fill_between(x_eval[:, ], mdl_mean - mdl_std, mdl_mean + mdl_std, color=l.get_color(), alpha=0.5)
    ax[0].plot(x_eval, true_mean, '--k')

    # clean it up
    ax[0].set_ylim([-20, 20])
    ax[0].set_ylabel('y')

    # plot the std
    ax[1].plot(x_eval, mdl_std, label='predicted')
    ax[1].plot(x_eval, true_std, '--k', label='truth')
    ax[1].set_ylim([0, 5])
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('std(y|x)')
    plt.legend()

    return fig


if __name__ == '__main__':

    # enable background tiles on plots
    sns.set(color_codes=True)

    # random number seeds
    np.random.seed(123)
    tf.random.set_seed(123)

    # unit test
    test = np.random.uniform(-10, 10, 100)
    assert (np.abs(softplus_inverse(tf.nn.softplus(test)) - test) < 1e-6).all()
    test = np.random.uniform(0, 10, 100)
    assert (np.abs(tf.nn.softplus(softplus_inverse(test)) - test) < 1e-6).all()

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=str, default='vamp_uniform',
                        help="{mle, standard, vamp, vamp_uniform, vamp_trainable, vbem}")
    args = parser.parse_args()

    # set configuration
    D_HIDDEN = 50
    PRIOR_TYPE = args.prior
    N_MC_SAMPLES = 20
    LEARNING_RATE = 1e-2
    EPOCHS = int(6e3)

    # load data
    x_train, y_train, x_eval, true_mean, true_std = generate_toy_data()
    ds_train = tf.data.Dataset.from_tensor_slices({'x': x_train, 'y': y_train}).batch(x_train.shape[0])

    # VAMP prior pseudo-input initializers
    u = np.expand_dims(np.linspace(np.min(x_eval), np.max(x_eval), 20), axis=-1)

    # declare Gamma-Normal model
    a, _, b_inv = sps.gamma.fit(1 / true_std[(np.min(x_train) <= x_eval) * (x_eval <= np.max(x_train))] ** 2, floc=0)
    print('Gamma Prior:', a, 1 / b_inv)
    mdl = GammaNormalRegression(d_in=x_train.shape[1],
                                d_hidden=D_HIDDEN,
                                f_hidden='sigmoid',
                                d_out=y_train.shape[1],
                                prior=PRIOR_TYPE,
                                y_mean=0.0,
                                y_var=1.0,
                                a=a,
                                b=1 / b_inv,
                                k=20,
                                u=u,
                                n_mc=N_MC_SAMPLES)

    # build the model. loss=[None] avoids warning "Output output_1 missing from loss dictionary".
    mdl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=[None], run_eagerly=False)

    # train, evaluate on test points, and plot results
    hist = mdl.fit(ds_train, epochs=EPOCHS, verbose=0, callbacks=[RegressionCallback(EPOCHS)])
    plt.figure()
    plt.plot(hist.history['LL (adjusted)'])

    # print and plot results
    mdl.num_mc_samples = 2000
    print(mdl.posterior_predictive_log_likelihood(x_train, y_train, exact=True))
    print(mdl.posterior_predictive_log_likelihood(x_train, y_train, exact=False))
    mdl_mean, mdl_std = mdl.posterior_predictive_mean(x_eval), mdl.posterior_predictive_std(x_eval)
    fig = fancy_plot(x_train, y_train, x_eval, true_mean, true_std, mdl_mean, mdl_std, mdl.type)
    if PRIOR_TYPE == 'vamp_uniform':
        fig.savefig(os.path.join('assets', 'fig_vamp_uniform_gamma.pdf'))

    # declare LogNormal-Normal model
    precisions = 1 / true_std[(np.min(x_train) <= x_eval) * (x_eval <= np.max(x_train))] ** 2
    a, b = np.mean(np.log(precisions)), np.std(np.log(precisions))
    print('LogNormal Prior:', a, b)
    mdl = LogNormalNormalRegression(d_in=x_train.shape[1],
                                    d_hidden=D_HIDDEN,
                                    f_hidden='sigmoid',
                                    d_out=y_train.shape[1],
                                    prior=PRIOR_TYPE,
                                    y_mean=0.0,
                                    y_var=1.0,
                                    a=a,
                                    b=b,
                                    k=20,
                                    u=u,
                                    n_mc=N_MC_SAMPLES)

    # build the model. loss=[None] avoids warning "Output output_1 missing from loss dictionary".
    mdl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=[None], run_eagerly=False)

    # train, evaluate on test points, and plot results
    hist = mdl.fit(ds_train, epochs=EPOCHS, verbose=0, callbacks=[RegressionCallback(EPOCHS)])
    plt.figure()
    plt.plot(hist.history['LL (adjusted)'])

    # print and plot results
    mdl.num_mc_samples = 2000
    print(mdl.posterior_predictive_log_likelihood(x_train, y_train, exact=True))
    print(mdl.posterior_predictive_log_likelihood(x_train, y_train, exact=False))
    mdl_mean, mdl_std = mdl.posterior_predictive_mean(x_eval), mdl.posterior_predictive_std(x_eval)
    fig = fancy_plot(x_train, y_train, x_eval, true_mean, true_std, mdl_mean, mdl_std, mdl.type)
    if PRIOR_TYPE == 'vamp_uniform':
        fig.savefig(os.path.join('assets', 'fig_vamp_uniform_log_normal.pdf'))

    # hold the plots
    plt.show()
