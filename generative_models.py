import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from scipy.stats import gamma

from generative_data import load_data_set
from regression_models import softplus_inverse
from callbacks import LearningCurveCallback, ReconstructionCallback, LatentVisualizationCallback2D

# workaround: https://github.com/tensorflow/tensorflow/issues/34888
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, enable=True)
tf.config.experimental.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')

EPSILON = 1e-6
EARLY_STOP_METRIC = 'vi 2 Log Likelihood'


def encoder_dense(dim_in, dim_out, batch_norm, name):

    enc = tf.keras.Sequential(name=name)
    enc.add(tf.keras.Input(shape=dim_in, dtype=tf.float32))
    enc.add(tf.keras.layers.Flatten())
    enc.add(tf.keras.layers.Dense(units=512))
    if batch_norm:
        enc.add(tf.keras.layers.BatchNormalization())
    enc.add(tf.keras.layers.ELU())
    enc.add(tf.keras.layers.Dense(units=256))
    if batch_norm:
        enc.add(tf.keras.layers.BatchNormalization())
    enc.add(tf.keras.layers.ELU())
    enc.add(tf.keras.layers.Dense(units=128))
    if batch_norm:
        enc.add(tf.keras.layers.BatchNormalization())
    enc.add(tf.keras.layers.ELU())
    enc.add(tf.keras.layers.Dense(units=dim_out))
    return enc


def decoder_dense(dim_in, dim_out, batch_norm, final_activation, name):

    dec = tf.keras.Sequential(name=name)
    dec.add(tf.keras.Input(shape=dim_in, dtype=tf.float32))
    dec.add(tf.keras.layers.Flatten())
    dec.add(tf.keras.layers.Dense(units=128))
    if batch_norm:
        dec.add(tf.keras.layers.BatchNormalization())
    dec.add(tf.keras.layers.ELU())
    dec.add(tf.keras.layers.Dense(units=256))
    if batch_norm:
        dec.add(tf.keras.layers.BatchNormalization())
    dec.add(tf.keras.layers.ELU())
    dec.add(tf.keras.layers.Dense(units=512))
    if batch_norm:
        dec.add(tf.keras.layers.BatchNormalization())
    dec.add(tf.keras.layers.ELU())
    dec.add(tf.keras.layers.Dense(units=dim_out, activation=final_activation))
    return dec


def encoder_convolution(dim_in, dim_out, _, name):

    return tf.keras.Sequential(name=name, layers=[
        tf.keras.Input(shape=dim_in, dtype=tf.float32),
        tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        tf.keras.layers.ELU(),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        tf.keras.layers.ELU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128),
        tf.keras.layers.ELU(),
        tf.keras.layers.Dense(units=dim_out)])


def decoder_convolution(dim_in, dim_out, _, final_activation, name):

    return tf.keras.Sequential(name=name, layers=[
        tf.keras.Input(shape=dim_in, dtype=tf.float32),
        tf.keras.layers.Dense(units=128),
        tf.keras.layers.ELU(),
        tf.keras.layers.Dense(units=64 * dim_out[0] // 4 * dim_out[1] // 4),
        tf.keras.layers.ELU(),
        tf.keras.layers.Reshape((dim_out[0] // 4, dim_out[1] // 4, 64)),
        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.UpSampling2D(interpolation='bilinear', data_format='channels_last'),
        tf.keras.layers.ELU(),
        tf.keras.layers.Conv2DTranspose(filters=dim_out[-1], kernel_size=5, strides=1, activation=final_activation, padding='same'),
        tf.keras.layers.UpSampling2D(interpolation='bilinear', data_format='channels_last'),
        tf.keras.layers.Flatten()])


def mixture_network(dim_in, dim_out, batch_norm, name):
    net = tf.keras.Sequential(name=name)
    net.add(tf.keras.Input(shape=dim_in, dtype=tf.float32))
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(units=10))
    if batch_norm:
        net.add(tf.keras.layers.BatchNormalization())
    net.add(tf.keras.layers.ELU())
    net.add(tf.keras.layers.Dense(units=10))
    if batch_norm:
        net.add(tf.keras.layers.BatchNormalization())
    net.add(tf.keras.layers.ELU())
    net.add(tf.keras.layers.Dense(units=10))
    if batch_norm:
        net.add(tf.keras.layers.BatchNormalization())
    net.add(tf.keras.layers.ELU())
    net.add(tf.keras.layers.Dense(units=dim_out, activation='softmax'))
    return net


def precision_prior_params(data, num_classes, pseudo_inputs_per_class):

    # load the data into RAM to support sample with replacement
    x = []
    y = []
    for batch in data:
        x.append(batch['image'])
        y.append(batch['label'])
    x = tf.concat(x, axis=0)
    y = tf.concat(y, axis=0)

    # git distribution of precision across pixel positions
    variance = tf.math.reduce_variance(tf.keras.layers.Flatten()(x), axis=0)
    precision = 1 / tf.clip_by_value(variance, clip_value_min=(1 / 255), clip_value_max=np.inf)
    a, _, b_inv = gamma.fit(precision, floc=0)
    b = 1 / b_inv

    # randomly select pseudo inputs
    u = []
    for i in range(num_classes):
        i_choice = np.random.choice(np.where(y == i)[0], size=pseudo_inputs_per_class, replace=False)
        u.append(tf.gather(params=x, indices=i_choice, axis=0))
    u = tf.concat(u, axis=0)

    return a, b, u


class VAE(tf.keras.Model):

    def __init__(self, dim_x, dim_z, architecture, batch_norm, num_mc_samples=1, latex_metrics=True):
        super(VAE, self).__init__()
        assert isinstance(dim_x, list) or isinstance(dim_x, tuple)
        assert isinstance(dim_z, int) and dim_z > 0
        assert architecture in {'dense', 'convolution'}
        assert isinstance(batch_norm, bool) and not (batch_norm and architecture == 'convolution')
        assert isinstance(num_mc_samples, int) and num_mc_samples > 0
        assert isinstance(latex_metrics, bool)

        # save configuration
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.num_mc_samples = num_mc_samples
        self.latex_metrics = latex_metrics

        # set prior for latent normal
        self.pz = tfp.distributions.Normal(loc=tf.zeros(self.dim_z), scale=tf.ones(self.dim_z))
        self.pz = tfp.distributions.Independent(self.pz, reinterpreted_batch_ndims=1)

        # flatten layer
        self.flatten = tf.keras.layers.Flatten()

        # encoder
        encoder = encoder_dense if architecture == 'dense' else encoder_convolution
        self.qz = encoder(dim_x, 2 * dim_z, batch_norm, name='qz')
        self.qz.add(tfp.layers.IndependentNormal(dim_z))

    def posterior_predictive(self, x=None, z=None):
        assert x is not None or z is not None
        if z is None:
            z = self.qz(x).mean()
        px_z = self.px(z)
        x_mean = tf.reshape(px_z.mean(), [-1] + list(self.dim_x))
        x_std = tf.reshape(px_z.stddev(), [-1] + list(self.dim_x))
        x_new = tf.reshape(px_z.sample(), [-1] + list(self.dim_x))
        return x_mean, x_std, x_new, px_z.entropy()

    def call(self, inputs, **kwargs):

        # variational objective
        elbo, ll, dkl_z, dkl_p = self.variational_objective(x=inputs['image'])

        # add loss
        self.add_loss(-tf.reduce_mean(elbo))

        # add metrics for call backs
        self.add_metric(elbo, name='vi 1 ELBO' if self.latex_metrics else 'ELBO', aggregation='mean')
        self.add_metric(ll, name=EARLY_STOP_METRIC if self.latex_metrics else 'LL', aggregation='mean')
        dkl_z_name = 'vi 3 $D_{KL}(q(z|x)||p(z))$' if self.latex_metrics else 'DKL(z)'
        self.add_metric(dkl_z, name=dkl_z_name, aggregation='mean')
        if dkl_p is not None:
            dkl_p_name = 'vi 4 $D_{KL}(q(\\lambda|z)||p(\\lambda))$' if self.latex_metrics else 'DKL(p)'
            self.add_metric(dkl_p, name=dkl_p_name, aggregation='mean')

        return tf.constant(0.0, dtype=tf.float32)


class NormalVAE(VAE):

    def __init__(self, dim_x, dim_z, architecture, batch_norm, split_decoder, a=None, b=None, num_mc_samples=1, latex_metrics=True):
        super(NormalVAE, self).__init__(dim_x, dim_z, architecture, batch_norm, num_mc_samples, latex_metrics)
        assert architecture in {'dense', 'convolution'}
        assert isinstance(split_decoder, bool)
        assert isinstance(a, (type(None), float))
        assert isinstance(b, (type(None), float))

        # save configuration
        self.split_decoder = split_decoder
        self.a = None if a is None else tf.constant(a, dtype=tf.float32)
        self.b = None if b is None else tf.constant(b, dtype=tf.float32)

        # select network architectures accordingly
        decoder = decoder_dense if architecture == 'dense' else decoder_convolution
        dim_out = np.prod(self.dim_x) if architecture == 'dense' else self.dim_x
        
        # decoder
        if split_decoder:
            self.mu = decoder(dim_z, dim_out, batch_norm, final_activation=None, name='mu_x')
            self.sigma = decoder(dim_z, dim_out, batch_norm, final_activation='softplus', name='sigma_x')
            self.px = lambda z: tfp.distributions.MultivariateNormalDiag(self.mu(z), self.sigma(z) ** 2 + EPSILON)
        else:
            dim_out = 2 * dim_out if architecture == 'dense' else list(dim_out[:-1]) + [2 * dim_out[-1]]
            self.px = decoder(dim_z, dim_out, batch_norm, final_activation=None, name='px')
            self.px.add(tfp.layers.IndependentNormal(np.prod(dim_x)))

    def variational_objective(self, x):

        # variational family q(z;x)
        qz_x = self.qz(x)
        dkl_z = qz_x.kl_divergence(self.pz)

        # monte-carlo estimate E[log p(x|z)]
        x = self.flatten(x)
        z_samples = qz_x.sample(sample_shape=self.num_mc_samples)
        ll = tf.reduce_mean(tf.map_fn(lambda z: self.px(z).log_prob(x), z_samples), axis=0)

        # MAP-VAE option: adds log p(precision) to ELBO
        if self.b is not None:
            precision = tf.reduce_mean(1 / tf.map_fn(lambda z: self.px(z).variance(), z_samples), axis=0)
            pp = tfp.distributions.Gamma(tf.constant(1.0, dtype=tf.float32) if self.a is None else self.a, self.b)
            ll_precision = tf.reduce_sum(pp.log_prob(precision), axis=1)
        else:
            ll_precision = tf.constant(0.0, dtype=tf.float32)

        # evidence lower bound
        elbo = ll - dkl_z + ll_precision

        return elbo, ll, dkl_z, None


class StudentVAE(VAE):

    def __init__(self, dim_x, dim_z, architecture, batch_norm, num_mc_samples=1, latex_metrics=True):
        super(StudentVAE, self).__init__(dim_x, dim_z, architecture, batch_norm, num_mc_samples, latex_metrics)
        assert architecture in {'dense', 'convolution'}

        # select network architectures accordingly
        decoder = decoder_dense if architecture == 'dense' else decoder_convolution
        dim_out = np.prod(self.dim_x) if architecture == 'dense' else self.dim_x

        # decoder
        self.mu = decoder(dim_z, dim_out, batch_norm, final_activation=None, name='mu_x')
        self.nu = decoder(dim_z, dim_out, batch_norm, final_activation='softplus', name='nu_x')
        self.precision = decoder(dim_z, dim_out, batch_norm, final_activation='softplus', name='lambda_x')

    def px(self, z):
        scale = 1 / tf.sqrt(self.precision(z) + EPSILON)
        px = tfp.distributions.StudentT(df=self.nu(z) + 2 + EPSILON, loc=self.mu(z), scale=scale)
        return tfp.distributions.Independent(px)

    def variational_objective(self, x):

        # variational family q(z;x)
        qz_x = self.qz(x)
        dkl_z = qz_x.kl_divergence(self.pz)

        # monte-carlo estimate E[log p(x|z)]
        x = self.flatten(x)
        z_samples = qz_x.sample(sample_shape=self.num_mc_samples)
        ll = tf.reduce_mean(tf.map_fn(lambda z: self.px(z).log_prob(x), z_samples), axis=0)

        # evidence lower bound
        elbo = ll - dkl_z

        return elbo, ll, dkl_z, None


class VariationalVarianceVAE(VAE):

    def __init__(self, dim_x, dim_z, architecture, batch_norm, prior, a=None, b=None, u=None, k=None, num_mc_samples=1, latex_metrics=True):
        super(VariationalVarianceVAE, self).__init__(dim_x, dim_z, architecture, batch_norm, num_mc_samples, latex_metrics)
        assert prior in {'mle', 'standard', 'vamp', 'vamp_trainable', 'vbem'}
        assert isinstance(a, (type(None), float))
        assert isinstance(b, (type(None), float))

        # save configuration
        self.prior = prior
        if self.prior != 'mle':
            self.a = tf.constant([a] * np.prod(dim_x), dtype=tf.float32)
            self.b = tf.constant([b] * np.prod(dim_x), dtype=tf.float32)

        # precision prior options
        if self.prior == 'standard':
            self.pp = tfp.distributions.Gamma(concentration=self.a, rate=self.b)
            self.pp = tfp.distributions.Independent(self.pp, reinterpreted_batch_ndims=1)
        elif 'vamp' in self.prior:
            trainable = 'trainable' in self.prior
            self.u = tf.Variable(initial_value=u, dtype=tf.float32, trainable=trainable, name='u')
        elif self.prior == 'vbem':
            if k > 1:
                u = tf.random.uniform(shape=(k, np.prod(dim_x)), minval=-3, maxval=3, dtype=tf.float32)
                v = tf.random.uniform(shape=(k, np.prod(dim_x)), minval=-3, maxval=3, dtype=tf.float32)
            else:
                u = softplus_inverse(tf.expand_dims(self.a, axis=0))
                v = softplus_inverse(tf.expand_dims(self.b, axis=0))
            self.u = tf.Variable(initial_value=u, dtype=tf.float32, trainable=True, name='u')
            self.v = tf.Variable(initial_value=v, dtype=tf.float32, trainable=True, name='v')

        # select network architectures accordingly
        decoder = decoder_dense if architecture == 'dense' else decoder_convolution
        dim_out = np.prod(self.dim_x) if architecture == 'dense' else self.dim_x

        # build parameter networks
        self.mu = decoder(dim_z, dim_out, batch_norm, final_activation=None, name='mu_x')
        self.alpha = decoder(dim_z, dim_out, batch_norm, final_activation='softplus', name='alpha_x')
        self.beta = decoder(dim_z, dim_out, batch_norm, final_activation='softplus', name='beta_x')
        if self.prior in {'vamp', 'vamp_trainable', 'vbem'}:
            self.pi = mixture_network(dim_z, self.u.shape[0], batch_norm, name='pi')
            self.pc = tfp.distributions.Categorical(logits=[1] * self.u.shape[0])

    def px(self, z):
        """Not used in training--Only in posterior-predictive of super class."""
        mu = self.mu(z)
        alpha = self.alpha(z) + EPSILON
        beta = self.beta(z) + EPSILON
        px = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.sqrt(beta / alpha))
        return tfp.distributions.Independent(px)

    def map_fn(self, x, z):

        # run parameter networks
        mu = self.mu(z)
        alpha = self.alpha(z) + EPSILON
        beta = self.beta(z) + EPSILON

        # variational family q(precision|z)
        qp = tfp.distributions.Independent(tfp.distributions.Gamma(alpha, beta))

        # compute log p(x|z)
        expected_log_lambda = tf.math.digamma(alpha) - tf.math.log(beta)
        ll = 0.5 * (expected_log_lambda - tf.math.log(2 * np.pi) - (self.flatten(x) - mu) ** 2 * qp.mean())
        ll = tf.reduce_sum(ll, axis=1)

        # compute kl-divergence depending on prior type
        if self.prior == 'standard':
            dkl = qp.kl_divergence(self.pp)
        elif self.prior in {'vamp', 'vamp_trainable', 'vbem'}:
            if self.prior in {'vamp', 'vamp_trainable'}:
                a = self.alpha(self.qz(self.u).sample()) + EPSILON
                b = self.beta(self.qz(self.u).sample()) + EPSILON
            else:
                a = tf.nn.softplus(self.u) + EPSILON
                b = tf.nn.softplus(self.v) + EPSILON

            # compute VAMP prior's mixing densities
            priors = tfp.distributions.Independent(tfp.distributions.Gamma(a, b), reinterpreted_batch_ndims=1)

            # MC estimate kl-divergence due to pesky log-sum
            pi_z = tf.clip_by_value(self.pi(z), clip_value_min=EPSILON, clip_value_max=tf.float32.max)
            p = qp.sample(self.num_mc_samples)
            log_qp = qp.log_prob(p)
            p = tf.tile(tf.expand_dims(p, axis=-2), [1, 1] + priors.batch_shape.as_list() + [1])
            log_pp = tf.reduce_logsumexp(priors.log_prob(p) + tf.math.log(tf.expand_dims(pi_z, axis=0)), axis=-1)
            dkl = tf.reduce_mean(log_qp - log_pp, axis=0)
            dkl += tfp.distributions.Categorical(logits=pi_z).kl_divergence(self.pc)

        else:
            dkl = tf.constant(0.0, dtype=tf.float32)

        return ll, dkl

    def variational_objective(self, x):

        # variational family: q(z;x)
        qz_x = self.qz(x)
        dkl_z = qz_x.kl_divergence(self.pz)

        # monte-carlo estimate E[log p(x|z)] and DKL(q(p)||p(p))
        x = self.flatten(x)
        z_samples = qz_x.sample(sample_shape=self.num_mc_samples)
        ll, dkl_p = tf.map_fn(lambda z: self.map_fn(x, z), z_samples, dtype=(tf.float32, tf.float32))
        ll = tf.reduce_mean(ll, axis=0)
        dkl_p = tf.reduce_mean(dkl_p, axis=0)

        # evidence lower bound
        elbo = ll - dkl_z - dkl_p

        return elbo, ll, dkl_z, dkl_p


if __name__ == '__main__':

    # set configuration
    PX_FAMILY = 'Normal'
    BATCH_SIZE = 250
    ARCH = 'dense'
    BATCH_NORM = False
    DIM_Z = 10
    PSEUDO_INPUTS_PER_CLASS = 10

    # load the data set
    train_set, test_set, info = load_data_set(data_set_name='mnist', px_family=PX_FAMILY, batch_size=BATCH_SIZE)
    DIM_X = info.features['image'].shape

    # get precision prior parameters
    A, B, U = precision_prior_params(data=train_set,
                                     num_classes=info.features['label'].num_classes,
                                     pseudo_inputs_per_class=PSEUDO_INPUTS_PER_CLASS)

    # VAE with shared mean/variance decoder network -- good at 1e-4 lr and < 100 epochs
    # vae = NormalVAE(dim_x=DIM_X, dim_z=DIM_Z, architecture=ARCH, batch_norm=BATCH_NORM, split_decoder=False)

    # VAE with split mean/variance decoder network -- good at 1e-4 lr and < 100 epochs
    # vae = NormalVAE(dim_x=DIM_X, architecture=ARCH, batch_norm=BATCH_NORM, split_decoder=True)

    # MAP VAE (Takahashi, 2018) -- got at 1e-4 lr and < 150
    # vae = NormalVAE(dim_x=DIM_X, architecture=ARCH, batch_norm=BATCH_NORM, split_decoder=True, b=1e-3)

    # Student-T VAE (Takahashi, 2018) -- good at 1e-4 lr and 750 epochs
    # vae = StudentVAE(dim_x=DIM_X, dim_z=DIM_Z, architecture=ARCH, batch_norm=BATCH_NORM)

    # Empirical-Bayes MAP VAE (ours)
    # vae = NormalVAE(dim_x=DIM_X, architecture=ARCH, batch_norm=BATCH_NORM, split_decoder=True, a=A, b=B)

    # Variational Variance VAE (ours)
    vae = VariationalVarianceVAE(dim_x=DIM_X, dim_z=DIM_Z, architecture=ARCH, batch_norm=BATCH_NORM,
                                 prior='standard', a=1., b=1e-3)

    # build the model. loss=[None] avoids warning "Output output_1 missing from loss dictionary".
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=[None], run_eagerly=False)

    # train the model
    vae.fit(train_set, validation_data=test_set, epochs=1000, verbose=0,
            validation_steps=np.ceil(info.splits['test'].num_examples // BATCH_SIZE),
            callbacks=[LearningCurveCallback(train_set),
                       ReconstructionCallback(train_set, info.features['label'].num_classes),
                       LatentVisualizationCallback2D(vae.dim_x, vae.dim_z),
                       tf.keras.callbacks.EarlyStopping(monitor='val_' + EARLY_STOP_METRIC,
                                                        min_delta=1.0, patience=50, mode='max')])
    print('Done!')

    # keep plots open
    plt.show()
