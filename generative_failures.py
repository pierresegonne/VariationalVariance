import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from generative_data import load_data_set
from generative_models import precision_prior_params, VariationalVarianceVAE
from generative_analysis import image_reshape

# import Detlefsen baseline model
sys.path.append(os.path.join(os.getcwd(), 'john-master'))
from experiment_vae import detlefsen_vae_baseline

# set some configurations
priors = ['vbem']
num_methods = 1 + len(priors)
num_epochs = 250
batch_size = 250
dim_z = 10

# initialize figure
fig, ax = plt.subplots(num_methods, 1, figsize=(16, 1.3 * num_methods))
plt.subplots_adjust(left=0.03, bottom=0.01, right=0.99, top=0.99, wspace=0.0, hspace=0.0)

# load data
train_set, test_set, info = load_data_set(data_set_name='mnist', px_family='Normal', batch_size=batch_size)

# get pseudo-inputs
u_init = precision_prior_params(data=test_set,
                                num_classes=info.features['label'].num_classes,
                                pseudo_inputs_per_class=10)[-1]

# get plot points
x_plot = precision_prior_params(data=test_set,
                                num_classes=info.features['label'].num_classes,
                                pseudo_inputs_per_class=10)[-1]

# plot Detlefsen performance
x_train = np.concatenate([x['image'] for x in train_set.as_numpy_iterator()], axis=0)
x_test = np.concatenate([x['image'] for x in test_set.as_numpy_iterator()], axis=0)
_, _, _, x_mean, x_std, x_new = detlefsen_vae_baseline(x_train, x_test, x_plot, dim_z, num_epochs, batch_size)
x_mean = np.squeeze(image_reshape(x_mean))
x_std = np.squeeze(image_reshape(x_std))
x_new = np.squeeze(image_reshape(x_new))
ax[0].imshow(np.concatenate((np.squeeze(image_reshape(x_plot)), x_mean, x_std, x_new), axis=0),
             vmin=0, vmax=1, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_ylabel('Detlefsen')

# loop over our priors that didn't pan out
for i, prior in enumerate(priors):
    vae = VariationalVarianceVAE(dim_x=info.features['image'].shape,
                                 dim_z=dim_z,
                                 architecture='dense',
                                 batch_norm=False,
                                 prior=prior,
                                 a=1.,
                                 b=1e-3,
                                 u=u_init,
                                 k=100,
                                 latex_metrics=False)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=[None], run_eagerly=False)
    vae.fit(train_set, validation_data=test_set, epochs=num_epochs, verbose=1,
            validation_steps=np.ceil(info.splits['test'].num_examples // batch_size),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_LL', min_delta=1.0, patience=50, mode='max')])
    x_mean, x_std, x_new, _ = vae.posterior_predictive(x=x_plot)
    x_mean = np.squeeze(image_reshape(x_mean))
    x_std = np.squeeze(image_reshape(x_std))
    x_new = np.squeeze(image_reshape(x_new))
    ax[i + 1].imshow(np.concatenate((np.squeeze(image_reshape(x_plot)), x_mean, x_std, x_new), axis=0),
                     vmin=0, vmax=1, cmap='Greys')
    ax[i + 1].set_xticks([])
    ax[i + 1].set_yticks([])
    ax[i + 1].set_ylabel(prior.replace('_trainable', '*').upper())

# save figure
fig.savefig(os.path.join('assets', 'fig_vae_fails.pdf'))
plt.show()
