#!/bin/sh

# generates figures showing that uniform VAMP prior fails on heteroscedastic variance
python regression_models.py --prior vamp_uniform

# generate figures for VAE methods that failed
python generative_failures.py