#!/bin/sh

# download data
python regression_data.py

# run toy data experiments
python regression_experiments.py --data toy

# run UCI regression experiments
python regression_experiments.py --data boston
python regression_experiments.py --data carbon
python regression_experiments.py --data concrete
python regression_experiments.py --data energy
python regression_experiments.py --data naval
python regression_experiments.py --data "power plant"
python regression_experiments.py --data superconductivity
python regression_experiments.py --data wine-red
python regression_experiments.py --data wine-white
python regression_experiments.py --data yacht

# run active learning experiments
python active_learning_experiments.py --data boston
python active_learning_experiments.py --data carbon
python active_learning_experiments.py --data concrete
python active_learning_experiments.py --data energy
python active_learning_experiments.py --data naval
python active_learning_experiments.py --data "power plant"
python active_learning_experiments.py --data superconductivity
python active_learning_experiments.py --data wine-red
python active_learning_experiments.py --data wine-white
python active_learning_experiments.py --data yacht

# run VAE experiments
python generative_experiments.py --data fashion_mnist
python generative_experiments.py --data mnist
python generative_experiments.py --data svhn_cropped
