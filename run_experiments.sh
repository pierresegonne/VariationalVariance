#!/bin/sh

# download data
python regression_data.py

# set mode
MODE="resume"

# run toy data experiments
BATCH_ITERATIONS=6000
python regression_experiments_v2.py --algorithm "Detlefsen" --dataset toy --batch_iterations 6000 --mode $MODE
python regression_experiments_v2.py --algorithm "Detlefsen (fixed)" --dataset toy --batch_iterations 6000 --mode $MODE
declare -a Algorithms=("Gamma-Normal" "LogNormal-Normal")
declare -a PriorTypes=("MLE" "Standard")
for alg in "${Algorithms[@]}"; do
  for prior in "${PriorTypes[@]}"; do
    python regression_experiments_v2.py --algorithm $alg --dataset toy --batch_iterations 6000 --mode $MODE \
      --prior_type $prior --k 20
  done
done

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
