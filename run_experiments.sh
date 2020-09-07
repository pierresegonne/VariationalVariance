#!/bin/sh

# download data
#python regression_data.py

# set number of parallel jobs
N=4

# set mode
MODE="resume"

# run toy data experiments
#python regression_experiments_v2.py --algorithm "Detlefsen" --dataset toy --batch_iterations 6000 --mode $MODE
#python regression_experiments_v2.py --algorithm "Detlefsen (fixed)" --dataset toy --batch_iterations 6000 --mode $MODE
#declare -a Algorithms=("Gamma-Normal" "LogNormal-Normal")
#declare -a PriorTypes=("MLE" "Standard" "VAMP" "VAMP*" "xVAMP" "xVAMP*" "VBEM*")
#for alg in "${Algorithms[@]}"; do
#  for prior in "${PriorTypes[@]}"; do
#    python regression_experiments_v2.py --dataset toy --algorithm $alg --prior_type $prior --mode $MODE \
#      --batch_iterations 6000 --k 20
#  done
#done

# run UCI regression experiments
declare -a Datasets=("boston" "carbon" "concrete" "energy" "naval" "power plant" "superconductivity" "wine-red" "wine-white" "yacht")
declare -a Algorithms=("Gamma-Normal" "LogNormal-Normal")
declare -a PriorTypes=("MLE" "VAMP" "VAMP*" "xVAMP" "xVAMP*" "VBEM*")
for data in  "${Datasets[@]}"; do
  for alg in "${Algorithms[@]}"; do
    for prior in "${PriorTypes[@]}"; do

      # run job
      python regression_experiments_v2.py --dataset $data --algorithm $alg --prior_type $prior --mode $MODE \
        --batch_iterations 200000 --k 100 --parallel 1 &

      # allow N jobs in parallel
      if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        wait -n
      fi
    done
  done
done
wait
echo "UCI done!"
