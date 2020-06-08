#!/bin/sh

# run analysis scripts
python regression_analysis.py --experiment toy
python regression_analysis.py --experiment uci
python active_learning_analysis.py
python generative_analysis.py
