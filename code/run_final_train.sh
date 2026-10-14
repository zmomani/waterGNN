#!/bin/bash
# Final models with the configuration chosen by the sweep (step 15).
cd "$(dirname "$0")"; M=../models; R=../results
t(){ [ -f $M/reconstructor_$2.pt ] || OMP_NUM_THREADS=2 python3 06_train_reconstructor.py --run $1 --tag $2 --hide --hidden 128 --embed 64 ${@:3} > $R/train_$2.log 2>&1; }
t intermittent_S48_noleak s48_tuned --epochs 72
t continuous_CWS_noleak   cws_tuned --epochs 36
echo FINAL TRAIN DONE
