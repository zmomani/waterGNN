#!/bin/bash
# Extra trainings for the robustness tests; finished steps are skipped.
cd "$(dirname "$0")"; M=../models
t(){ [ -f $M/reconstructor_$2.pt ] || python3 06_train_reconstructor.py --run $1 --tag $2 --hide --epochs 12 ${@:3} > ../results/train_$2.log 2>&1; }
t intermittent_S48_noleak s48_m1 --months 1
t intermittent_S48_noleak s48_m3 --months 3
echo TRAIN DONE
