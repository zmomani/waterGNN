#!/bin/bash
# third round: embedding size and training length around the best configuration
cd "$(dirname "$0")"; M=../models; R=../results
t(){ [ -f $M/reconstructor_$1.pt ] || OMP_NUM_THREADS=2 python3 06_train_reconstructor.py --run intermittent_S48_noleak --hide --tag $1 ${@:2} > $R/train_$1.log 2>&1; }
t sw_k10_l4_c128_e64      --hidden 128 --embed 64 --epochs 12
t sw_k10_l4_c128_e16_ep36 --hidden 128 --embed 16 --epochs 36
t sw_k10_l4_c128_e64_ep36 --hidden 128 --embed 64 --epochs 36
echo SWEEP3 DONE
