#!/bin/bash
# Hyperparameter sweep of the GNN on the intermittent dataset (S48). Chosen on the
# leak-free validation months of 2018 only. Finished runs are skipped.
cd "$(dirname "$0")"; M=../models; R=../results
t(){ [ -f $M/reconstructor_$1.pt ] || OMP_NUM_THREADS=2 python3 06_train_reconstructor.py --run intermittent_S48_noleak --hide --epochs 12 --tag $1 ${@:2} > $R/train_$1.log 2>&1; }
t sw_k3_l4_c64   --order 3
t sw_k5_l8_c64   --order 5 --layers 8
t sw_k10_l4_c128 --hidden 128
t sw_k10_l4_c64_lr5e-4 --lr 5e-4
t sw_k10_l4_c64_wd1e-4 --wd 1e-4
t sw_k3_l8_c128  --order 3 --layers 8 --hidden 128
t sw_k10_l2_c64  --layers 2
echo SWEEP1 DONE
# second round: learned node embedding
t sw_k10_l4_c64_e16 --embed 16
t sw_k5_l8_c64_e16  --order 5 --layers 8 --embed 16
t sw_k3_l4_c64_e16  --order 3 --embed 16
t sw_k10_l4_c128_e16 --hidden 128 --embed 16
echo SWEEP2 DONE
