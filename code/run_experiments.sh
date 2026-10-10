#!/bin/bash
# Trains the models and runs experiments E1 to E3. Finished steps are skipped,
# so the script can simply be started again after an interruption.
cd "$(dirname "$0")"
M=../models; R=../results; mkdir -p $M $R
[ -f $M/reconstructor_cws_hide.pt ] || python3 06_train_reconstructor.py --run continuous_CWS_noleak --tag cws_hide --hide --epochs 12
[ -f $M/reconstructor_s48_hide.pt ] || python3 06_train_reconstructor.py --run intermittent_S48_noleak --tag s48_hide --hide --epochs 12
[ -f $M/reconstructor_s48_aware.pt ] || python3 06_train_reconstructor.py --run intermittent_S48_noleak --tag s48_aware --hide --aware --epochs 12
[ -f $R/detection_E2.json ] || python3 10_detect_regimes.py --name E2 --tag cws_hide --ref continuous_CWS_noleak --val continuous_CWS_leaks --test intermittent_S48_leaks
[ -f $R/detection_E3_blind.json ] || python3 10_detect_regimes.py --name E3_blind --tag s48_hide --ref intermittent_S48_noleak --val intermittent_S48_leaks --test intermittent_S48_leaks
[ -f $R/detection_E3_aware.json ] || python3 10_detect_regimes.py --name E3_aware --tag s48_aware --aware --ref intermittent_S48_noleak --val intermittent_S48_leaks --test intermittent_S48_leaks
echo ALL DONE
