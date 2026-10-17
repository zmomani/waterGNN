#!/bin/bash
cd "$(dirname "$0")"; R=../results; export OMP_NUM_THREADS=2
m(){ grep -q '"MLP"' $R/detection_$1.json 2>/dev/null || python3 10_detect_regimes.py --residual normal --only MLP --name $1 ${@:2} > $R/detect_$1_MLP.log 2>&1; }
S="--ref intermittent_S48_noleak --val intermittent_S48_leaks"; C="--ref continuous_CWS_noleak --val continuous_CWS_leaks"
m N_tuned_E3        --tag s48_tuned $S --test intermittent_S48_leaks
m N_tuned_E1        --tag cws_tuned $C --test continuous_CWS_leaks
m N_tuned_drop6_S48 --tag s48_tuned $S --test intermittent_S48_leaks --drop 6
m N_tuned_E2        --tag cws_tuned $C --test intermittent_S48_leaks
m N_tuned_S72       --tag s48_tuned $S --test intermittent_S72_leaks
m N_tuned_S24       --tag s48_tuned $S --test intermittent_S24_leaks
echo MLP NORMAL DONE
