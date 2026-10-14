#!/bin/bash
# Detector settings chosen on a wider grid, for both GNN sets and the ridge; fields are cached.
cd "$(dirname "$0")"; R=../results; export OMP_NUM_THREADS=1
while [ ! -f $R/localisation_whole_tuned.json ]; do sleep 30; done
d(){ [ -f $R/detection_$1.json ] || python3 10_detect_regimes.py --wide --name $1 ${@:2} > $R/detect_$1.log 2>&1; }
S="--ref intermittent_S48_noleak --val intermittent_S48_leaks"; C="--ref continuous_CWS_noleak --val continuous_CWS_leaks"
for set in tuned hide; do
d W_${set}_E3        --tag s48_$set $S --test intermittent_S48_leaks
d W_${set}_E1        --tag cws_$set $C --test continuous_CWS_leaks
d W_${set}_E2        --tag cws_$set $C --test intermittent_S48_leaks
d W_${set}_S24       --tag s48_$set $S --test intermittent_S24_leaks
d W_${set}_S72       --tag s48_$set $S --test intermittent_S72_leaks
d W_${set}_drop6_S48 --tag s48_$set $S --test intermittent_S48_leaks --drop 6
d W_${set}_drop6_CWS --tag cws_$set $C --test continuous_CWS_leaks --drop 6
done
echo WIDE DONE
