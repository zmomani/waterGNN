#!/bin/bash
# Detection with the normal-residual (reconstruction against hour-of-week normal), all models.
cd "$(dirname "$0")"; R=../results; export OMP_NUM_THREADS=2
d(){ [ -f $R/detection_$1.json ] || python3 10_detect_regimes.py --residual normal --name $1 ${@:2} > $R/detect_$1.log 2>&1; }
S="--ref intermittent_S48_noleak --val intermittent_S48_leaks"; C="--ref continuous_CWS_noleak --val continuous_CWS_leaks"
for set in tuned hide; do
d N_${set}_E3        --tag s48_$set $S --test intermittent_S48_leaks
d N_${set}_E1        --tag cws_$set $C --test continuous_CWS_leaks
d N_${set}_drop6_S48 --tag s48_$set $S --test intermittent_S48_leaks --drop 6
d N_${set}_E2        --tag cws_$set $C --test intermittent_S48_leaks
d N_${set}_S72       --tag s48_$set $S --test intermittent_S72_leaks
d N_${set}_S24       --tag s48_$set $S --test intermittent_S24_leaks
done
for set in tuned hide; do for c in E3 E1 drop6_S48 E2 S72 S24; do
  grep -q '"MLP"' $R/detection_N_${set}_$c.json 2>/dev/null || true
done; done
echo NORMAL DONE
