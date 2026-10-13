#!/bin/bash
# Adds the multilayer perceptron to every detection experiment.
cd "$(dirname "$0")"; R=../results
d(){ grep -q '"MLP"' $R/detection_$1.json 2>/dev/null || python3 10_detect_regimes.py --only MLP --name $1 ${@:2} > $R/detect_$1_MLP.log 2>&1; }
S="--ref intermittent_S48_noleak --val intermittent_S48_leaks"; C="--ref continuous_CWS_noleak --val continuous_CWS_leaks"
d E3_blind --tag s48_hide $S --test intermittent_S48_leaks
d E1_MLP --tag cws_hide $C --test continuous_CWS_leaks
d E2 --tag cws_hide $C --test intermittent_S48_leaks
d R_S24_blind --tag s48_hide $S --test intermittent_S24_leaks
d R_S72_blind --tag s48_hide $S --test intermittent_S72_leaks
d R_drop6_S48 --tag s48_hide $S --test intermittent_S48_leaks --drop 6
d R_drop6_CWS --tag cws_hide $C --test continuous_CWS_leaks --drop 6
echo MLP DONE
