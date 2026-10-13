#!/bin/bash
# Detection under changed conditions; finished steps are skipped.
cd "$(dirname "$0")"; R=../results
d(){ [ -f $R/detection_$1.json ] || python3 10_detect_regimes.py --name $1 ${@:2} > $R/detect_$1.log 2>&1; }
S="--ref intermittent_S48_noleak --val intermittent_S48_leaks"
d R_S24_blind --tag s48_hide $S --test intermittent_S24_leaks
d R_S72_blind --tag s48_hide $S --test intermittent_S72_leaks
d R_drop6_S48 --tag s48_hide $S --test intermittent_S48_leaks --drop 6
d R_drop6_CWS --tag cws_hide --ref continuous_CWS_noleak --val continuous_CWS_leaks --test continuous_CWS_leaks --drop 6
echo DETECT DONE
