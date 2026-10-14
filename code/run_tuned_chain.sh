#!/bin/bash
# Every experiment again with the tuned GNN (ridge and MLP unchanged). Finished steps are skipped.
cd "$(dirname "$0")"; R=../results; export OMP_NUM_THREADS=2
while [ ! -f ../models/reconstructor_cws_tuned.pt ]; do sleep 30; done
d(){ [ -f $R/detection_$1.json ] || python3 10_detect_regimes.py --name $1 ${@:2} > $R/detect_$1.log 2>&1; }
S="--ref intermittent_S48_noleak --val intermittent_S48_leaks"; C="--ref continuous_CWS_noleak --val continuous_CWS_leaks"
d T_E3      --tag s48_tuned $S --test intermittent_S48_leaks
d T_E1      --tag cws_tuned $C --test continuous_CWS_leaks
d T_E2      --tag cws_tuned $C --test intermittent_S48_leaks
d T_S24     --tag s48_tuned $S --test intermittent_S24_leaks
d T_S72     --tag s48_tuned $S --test intermittent_S72_leaks
d T_drop6_S48 --tag s48_tuned $S --test intermittent_S48_leaks --drop 6
d T_drop6_CWS --tag cws_tuned $C --test continuous_CWS_leaks --drop 6
export GNN_SET=tuned
[ -f $R/robustness_sensors_tuned.json ]  || python3 12_robustness.py sensors  > $R/robust_sensors_tuned.log 2>&1
[ -f $R/robustness_schedule_tuned.json ] || python3 12_robustness.py schedule > $R/robust_schedule_tuned.log 2>&1
[ -f $R/localisation_whole_tuned.json ]  || python3 14_localisation.py whole   > $R/localisation_tuned.log 2>&1
echo TUNED CHAIN DONE
