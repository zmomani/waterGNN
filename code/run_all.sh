#!/bin/bash
# Regenerates every dataset used in the paper: 2018 and 2019 as one two-year run
# (about 15 minutes per run; two runs at a time).
cd "$(dirname "$0")"
python3 01_build_network.py
mkdir -p ../outputs
D=730
python3 02_simulate.py --regime continuous --days $D --leaks --events &
python3 02_simulate.py --regime intermittent --schedule S48 --days $D --leaks --events &
wait
python3 02_simulate.py --regime continuous --days $D &
python3 02_simulate.py --regime intermittent --schedule S48 --days $D &
wait
for S in S24 S72; do
  python3 02_simulate.py --regime intermittent --schedule $S --days $D --leaks --events &
  python3 02_simulate.py --regime intermittent --schedule $S --days $D &
  wait
done
