# waterGNN

Intermittency-Robust Leak Detection and Localisation in Water Distribution Networks Using Graph Neural Networks

Code, dataset generator, trained models and results for a study of learned leak detection under
intermittent (rationed) water supply, built on the public L-Town benchmark (BattLeDIM).

## Branches

| Branch | Contents |
|---|---|
| `dataset-generator` | all code, trained models and results (this branch) |
| `dataset` | the generated datasets: eight two-year runs, split by year (CSV and NPZ) |
| `main` | not yet updated; the pull request from `dataset-generator` is pending |

## Versions

| Tag | Contents |
|---|---|
| `v0.1` | dataset generator and verification |
| `v0.2` | reconstruction GNN, ridge baseline, experiments E1 to E3, BattLeDIM scoring |
| `v0.3` | robustness tests, MLP baseline, threshold-independent localisation, hyperparameter search, tuned GNN |

## Pipeline

```
pip install -r requirements.txt
bash code/run_all.sh                 # 1. generate the eight datasets (about 15 min per run)
python code/05_verify_dataset.py     # 2. check them
bash code/run_experiments.sh         # 3. train the original GNNs, run E1 to E3
bash code/run_robustness_train.sh    # 4. robustness: extra trainings
bash code/run_robustness_detect.sh   #    robustness: detection under changed conditions
bash code/run_mlp_detect.sh          #    MLP baseline in every experiment
python code/12_robustness.py sensors # 5. reconstruction under sensor failure, schedule change, less data
python code/12_robustness.py schedule
python code/12_robustness.py months
python code/12_robustness.py mlp
python code/14_localisation.py whole # 6. localisation of all 23 leaks, independent of the alarm threshold
bash code/run_sweep.sh               # 7. hyperparameter search (rounds 1 and 2)
bash code/run_sweep3.sh              #    round 3
bash code/run_final_train.sh         #    final tuned models
bash code/run_tuned_chain.sh         # 8. every experiment again with the tuned GNN
bash code/run_wide.sh                #    detector settings on the wide grid, all models
python code/11_plot_summary.py       # 9. figures
python code/13_plot_robustness.py
python code/15_plot_sweep.py
python code/16_plot_tuned.py W
```

Every script skips steps whose outputs exist, so a chain can be restarted after an interruption.
All runs reported in the paper were made on two processor cores without a GPU; training the
tuned GNN takes about 95 minutes.

## Code

| File | Purpose |
|---|---|
| `code/common.py` | every parameter and assumption of the dataset |
| `code/01_build_network.py` | supply sectors, isolation valves, household tanks, leak nodes |
| `code/02_simulate.py` | continuous or intermittent simulation, one 5-minute step at a time |
| `code/05_verify_dataset.py` | checks of the generated runs |
| `code/scoring.py` | BattLeDIM economic score (Vrachimis et al. 2022) |
| `code/gnn_data.py` | graph, node features, sensor groups, data loading |
| `code/gnn_model.py` | Chebyshev graph convolution and the reconstruction network (optional node embedding) |
| `code/gnn_load.py` | loads a trained model with the architecture its training recorded |
| `code/06_train_reconstructor.py` | trains a GNN; `--hidden --order --layers --embed --epochs --months` |
| `code/08_detect.py`, `09_detect_nodes.py` | detectors of experiment E1 (sensor level, node level) |
| `code/10_detect_regimes.py` | detector for all experiments; ridge and MLP baselines; `--drop` failed sensors; `--wide` grid |
| `code/12_robustness.py` | reconstruction error under sensor failure, schedule change, shorter training; MLP |
| `code/14_localisation.py` | localisation of every leak from the residual field |
| `code/15_plot_sweep.py`, `16_plot_tuned.py` | hyperparameter search and tuned-versus-baseline figures |

## Models

`models/` holds the trained weights used in the paper: the original GNNs (`*_hide.pt`), the
tuned GNNs (`*_tuned.pt`, with their `config_*.json` and normalisation `norm_*.npz`) and the MLPs.
The ridge baseline is refitted in seconds by the scripts and is not stored.

## Results

`results/` holds every result file the paper draws on: `detection_*.json` (one per experiment;
`T_` tuned GNN, `W_` wide grid), `detections_*.csv` (every alarm), `robustness_*.json`,
`localisation_*.json`, `sweep_S48.json` and the figures.

## Data

The nominal L-Town model (`data/L-TOWN.inp`) is the one distributed with the EPyT package; the
leak timetable and sensor list (`data/battledim_configuration.yaml`) are from the BattLeDIM
generator. The continuous-supply runs are regenerated from this model and are not the published
BattLeDIM SCADA files.
