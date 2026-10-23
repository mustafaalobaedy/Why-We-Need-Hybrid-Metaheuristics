#!/bin/bash
# Start the budget-split ablation once the main experiment has finished.
cd "$(dirname "$0")/../.."
while pgrep -f "run_cec2017.py --runs 51 --out 05_Results/raw/cec2017_main.csv" > /dev/null; do sleep 120; done
python3 03_Code/scripts/run_cec2017.py --runs 51 --dims 10 30 \
  --algs MA-DE@0.05 MA-DE@0.1 MA-DE@0.4 MA-LSHADE@0.05 MA-LSHADE@0.1 MA-LSHADE@0.4 \
         DE\>NM@0.05 DE\>NM@0.1 DE\>NM@0.4 LSHADE\>CMAES@0.05 LSHADE\>CMAES@0.1 LSHADE\>CMAES@0.4 \
  --out 05_Results/raw/cec2017_ablation_rho.csv
