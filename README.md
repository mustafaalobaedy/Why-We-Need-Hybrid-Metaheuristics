# Why We Need Hybrid Metaheuristics

Code and data for the study **"Why We Need Hybrid Metaheuristics: An Empirical Analysis of When Hybridization Pays Off"** (manuscript under review).

The study measures the effect of hybridization itself: nine hybrid metaheuristics, covering the four classes of Talbi's taxonomy, are compared with **their own parent algorithms** under identical budgets on the CEC 2017 benchmark suite (D = 10, 30, 50; 51 runs), and the analysis is repeated on a traveling salesman case study (TSPLIB).

| Parents | GA, DE, PSO, Nelder–Mead, L-SHADE, CMA-ES (IPOP) |
|---|---|
| Hybrids | GA-PSO, DEPSO, NM-PSO (LTH) · MA-GA, MA-DE, MA-LSHADE (LRH) · DE→NM, L-SHADE→CMA-ES (HRH) · Island GA/DE/PSO (HTH) |
| TSP | ACS, ILS (2-opt + double bridge), ACS+2-opt, ACS→ILS |
| Measures | hybridization advantage rate (HAR), hybridization loss rate (HLR), net hybridization gain (NHG) against the best parent |

## Repository layout
```
03_Code/src/       algorithms.py (all 15 algorithms), core.py (budget/checkpoints), cec17.py (CEC2017 interface),
                   stats.py (Wilcoxon, Holm, Vargha-Delaney A12, Friedman), tsp.py + tsp/tsp_meta.c (TSP solvers)
03_Code/scripts/   run_cec2017.py, run_tsp.py (experiments), analyze.py, analyze_tsp.py (tables and figures),
                   bibliometrics.py, fig_bibliometrics.py (OpenAlex analysis), verify_cec17.py (benchmark check)
04_Data/           TSPLIB instances, OpenAlex bibliometric counts
05_Results/raw/    one CSV row per run: final error and the error at 14 budget checkpoints
```

## Requirements and build
Python 3.10, a C/C++ compiler, and the packages in `requirements.txt`:
```
pip install -r requirements.txt
./build.sh            # compiles the CEC2017 and TSP libraries and checks the benchmark optima
```

## Reproducing the experiments
```
python3 03_Code/scripts/run_cec2017.py --runs 51 --out 05_Results/raw/cec2017_main.csv
python3 03_Code/scripts/run_tsp.py --out 05_Results/raw/tsp_main.csv
python3 03_Code/scripts/analyze.py 05_Results/raw/cec2017_main.csv 05_Results/tables 06_Figures
python3 03_Code/scripts/analyze_tsp.py 05_Results/raw/tsp_main.csv 05_Results/tables 06_Figures
```
The experiments are resumable: tasks already present in the output CSV are skipped. All algorithms receive the same random seed for a given function, dimension and run (common random numbers).

## Credits
* CEC 2017 benchmark: official C++ code and data by Awad, Ali, Suganthan, Liang and Qu (fast-pow variant by V. Stanovov), redistributed unchanged from <https://github.com/P-N-Suganthan/CEC2017-BoundContrained>.
* CMA-ES: pycma by N. Hansen et al.
* TSPLIB: G. Reinelt (1991).

## License
MIT (see `LICENSE`); third-party benchmark code and data remain under their original terms.

## Author
Mustafa Muwafak Alobaedy
