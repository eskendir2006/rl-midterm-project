| Agent | Mean return | Std (over seeds) | Apples delivered / ep | Apples expired / ep | Stuck episodes (%) | Seeds | Eval episodes / seed |
|---|---|---|---|---|---|---|---|
| Random | -72.25 | 0.20 | 0.05 ± 0.01 | 13.61 ± 0.04 | 0.0 | 5 | 500 |
| Greedy (fixed rule) | 146.80 | 0.50 | 13.75 ± 0.05 | 0.00 ± 0.00 | 0.0 | 5 | 500 |
| SARSA (timer) | 145.39 | 2.61 | 13.63 ± 0.24 | 0.00 ± 0.00 | 0.0 | 5 | 500 |
| SARSA (no timer) | 120.80 | 35.77 | 11.56 ± 3.27 | 0.40 ± 0.46 | 15.5 | 5 | 500 |
| SARSA (timer, fixed eps=0.1) | 141.27 | 2.63 | 13.25 ± 0.24 | 0.00 ± 0.00 | 0.0 | 5 | 500 |
