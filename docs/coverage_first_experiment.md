# Coverage-first experiment (branch `coverage-first`)

Question: can the models still below 90 % feature coverage reach it if the singularity criterion is
dropped? Setting: `--coverage-mode --path-subdiv 3 --sing-budget 150 --tri-budget 0.019`
(the coverage stage may add up to 150 irregular vertices per 1k faces instead of staying under
Instant Meshes). Scored with `build/quadeval`, 5000 target faces, 35° creases.

| model | feature coverage | quads | mean SJ | distance | sing./1k | Instant Meshes sing./1k |
|---|---|---|---|---|---|---|
| fandisk_noisy | 90.4 % | 98.2 % | 0.761 | 0.083 % | 170.6 | 36.2 |
| rocker_arm | 95.9 % | 99.0 % | 0.785 | 0.070 % | 149.1 | 34.0 |
| bunny | 88.4 % | 98.9 % | 0.731 | 0.089 % | 170.9 | 46.5 |
| beetle | 78.5 % | 98.5 % | 0.744 | 0.034 % | 141.5 | 66.6 |
| igea | 77.0 % | 99.3 % | 0.811 | 0.049 % | 102.7 | 30.0 |
| damask_chair | 73.2 % | 98.9 % | 0.693 | 0.139 % | 154.6 | 72.1 |
| nefertiti | 73.0 % | 98.3 % | 0.786 | 0.075 % | 158.6 | 48.1 |
| armadillo | 59.9 % | 99.9 % | 0.903 | 0.122 % | 67.2 | 71.5 |

Result: only rocker arm and fandisk_noisy pass 90 %, and both then fail two other criteria
(singularities 4-5x Instant Meshes, mean SJ below 0.8). The organic scans, beetle and damask stay
at 60-79 % even with the singularity limit removed. Not a usable setting; kept as evidence that
the remaining gap is a trade-off against the other criteria, not an unexplored option.
