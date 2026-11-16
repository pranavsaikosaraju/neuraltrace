## Detection (test AUROC of the difference-of-means direction at the validation-selected layer)

| model | lang | layer / n | A: known vs fictional | A token-matched | C: known vs rest | probe C | surface-only C | pop. Spearman |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5 0.5B | en | 11/24 | 0.912 [0.869, 0.949] | 0.916 | 0.886 [0.832, 0.934] | 0.867 [0.804, 0.919] | 0.711 | 0.133 |
| Qwen2.5 0.5B | es | 12/24 | 0.918 [0.868, 0.959] | 0.946 | 0.900 [0.840, 0.944] | 0.884 [0.825, 0.934] | 0.649 | 0.097 |
| Qwen2.5 0.5B | hi | – | insufficient data | | | | | |
| Qwen2.5 1.5B | en | 10/28 | 0.960 [0.943, 0.975] | 0.960 | 0.925 [0.898, 0.949] | 0.917 [0.886, 0.946] | 0.666 | 0.189 |
| Qwen2.5 1.5B | es | 9/28 | 0.958 [0.935, 0.979] | 0.949 | 0.916 [0.880, 0.945] | 0.872 [0.825, 0.917] | 0.652 | 0.155 |
| Qwen2.5 1.5B | hi | – | insufficient data | | | | | |

## Behaviour of the unmodified model (plain prompt)

| model | lang | accuracy on real | abstain on real | abstain on fictional | with abstention prompt: abstain on fictional |
|---|---|---|---|---|---|
| Qwen2.5 0.5B | en | 0.101 | 0.001 | 0.000 | 0.151 |
| Qwen2.5 0.5B | es | 0.084 | 0.001 | 0.002 | 0.522 |
| Qwen2.5 0.5B | hi | 0.014 | 0.002 | 0.001 | 0.033 |
| Qwen2.5 1.5B | en | 0.177 | 0.002 | 0.013 | 0.783 |
| Qwen2.5 1.5B | es | 0.145 | 0.055 | 0.138 | 0.793 |
| Qwen2.5 1.5B | hi | 0.025 | 0.002 | 0.001 | 0.323 |

## Steering (English): abstention rate on unknown entities / on known entities, 95% Wilson CI

| model | condition | abstain (unknown) | abstain (known) | correct (known) |
|---|---|---|---|---|
| Qwen2.5 0.5B | baseline | 0.000 [0.000, 0.016] | 0.000 [0.000, 0.082] | 1.000 [0.918, 1.000] |
| Qwen2.5 0.5B | prompt_abstain | 0.121 [0.085, 0.168] | 0.000 [0.000, 0.082] | 0.930 [0.814, 0.976] |
| Qwen2.5 0.5B | dom_a-2 | 0.008 [0.002, 0.030] | 0.000 [0.000, 0.082] | 0.093 [0.037, 0.216] |
| Qwen2.5 0.5B | dom_a-1 | 0.000 [0.000, 0.016] | 0.000 [0.000, 0.082] | 0.791 [0.648, 0.886] |
| Qwen2.5 0.5B | dom_a+1 | 0.000 [0.000, 0.016] | 0.000 [0.000, 0.082] | 0.837 [0.700, 0.919] |
| Qwen2.5 0.5B | dom_a+2 | 0.000 [0.000, 0.016] | 0.000 [0.000, 0.082] | 0.302 [0.186, 0.451] |
| Qwen2.5 0.5B | probe_a-1 | 0.000 [0.000, 0.016] | 0.000 [0.000, 0.082] | 0.837 [0.700, 0.919] |
| Qwen2.5 0.5B | probe_a+1 | 0.000 [0.000, 0.016] | 0.000 [0.000, 0.082] | 0.860 [0.727, 0.934] |
| Qwen2.5 0.5B | dom_ablate | 0.000 [0.000, 0.016] | 0.000 [0.000, 0.082] | 0.721 [0.573, 0.833] |
| Qwen2.5 0.5B | random dirs alpha=-1 (n=8) | mean 0.001, range [0.000, 0.004] | | |
| Qwen2.5 0.5B | random dirs alpha=+1 (n=8) | mean 0.009, range [0.000, 0.037] | | |
| Qwen2.5 0.5B | shuffled-label dirs alpha=-1 (n=4) | mean 0.001, range [0.000, 0.004] | | |
| Qwen2.5 0.5B | shuffled-label dirs alpha=+1 (n=4) | mean 0.000, range [0.000, 0.000] | | |
| Qwen2.5 1.5B | baseline | 0.008 [0.002, 0.030] | 0.000 [0.000, 0.042] | 1.000 [0.958, 1.000] |
| Qwen2.5 1.5B | prompt_abstain | 0.717 [0.657, 0.770] | 0.170 [0.106, 0.262] | 0.784 [0.687, 0.857] |
| Qwen2.5 1.5B | dom_a-2 | 0.042 [0.023, 0.075] | 0.023 [0.006, 0.079] | 0.398 [0.302, 0.502] |
| Qwen2.5 1.5B | dom_a-1 | 0.033 [0.017, 0.064] | 0.011 [0.002, 0.062] | 0.886 [0.803, 0.937] |
| Qwen2.5 1.5B | dom_a+1 | 0.000 [0.000, 0.016] | 0.000 [0.000, 0.042] | 0.818 [0.725, 0.885] |
| Qwen2.5 1.5B | dom_a+2 | 0.000 [0.000, 0.016] | 0.000 [0.000, 0.042] | 0.261 [0.181, 0.362] |
| Qwen2.5 1.5B | probe_a-1 | 0.071 [0.045, 0.110] | 0.000 [0.000, 0.042] | 0.909 [0.831, 0.953] |
| Qwen2.5 1.5B | probe_a+1 | 0.000 [0.000, 0.016] | 0.000 [0.000, 0.042] | 0.920 [0.845, 0.961] |
| Qwen2.5 1.5B | dom_ablate | 0.017 [0.006, 0.042] | 0.000 [0.000, 0.042] | 0.830 [0.738, 0.894] |
| Qwen2.5 1.5B | random dirs alpha=-1 (n=8) | mean 0.018, range [0.000, 0.054] | | |
| Qwen2.5 1.5B | random dirs alpha=+1 (n=8) | mean 0.043, range [0.000, 0.217] | | |
| Qwen2.5 1.5B | shuffled-label dirs alpha=-1 (n=4) | mean 0.024, range [0.004, 0.062] | | |
| Qwen2.5 1.5B | shuffled-label dirs alpha=+1 (n=4) | mean 0.007, range [0.000, 0.017] | | |
