## Detection (test AUROC of the difference-of-means direction at the validation-selected layer)

| model | lang | layer / n | A: known vs fictional | A token-matched | C: known vs rest | probe C | surface-only C | pop. Spearman |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5 0.5B | en | 11/24 | 0.912 [0.869, 0.949] | 0.916 | 0.886 [0.832, 0.934] | 0.867 [0.804, 0.919] | 0.711 | 0.133 |
| Qwen2.5 0.5B | es | 12/24 | 0.918 [0.868, 0.959] | 0.946 | 0.900 [0.840, 0.944] | 0.884 [0.825, 0.934] | 0.649 | 0.097 |
| Qwen2.5 0.5B | hi | – | insufficient data | | | | | |

## Behaviour of the unmodified model (plain prompt)

| model | lang | accuracy on real | abstain on real | abstain on fictional | with abstention prompt: abstain on fictional |
|---|---|---|---|---|---|
| Qwen2.5 0.5B | en | 0.101 | 0.001 | 0.000 | 0.151 |
| Qwen2.5 0.5B | es | 0.084 | 0.001 | 0.002 | 0.522 |
| Qwen2.5 0.5B | hi | 0.014 | 0.002 | 0.001 | 0.033 |

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
