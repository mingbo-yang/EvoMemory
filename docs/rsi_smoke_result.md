# ALFWorld paired engineering validation

The published development commit `43f4443f74fa8f489132aaba9e8fa0dd9039b76e` ran one fresh six-game ALFWorld engineering batch on resident Gemma 4 31B. All three panels contained two games. Paired-path coverage was explicitly requested. Model thinking was disabled.

| Measurement | Observed value |
|---|---:|
| Natural trigger | False |
| Closed proposal slots | 9 |
| Passed pre-scoring contract checks | 9 |
| Native episode receipts | 18 |
| Actor requests | 215 |
| Role request intents | 14 |
| Winning old-arm option | retained |
| Winning new-arm option | retained |
| Revised proposal policy adopted | False |

| Meta option | Successes on two tasks |
|---|---:|
| N1 | 2 |
| N2 | 2 |
| N3 | 0 |
| O1 | 2 |
| O2 | 2 |
| O3 | 2 |
| parent | 2 |
| retained | 2 |

Independent auditing reproduced the selected policy and mechanism from the original parent, common proposal evidence, native panel outcomes and the committed pre-evaluation memory. This bounded batch documents engineering behavior; it does not estimate generalization.
