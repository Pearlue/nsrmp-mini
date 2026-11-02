### Multi-step test set, 1-2 steps (mean ± sample std over seeds)

| Model | IOU | IOU-M | Subject acc. | Placement acc. |
|---|---|---|---|---|
| Ours (neuro-symbolic) | 0.905 ± 0.001 | 0.776 ± 0.002 | 0.999 ± 0.000 | 0.968 ± 0.003 |
| Parse-matched neural baseline (NMN+-like) | 0.931 ± 0.005 | 0.799 ± 0.012 | 0.999 ± 0.000 | 0.960 ± 0.002 |
| End-to-end GRU baseline | 0.657 ± 0.002 | 0.007 ± 0.002 | 0.272 ± 0.003 | 0.010 ± 0.001 |

### IOU-M vs number of steps (Fig. 4b analogue)

| Model | steps=1 | steps=2 | steps=3 | steps=4 | steps=5 |
|---|---|---|---|---|---|
| Ours (neuro-symbolic) | 0.790 ± 0.001 | 0.774 ± 0.001 | 0.738 ± 0.009 | 0.720 ± 0.007 | 0.694 ± 0.003 |
| Parse-matched neural baseline (NMN+-like) | 0.807 ± 0.014 | 0.797 ± 0.012 | 0.761 ± 0.018 | 0.749 ± 0.012 | 0.729 ± 0.008 |
| End-to-end GRU baseline | 0.003 ± 0.001 | 0.006 ± 0.001 | 0.007 ± 0.001 | 0.007 ± 0.002 | 0.008 ± 0.002 |
