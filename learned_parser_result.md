# Learned parser results

## Round 3 on the formal test scenes

Moved-object IoU (IoU-M), mean ± sample standard deviation across seeds 0–9. Each independently trained model uses its own final checkpoint, parser and move mode on 1,000 test scenes per instruction length.

```text
Group  Parser  Backend        Loss      Schedule Move   Single-step       Two-step
    1  learned NS            original     A     soft   0.2397 ± 0.3064   0.2253 ± 0.2866
    2  learned NS            squared      A     soft   0.5310 ± 0.0478   0.4843 ± 0.0551
    3  rule    NS            original     A     soft   0.7392 ± 0.0216   0.7046 ± 0.0223
    4  rule    NS            squared      A     soft   0.7741 ± 0.0170   0.7430 ± 0.0137
    5  learned NS            squared      B     soft   0.5238 ± 0.0705   0.4807 ± 0.0644
    6  learned NS            squared      B     hard   0.5606 ± 0.0562   0.5187 ± 0.0471
    7  rule    NS            squared      B     soft   0.6877 ± 0.0262   0.6483 ± 0.0208
    8  rule    NS            squared      B     hard   0.7061 ± 0.0221   0.6812 ± 0.0255
    9  learned ParsedNeural  squared      B     soft   0.4946 ± 0.0868   0.4447 ± 0.0786
   10  learned ParsedNeural  squared      B     hard   0.4798 ± 0.1077   0.4303 ± 0.1064
```

ParsedNeural denotes `ParsedNeuralBaseline`. [Per-seed scores and paired differences](results/ten-group-followup/round3/curriculum/summary.json)

**Schedule A (groups 1–4)** trains all learnable modules together using soft moves throughout. Epochs 1–20 use simple single-step instructions. Epochs 21–60 mix those with single-step instructions that use a spatial relation to identify an object.

**Schedule B (groups 5–10)** begins with 20 epochs of joint training on simple instructions using soft moves. In epochs 21–50, only the action simulator trains, still on simple instructions. In epochs 51–70, each group freezes the simulator and trains the relation-grounding module on mixed single-step instructions. Groups with a learned parser also train it during this final phase. (This schedule adapts the paper’s three-phase curriculum to compare soft and hard moves. Within each matched pair, one group keeps soft moves and the other switches to hard moves at epoch 21.)

- With NS, the rule parser led the learned parser under both move schedules across all seeds.
- With learned parsers, NS led ParsedNeural on average, especially with hard moves. Each backend also shaped its own parser during training.
- The squared loss improved mean placement for both parser types and reduced the large seed-to-seed variation of the learned parser under the original loss.
- Hard moves helped the NS groups on average. The neural backend scored slightly higher with soft moves. Under soft moves, the two learned NS training schedules scored similarly.

## Hyperparameters across the three matrices

Each round reran the same ten conditions from scratch on the same seeds. Changes were cumulative. The validation score gives equal weight to complex single-step and overall two-step IoU-M across the six learned-parser groups.

The training runner generates scenes from the configured seeds. The [baseline](experiments.json) and [round 1](results/ten-group-followup/round1.json) use 2,000 simple and 2,000 complex scenes per seed. [Round 2](results/ten-group-followup/round2.json) and [round 3](results/ten-group-followup/round3.json) add 4,000 complex scenes generated with `20000 + seed`. [Training dataset index](learned_parser_experiment.md#training-datasets)

```text
Matrix     Minimum LR   Complex scenes per seed   Samples per instruction*   Validation score
Baseline   0            2,000                     4                          0.31178
Round 1    0.0006       2,000                     4                          0.35642
Round 2    0.0006       6,000                     4                          0.36243
Round 3    0.0006       6,000                     8                          0.37667
```

*Program samples per instruction when training the learned parser. Adam starts at 0.003 in all matrices. The following pairs give single-step / two-step mean IoU-M on the formal test splits.

```text
Group     Round 1           Round 2           Round 3
    1     0.2257 / 0.2101   0.2464 / 0.2298   0.2397 / 0.2253
    2     0.4582 / 0.4152   0.5255 / 0.4819   0.5310 / 0.4843
    3     0.7358 / 0.7037   0.7392 / 0.7046   0.7392 / 0.7046
    4     0.7652 / 0.7344   0.7741 / 0.7430   0.7741 / 0.7430
    5     0.4759 / 0.4322   0.4756 / 0.4287   0.5238 / 0.4807
    6     0.5359 / 0.4962   0.5404 / 0.5040   0.5606 / 0.5187
    7     0.6877 / 0.6479   0.6877 / 0.6483   0.6877 / 0.6483
    8     0.7057 / 0.6810   0.7061 / 0.6812   0.7061 / 0.6812
    9     0.5264 / 0.4741   0.4684 / 0.4204   0.4946 / 0.4447
   10     0.4892 / 0.4382   0.5116 / 0.4619   0.4798 / 0.4303
```

- **Learning-rate floor.** Keeping the cosine learning rate at or above 20% of its starting value gave the largest validation gain. The phased learned-parser groups benefited most. [Round 1](results/ten-group-followup/round1/curriculum/summary.json)
- **Complex-scene diversity.** Adding 4,000 distinct complex scenes per seed produced a smaller average gain. Each mixed epoch still used 2,000 simple and 2,000 complex scenes, with the same update count. The soft neural group declined. [Round 2](results/ten-group-followup/round2/curriculum/summary.json)
- **REINFORCE sampling.** Doubling the candidates increased compute per policy update and raised the validation score, particularly for the phased learned NS groups. The rule-parser groups reproduced their round-2 scores exactly. [Validation by group](results/ten-group-followup/validation-comparison.txt)
