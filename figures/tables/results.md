Lengths in task steps. acc = answer accuracy on the 200-item final eval (100 on first rungs) of that stage's checkpoint; a stage passes when the 20-item gate reads >= 0.9, so a passed stage can sit below 0.9 here. Frontier = deepest trace decile with every decile up to it at >= 0.95 state accuracy, in steps.

| task | model | latent max | from | curriculum: last passed (acc) | curriculum: wall (acc) | one-stage frontier / target | no filler | open |
|---|---|---|---|---|---|---|---|---|
| mod sum | Qwen3.6-35B-A3B | 40 | `latent_mod2_qwen_curriculum` | 40 (0.83) | 60 (0.41) | 10 / 100 | 10 | ≥100 |
| mod sum | gpt-oss-20b | ≥100 | `latent_mod2_oss20b_curriculum` | 100 (0.97) | ladder ran out | 40 / 100 | 10 | ≥100 |
| A5 | Qwen3.6-35B-A3B | 30 | `latent_a5_qwen_curriculum` | 30 (0.84) | 40 (0.69) | 10 / 100 | 15 | ≥100 |
| A5 | gpt-oss-20b | 30 | `latent_a5_oss20b_curriculum` | 30 (0.75) | 40 (0.66) | 20 / 100 | 15 | ≥100 |
| GL(3,2) | Qwen3.6-35B-A3B | 60 | `latent_gl32_qwen_curriculum` | 60 (0.79) | 100 (0.47) | 20 / 100 | 0 | ≥100 |
| GL(3,2) | gpt-oss-20b | 10 | `latent_gl32_oss20b_direct (frontier)` | none | 10 (0.70) | 10 / 100 | 0 | ≥100 |
| modular arithmetic | Qwen3.6-35B-A3B | 3 | `latent_arith8mod_qwen_curriculum` | 3 (0.86) | 5 (0.68) | 0 / 100 | 1 | ≥100 |
| modular arithmetic | gpt-oss-20b | 1 | `latent_arith8mod_oss20b_curriculum` | 1 (1.00) | 2 (0.82) | 0 / 100 | 1 | ≥100 |
| ProntoQA | Qwen3.6-35B-A3B | 8 | `latent_chain_qwen_curriculum` | 8 (0.99) | 16 (0.54) | 5 / 50 | 8 | ≥50 |
| ProntoQA | gpt-oss-20b | 0 | `none` | none | 4 (0.54) | 0 / 50 | 8 | ≥50 |
| HornQA | Qwen3.6-35B-A3B | 0 | `none` | none | 8 (0.56) | 0 / 50 | 0 | ≥50 |
| HornQA | gpt-oss-20b | 0 | `none` | none | 8 (0.52) | 0 / 50 | 0 | ≥50 |
| function composition | Qwen3.6-35B-A3B | 2 | `latent_fc_qwen_curriculum` | 2 (0.92) | 4 (0.14) | 0 / 25 | 1 | ≥25 |
| function composition | gpt-oss-20b | 2 | `latent_fc_oss20b_curriculum` | 2 (1.00) | 4 (0.10) | 0 / 25 | 1 | ≥25 |
