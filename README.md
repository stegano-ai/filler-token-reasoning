# LLMs can learn (some) multi-step reasoning on filler tokens

Code, configs and results for the LessWrong post of the same name.

Filler-token reasoning is using extra token positions (here: a chain of thought made only of `.` tokens) to reason about a task, without the emitted tokens carrying any of the reasoning. We train LoRAs on Qwen3.6-35B-A3B and gpt-oss-20b to do multi-step tasks this way, measure how many steps they can learn, and compare with constructed upper bounds on the transformer depth each task needs.

## Repository structure

```
lib/side_tasks/            # the seven tasks: mod sum, A5, GL(3,2), modular arithmetic, ProntoQA, HornQA, function composition
lib/prontoqa/              # loader for the vendored PrOntoQA generator (Apache-2.0, see vendor/LICENSE)

experiments/latent_sft/    # training runs
  functions/               #   train_latent_sft.py (one stage), train_curriculum.py (ladder of stages)
  configs/                 #   one meta config per campaign
  results/                 #   metrics and configs of every run behind a figure

experiments/theory/        # depth upper bounds
  functions/               #   rs_transformer.py (executable transformer model), check_circuit.py, frontier.py
  problems/<task>/         #   problem.py, best_transformer_algorithm.md, bound.md, ideas/<name>/circuit.py
  frontier.json            #   shallowest verified algorithm per task, length and model

figures/                   # one folder per figure of the post, plus tables and task examples
test/
```

## Setup

```bash
uv sync
uv run pytest
```

## Reproducing the figures

All figures and tables are regenerated from the committed results; no API key is needed:

```bash
figures/reproduce.sh
```

| In the post | Script | Reads |
|---|---|---|
| Reasoning modes | `figures/reasoning_modes/plot.py` | nothing (schematic) |
| Training method | `figures/method_schematic/plot.py` | nothing (schematic) |
| Method comparison | `figures/method_comparison/plot.py` | `results/comparison_method` |
| Theoretical limits | `figures/theory_frontier/plot.py` | `experiments/theory/frontier.json` |
| Empirical limits | `figures/controls/plot.py` | `results/curriculum`, `direct`, `comparison_method`, `nocot`, `open` |
| Tables | `figures/tables/make_tables.py` | all of the above |
| Task examples | `figures/tasks/make_examples.py` | `configs/tasks.yaml` |

`figures/runs.py` is the single place where results are read and where "certified", "size" and "frontier" are defined.

## Rerunning the training

Training runs on [Tinker](https://tinker-docs.thinkingmachines.ai/) and logs to Weights & Biases, so it needs `TINKER_API_KEY` and `WANDB_API_KEY`. Experiments are orchestrated with [research_scaffold](https://github.com/jr-brown/research-scaffold).

```bash
cd experiments/latent_sft

python main.py -c configs/debug.yaml         # pipeline check, seconds
python main.py -c configs/curriculum.yaml -d # dry run: print the composed configs and exit
python main.py -c configs/curriculum.yaml    # real run
```

| Config | Campaign | Results folder |
|---|---|---|
| `curriculum.yaml` | filler-token reasoning, curriculum over task length | `results/curriculum` |
| `direct.yaml` | filler-token reasoning, trained at full length | `results/direct` |
| `open.yaml` | open-reasoning control | `results/open` |
| `nocot.yaml`, `nocot_oss.yaml` | answer-only control (no extra tokens) | `results/nocot` |
| `comparison.yaml`, `comparison_nocot_seeds.yaml` | ours vs. Pfau et al. vs. answer-only, three seeds | `results/comparison_method` |

`configs/tasks.yaml` holds the task × model axes (prompts, generator settings, curriculum ladders) shared by all campaigns; `configs/base.yaml` holds the hyperparameters. A run with a `.complete` marker in its results folder is skipped on relaunch; delete the folder to rerun it.

The Tinker checkpoint paths recorded in `checkpoints.jsonl` and `figures/tables/runs.md` have expired; they are kept as a record only.

## Theory

`experiments/theory/problems/explanation.md` defines the transformer model and what counts as a valid algorithm. Each `problems/<task>/ideas/<name>/circuit.py` is one construction; `check_circuit` runs it on the executable model, checks every reasoning state and the answer against ground truth, and measures its depth.

```bash
cd experiments/theory

python main.py -c configs/debug.yaml            # check one small construction
python main.py -c configs/rolling_mod_sum.yaml  # check every construction for one task
python main.py -c configs/frontier.yaml         # rebuild frontier.json from the checked constructions
```
