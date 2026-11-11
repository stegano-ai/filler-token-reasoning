# One example per task at a small length: the exact user prompt the model sees, the per-step reasoning states the dot
# positions are trained on, and the answer. Prompts and generator settings come from experiments/latent_sft/configs/tasks.yaml.
#   uv run --no-project --with pyyaml python figures/tasks/make_examples.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from runs import ROOT, SIZE_KEYS, TASKS, merge

sys.path.insert(0, str(ROOT))
from lib.side_tasks import get_side_task

SIZE = {"mod2": 10, "a5": 10, "gl32": 10, "arith8mod": 3, "fc": 3, "chain": 4, "horn": 8}
axes = yaml.safe_load((ROOT / "experiments/latent_sft/configs/tasks.yaml").read_text())["experiments"][0]["config_axes"][0]
out = []
for t in [t for t in axes if t['name'] in TASKS]:
    fk = t["function_kwargs"]
    st = merge(fk["side_task"], fk["stages"][0]["side_task"])
    st.update({next(k for k in SIZE_KEYS if k in st): SIZE[t["name"]], "n_samples": 200})
    side = get_side_task(**st)
    item = side.test_data[0]
    prompt = fk["task_prompt"].format(SIDE_INPUT=item["input"], N_VALUES=len(item["reasoning"]), MAX_VALUE=max(side.alphabet),
                                      SIDE_TASK_EXPLANATION=side.explanation)
    out.append(f"## {TASKS[t['name']][0].replace(chr(10), ' ')} (`{t['name']}`, {SIZE[t['name']]} steps)\n\n"
               f"```\n{prompt.strip()}\n```\n\nreasoning states (one per dot): `{item['reasoning']}`  \n"
               f"answer: `{item['output'][-1]}`\n")
(Path(__file__).resolve().parent / "examples.md").write_text("\n".join(out))
