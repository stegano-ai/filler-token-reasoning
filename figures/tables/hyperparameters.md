| | |
|---|---|
| LoRA rank | 32 |
| learning rate (Qwen / gpt-oss) | 1.6e-04 / 1.2e-04 |
| LR schedule | constant |
| batch size | 16 |
| train examples per stage | 2000 |
| epochs per stage (max) | 17 |
| max steps per stage | 2125 |
| eval every | 100 steps, 20 items |
| pass gate | eval/output_correct >= 0.9 after >= 50 steps |
| final eval | 200 items |
| renderer (Qwen / gpt-oss) | qwen3_5_disable_thinking / gpt_oss_no_sysprompt |
