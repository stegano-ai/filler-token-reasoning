from lib.side_tasks.rolling_mod_sum import make_task

MODULUS = 2
OPERAND_RANGE = 1
MIN_OPERAND = 0
REASONING_IN_OUTPUT = False


def instances(n: int, k: int, seed: int) -> list[dict]:
    task = make_task(n_samples=k, seq_length=n, operand_range=OPERAND_RANGE, modulus=MODULUS,
                     reasoning_in_output=REASONING_IN_OUTPUT, seed=seed, min_operand=MIN_OPERAND)
    return task.train_data + task.test_data
