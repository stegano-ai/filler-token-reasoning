import hashlib
from dataclasses import dataclass
from typing import Callable

_TRAIN_RATIO = 0.8


def split_value(name: str, key) -> float:
    """A problem's fixed position in [0, 1); below _TRAIN_RATIO is train territory.

    Membership is a stable hash of the problem itself (salted with the task name), not of
    the sampling draw, so no problem can appear on both sides of the split — under any
    seed, n_samples, or run."""
    digest = hashlib.md5(f"{name}:{key}".encode()).hexdigest()
    return int(digest[:8], 16) / 2**32


def sample_split(n_samples: int, draw: Callable[[], dict], key: Callable[[dict], object],
                 name: str) -> tuple[list[dict], list[dict]]:
    """Draw items i.i.d. until the 80/20 quotas fill, routing each by split_value of its key.

    Train keeps duplicate draws (it is the training distribution); test is deduplicated by
    key. Raises when the input space cannot supply enough distinct test problems."""
    n_train = int(n_samples * _TRAIN_RATIO)
    n_test = n_samples - n_train
    train, test, test_keys = [], [], set()
    for _ in range(200 * n_samples):
        if len(train) >= n_train and len(test) >= n_test:
            break
        item = draw()
        k = key(item)
        if split_value(name, k) < _TRAIN_RATIO:
            if len(train) < n_train:
                train.append(item)
        elif k not in test_keys and len(test) < n_test:
            test_keys.add(k)
            test.append(item)
    else:
        raise ValueError(
            f"{name}: could not fill the disjoint split after {200 * n_samples} draws "
            f"(train {len(train)}/{n_train}, test {len(test)}/{n_test} distinct problems); "
            f"the input space is too small — lower n_samples or enlarge the space")
    return train, test


def partition_split(data: list[dict], name: str) -> tuple[list[dict], list[dict]]:
    """Split a generated list by split_value of each item's input; test is deduplicated.

    Sizes land near 80/20 rather than exactly on it; zero overlap is the guarantee."""
    train, test, test_keys = [], [], set()
    for item in data:
        k = repr(item["input"])
        if split_value(name, k) < _TRAIN_RATIO:
            train.append(item)
        elif k not in test_keys:
            test_keys.add(k)
            test.append(item)
    return train, test


@dataclass
class SideTaskDefinition:
    name: str
    train_data: list[dict]  # each: {"input": ..., "reasoning": list[int], "output": list[int]}
    test_data: list[dict]
    explanation: str  # prompt fragment explaining the side task to the model
    alphabet: list[int]  # the values a single reasoning position can take
