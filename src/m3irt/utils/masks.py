import math
from typing import List, Optional, Tuple

import numpy as np


def strip_prefix(name: str) -> str:
    prefixes = ["<no_image>", "<no_question>", "<no_info>"]
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def create_balanced_mask(
    response_data: np.ndarray,
    test_percentage: float,
    item_names: List[str],
    seed: Optional[int] = None,
    val_percentage: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:

    num_students, num_items = response_data.shape

    if not (0 <= test_percentage <= 1):
        raise ValueError(f"test_percentage must be in [0, 1], got {test_percentage}")
    if not (0 <= val_percentage <= 1):
        raise ValueError(f"val_percentage must be in [0, 1], got {val_percentage}")
    if len(item_names) != num_items:
        raise ValueError(f"item_names length must be {num_items}, got {len(item_names)}")

    rnd = np.random.RandomState(seed)

    group_to_indices = {}
    for idx, name in enumerate(item_names):
        group = strip_prefix(name)
        group_to_indices.setdefault(group, []).append(idx)

    mask = np.full(response_data.shape, -1, dtype=int)

    for group, indices in group_to_indices.items():
        candidate_students = np.arange(num_students)
        test_count = max(1, math.ceil(test_percentage * num_students))
        selected_test = rnd.choice(candidate_students, size=test_count, replace=False)
        for student in selected_test:
            mask[student, indices] = 0

        candidate_students_val = []
        for student in range(num_students):
            if np.all(mask[student, indices] == -1):
                candidate_students_val.append(student)
        candidate_students_val = np.array(candidate_students_val)
        val_count = max(1, math.ceil(val_percentage * num_students))
        if candidate_students_val.size > 0 and val_percentage > 0:
            select_val = rnd.choice(
                candidate_students_val, size=min(candidate_students_val.size, val_count), replace=False
            )
            for student in select_val:
                mask[student, indices] = 2

    masked_data = np.where((mask == 0) | (mask == 2), response_data, -1)
    return mask, masked_data


def create_balanced_mask_with_fixed_test(
    mask: np.ndarray,
    response_data: np.ndarray,
    train_percentage: float,
    item_names: List[str],
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    num_students, num_items = response_data.shape

    if not (0 <= train_percentage <= 1):
        raise ValueError(f"train_percentage must be in [0, 1], got {train_percentage}")
    if len(item_names) != num_items:
        raise ValueError(f"item_names length must be {num_items}, got {len(item_names)}")

    final_mask = mask.copy()
    rnd = np.random.RandomState(seed)

    group_to_indices = {}
    for idx, name in enumerate(item_names):
        group = strip_prefix(name)
        group_to_indices.setdefault(group, []).append(idx)

    for group, indices in group_to_indices.items():
        candidate_students = []
        for student in range(num_students):
            if np.all(final_mask[student, indices] == -1):
                candidate_students.append(student)
        candidate_students = np.array(candidate_students)
        if candidate_students.size == 0:
            continue
        desired_count = max(1, math.ceil(train_percentage * num_students))
        select_count = min(candidate_students.size, desired_count)
        selected_students = rnd.choice(candidate_students, size=select_count, replace=False)
        for student in selected_students:
            final_mask[student, indices] = 1

    masked_data = np.where((final_mask == 0) | (final_mask == 1) | (final_mask == 2), response_data, -1)
    return final_mask, masked_data


def create_validation_mask_for_items(
    response_data: np.ndarray,
    item_names: List[str],
    val_item_names: List[str],
    val_percentage: float,
    seed_init: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    num_students, num_items = response_data.shape

    if not (0 <= val_percentage <= 1):
        raise ValueError(f"val_percentage must be in [0, 1], got {val_percentage}")
    for name in val_item_names:
        if name not in item_names:
            raise ValueError(f"val_item_names contains '{name}' which is not in item_names")

    mask = np.full(response_data.shape, fill_value=1, dtype=int)

    for i, val_name in enumerate(val_item_names):
        col_idx = item_names.index(val_name)
        candidates = list(range(num_students))
        if not candidates or val_percentage == 0:
            continue
        seed = None if seed_init is None else seed_init + i
        rnd = np.random.RandomState(seed)
        val_count = max(1, math.ceil(val_percentage * num_students))
        n_select = min(len(candidates), val_count)
        selected = rnd.choice(candidates, size=n_select, replace=False)
        for s in selected:
            mask[s, col_idx] = 2

    masked_data = np.where(mask == 2, response_data, -1)

    return mask, masked_data


def create_test_mask_for_groups(
    mask: np.ndarray,
    response_data: np.ndarray,
    test_percentage: float,
    item_names: List[str],
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    num_students, num_items = mask.shape

    if not (0 <= test_percentage <= 1):
        raise ValueError(f"test_percentage must be in [0, 1], got {test_percentage}")
    if len(item_names) != num_items:
        raise ValueError(f"item_names length must be {num_items}, got {len(item_names)}")

    rnd = np.random.RandomState(seed)
    new_mask = mask.copy()

    group_to_indices = {}
    for idx, name in enumerate(item_names):
        group = strip_prefix(name)
        group_to_indices.setdefault(group, []).append(idx)

    for group, indices in group_to_indices.items():
        candidates = [s for s in range(num_students) if np.all(new_mask[s, indices] == -1)]
        if not candidates:
            continue
        test_count = max(1, math.ceil(test_percentage * num_students))
        selected = rnd.choice(candidates, size=min(len(candidates), test_count), replace=False)
        for s in selected:
            new_mask[s, indices] = 0

    masked_data = np.where((new_mask == 0) | (new_mask == 2), response_data, -1)
    return new_mask, masked_data


if __name__ == "__main__":
    num_students = 10
    num_items = 8
    response_data = np.arange(num_students * num_items).reshape(num_students, num_items)

    item_names = [
        "Q1",
        "<no_image>Q1",
        "<no_question>Q1",
        "<no_info>Q1",
        "Q2",
        "<no_image>Q2",
        "<no_question>Q2",
        "<no_info>Q2",
    ]

    test_percentage = 0.05
    mask, masked_data = create_balanced_mask(response_data, test_percentage, item_names=item_names, seed=4)
    print("create_balanced_mask's result")
    print("mask:\n", mask)
    print("masked_data:\n", masked_data)

    mask2, masked_data2 = create_balanced_mask(
        response_data, test_percentage, val_percentage=0.05, item_names=item_names, seed=4
    )
    print("\ncreate_balanced_mask's result with val_percentage")
    print("mask2:\n", mask2)
    print("masked_data2:\n", masked_data2)

    train_percentage = 0.9
    final_mask, final_masked_data = create_balanced_mask_with_fixed_test(
        mask2, response_data, train_percentage, item_names, seed=42
    )
    print("\ncreate_balanced_mask_with_fixed_test's result")
    print("final_mask:\n", final_mask)
    print("final_masked_data:\n", final_masked_data)
