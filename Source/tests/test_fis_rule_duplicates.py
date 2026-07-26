#!/usr/bin/env python3
"""Validate FIS rule handling keeps duplicate/conflicting sample rules."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    root = repo_root()
    source_code = root / "Source_code"
    if str(source_code) not in sys.path:
        sys.path.insert(0, str(source_code))

    from module.Rules_Function.Rules_reduce import reduce_rule, remove_rule  # noqa: PLC0415
    from module.Test_FIS.match_rule import match_rule  # noqa: PLC0415

    # Columns: antecedent_0, antecedent_1, consequent, certainty, raw_label.
    weighted_rules = np.array(
        [
            [1, 1, 2, 0.95, 0],
            [1, 1, 2, 0.90, 0],
            [1, 1, 3, 0.92, 1],
            [2, 1, 2, 0.80, 0],
        ],
        dtype=float,
    )

    reduced = reduce_rule(h=4, col_num=2, rules=weighted_rules)
    model_rules = remove_rule(h=4, col_num=2, rules=reduced)

    assert reduced.tolist() == [
        [1.0, 1.0, 2.0],
        [1.0, 1.0, 2.0],
        [1.0, 1.0, 3.0],
        [2.0, 1.0, 2.0],
    ]
    assert model_rules.tolist() == reduced.tolist()

    # Duplicate A->2 appears twice and A->3 appears once, so duplicate-aware
    # matching should return 2 instead of whichever matching rule appears first.
    assert int(match_rule(np.array([1, 1]), model_rules)) == 2
    assert int(match_rule(np.array([2, 1]), model_rules)) == 2
    assert match_rule(np.array([9, 9]), model_rules) is None

    print("PASS: FIS duplicate/conflicting rules are preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
