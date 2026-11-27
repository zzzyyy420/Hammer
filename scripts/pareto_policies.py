"""Candidate-only Pareto interventions. No labels, oracle flags or view IDs required."""
from copy import deepcopy
from hashlib import sha256
import math


def priorities(request_id, candidate_count, seed):
    if candidate_count < 0:
        raise ValueError('negative candidate count')
    # A shared request/slot draw across views and ratios; Python hash() is not stable.
    return [int.from_bytes(sha256(f'routeB-pareto/{seed}/{request_id}/{slot}'.encode()).digest(), 'big')
            for slot in range(candidate_count)]


def random_mask(request_id, candidate_count, ratio, seed):
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('ratio must be finite and in [0, 1]')
    numerator, denominator = float(ratio).as_integer_ratio()
    return [value * denominator < numerator * (1 << 256)
            for value in priorities(request_id, candidate_count, seed)]


def dose_matched_mask(request_id, candidate_count, masked_count, seed):
    if not isinstance(masked_count, int) or not 0 <= masked_count <= candidate_count:
        raise ValueError('masked count must be an integer within the candidate pool')
    values = priorities(request_id, candidate_count, seed)
    chosen = set(sorted(range(candidate_count), key=lambda i: (values[i], i))[:masked_count])
    return [i in chosen for i in range(candidate_count)]


def apply_name_mask(tools, aliases, selected):
    if len(tools) != len(aliases) or len(tools) != len(selected):
        raise ValueError('tool, alias and action lengths differ')
    if len(set(aliases)) != len(aliases):
        raise ValueError('aliases must be a bijection')
    output = deepcopy(tools)
    inverse = {}
    for tool, alias, active in zip(output, aliases, selected, strict=True):
        if active:
            inverse[alias] = tool['name']
            tool['name'] = alias
    if len({t['name'] for t in output}) != len(output):
        raise ValueError('masked names collide with retained names')
    return output, inverse
