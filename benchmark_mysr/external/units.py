"""Map public SRSD units to actual input columns, without reading equations."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=256)
def input_units(archive, task):
    if not task.startswith('srsd-'):
        return []
    archive = Path(archive)
    difficulty = next(d for d in ('easy', 'medium', 'hard') if f'-{d}-' in task)
    key = task.split(f'-{difficulty}-', 1)[1]
    raw = archive / 'raw'
    published = raw / 'srsd-published-data' / difficulty
    info = json.loads((published / 'supp_info.json').read_text())[key]
    properties, units = info['properties'][1:], info['si-derived_units'][1:]
    if len(properties) != len(units):
        raise ValueError('SRSD unit and property metadata lengths differ')
    variable_units = [unit for prop, unit in zip(properties, units)
                      if prop.split(',')[0].strip() == 'V']
    base = np.loadtxt(published / 'train' / f'{key}.txt', max_rows=32, ndmin=2)[:, :-1]
    if len(variable_units) != base.shape[1]:
        raise ValueError('SRSD variable-unit metadata does not match source inputs')
    if not task.startswith('srsd-dummy-variable-'):
        return variable_units
    dummy = raw / 'srsd-dummy-variable-data' / difficulty
    augmented = np.loadtxt(dummy / 'train' / f'{key}.txt', max_rows=32, ndmin=2)[:, :-1]
    if len(base) != len(augmented):
        raise ValueError('Paired SRSD input rows do not match')
    result = [None] * augmented.shape[1]
    matched = set()
    for original, unit in enumerate(variable_units):
        matches = [j for j in range(augmented.shape[1])
                   if np.array_equal(base[:, original], augmented[:, j])]
        if len(matches) != 1 or matches[0] in matched:
            raise ValueError('SRSD dummy input mapping is ambiguous or missing')
        j = matches[0]
        for split in ('val', 'test'):
            source = np.loadtxt(published / split / f'{key}.txt', max_rows=32, ndmin=2)
            paired = np.loadtxt(dummy / split / f'{key}.txt', max_rows=32, ndmin=2)
            if not np.array_equal(source[:, original], paired[:, j]):
                raise ValueError('SRSD dummy input mapping differs across splits')
        matched.add(j)
        result[j] = unit
    return result
