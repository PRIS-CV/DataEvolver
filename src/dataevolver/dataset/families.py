"""The shared pixel/dHash/declared-source family definition, without selection."""
from __future__ import annotations

import copy
from collections import defaultdict

from .contracts import payload_sha256


def assign_families(rows, hamming_distance, *, check=lambda: None):
    if type(hamming_distance) is not int or not 0 <= hamming_distance <= 64:
        raise ValueError("dHash distance must be an integer between zero and 64")
    rows = copy.deepcopy(rows)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError("family input has duplicate record IDs")
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(b)] = find(a)

    exact, known = {}, {}
    for i, row in enumerate(rows):
        check()
        key = row['pixel_sha256']
        if key in exact:
            union(i, exact[key])
        exact[key] = i
        if row['source_family']:
            key = (row['source_dataset'], row['source_family'])
            if key in known:
                union(i, known[key])
            known[key] = i
        for j in range(i):
            if (int(row['dhash'], 16) ^ int(rows[j]['dhash'], 16)).bit_count() <= hamming_distance:
                union(i, j)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    for group in groups.values():
        identity = 'family-' + payload_sha256(sorted(r['id'] for r in group))[:24]
        for row in group:
            row['family_id'] = identity
    return rows


def duplicate_method(distance):
    return {'pixel_hash': 'sha256', 'perceptual_hash': 'dhash64',
            'hamming_distance': distance, 'semantic_duplicate_detection': 'not_performed'}


def image_fingerprint(image):
    """The same canonical RGB pixel SHA and dHash used by catalog intake."""
    import hashlib
    from .contracts import canonical_json
    from PIL import Image
    image = image.convert('RGB')
    pixels = image.convert('L').resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    bits = [pixels[y*9+x] > pixels[y*9+x+1] for y in range(8) for x in range(8)]
    return {'pixel_sha256': hashlib.sha256(canonical_json(image.size).encode()+image.tobytes()).hexdigest(),
            'dhash': f'{sum(int(b) << i for i, b in enumerate(bits)):016x}'}
