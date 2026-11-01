"""Pure aggregation and exact-membership guards for image-generation evaluation.

Scorers supply real native receipts/probabilities. These helpers never replace a
probability with text, infer missing answers, or omit an expected image.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .contracts import file_sha256, payload_sha256
from dataevolver.harness.state_store import atomic_write_json


def spatial_answers(raw, *, questions=10):
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if len(lines) != questions or any(not re.match(r'^[A-Z](?:[:.)\s]|$)', line) for line in lines):
        raise ValueError('SpatialGenEval requires ordered valid answers for every question')
    return [line[0] for line in lines]


def spatial_score(rollouts, answers, *, valid_rollouts=5, count=4):
    if (len(answers) != 10 or len(rollouts) != valid_rollouts or count != 4
            or valid_rollouts != 5 or any(len(r) != 10 for r in rollouts)):
        raise ValueError('incomplete or changed SpatialGenEval protocol')
    passed = [sum(r[i] == answer for r in rollouts) >= count for i, answer in enumerate(answers)]
    return {'true-or-false': passed, 'avg_acc': sum(passed)/10,
            'basic_acc': sum(passed[:2])/2, 'spatial_acc': sum(passed[2:])/8,
            'Position': float(passed[2]), 'Occlusion': float(passed[7])}


def soft_tifa(atoms):
    """Both official aggregates consume the same persisted answer probabilities."""
    if not atoms or any(not row for row in atoms):
        raise ValueError('Soft-TIFA requires every expected prompt and atom')
    flat = [p for row in atoms for p in row]
    if any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 or p > 1 for p in flat):
        raise ValueError('Soft-TIFA needs actual finite answer-token probabilities')
    am = [sum(row)/len(row) for row in atoms]
    gm = [0. if 0 in row else math.exp(sum(math.log(p) for p in row)/len(row)) for row in atoms]
    return {'soft_tifa_am': 100*sum(am)/len(am), 'soft_tifa_gm': 100*sum(gm)/len(gm)}


class EvaluationJournal:
    """One immutable expected-ID manifest; results are keyed, never appended twice."""
    def __init__(self, root, manifest):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        ids = [r['id'] for r in manifest['items']]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('evaluation manifest needs unique expected IDs')
        self.manifest = manifest
        self.binding = payload_sha256(manifest)
        path = self.root/'manifest.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError('evaluation manifest changed')
        if not path.exists():
            atomic_write_json(path, manifest)

    def record(self, item_id, *, image_path, image_sha256, result, evidence):
        expected = {r['id']: r for r in self.manifest['items']}
        if item_id not in expected or expected[item_id]['image_sha256'] != image_sha256:
            raise ValueError('evaluation ID/image association changed')
        if file_sha256(Path(image_path)) != image_sha256:
            raise ValueError('evaluation image bytes changed')
        if not evidence or evidence.get('status') != 'VERIFIED':
            raise ValueError('evaluation needs confirmed native evidence')
        value = {'binding': self.binding, 'id': item_id, 'image_sha256': image_sha256,
                 'result': result, 'evidence': evidence}
        path = self.root/(payload_sha256(item_id)+'.result.json')
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError('evaluation result cannot be replaced after confirmation')
        else:
            atomic_write_json(path, value)
        return value

    def complete(self):
        rows = [json.loads(p.read_text()) for p in self.root.glob('*.result.json')]
        wanted = {r['id'] for r in self.manifest['items']}
        actual = [r['id'] for r in rows]
        if (len(actual) != len(set(actual)) or set(actual) != wanted
                or any(r['binding'] != self.binding for r in rows)):
            raise ValueError('evaluation incomplete; failed/missing IDs cannot leave the denominator')
        return {r['id']: r for r in rows}


def inference_costs(*, model_count, spatial_prompts, atom_lengths, generation_seeds,
                    training_jobs, checkpoints, development_prompts, development_seeds=1):
    values = [model_count, spatial_prompts, generation_seeds, training_jobs, checkpoints,
              development_prompts, development_seeds, *atom_lengths]
    if any(type(v) is not int or v < 1 for v in values):
        raise ValueError('complete-path counts must be positive integers')
    return {'final_images': model_count*generation_seeds*(spatial_prompts+len(atom_lengths)),
            'checkpoint_images': training_jobs*checkpoints*development_prompts*development_seeds,
            'spatial_valid_calls': model_count*generation_seeds*spatial_prompts*5,
            'spatial_outer_attempt_ceiling': model_count*generation_seeds*spatial_prompts*25,
            'geneval2_atom_calls': model_count*generation_seeds*sum(atom_lengths)}
