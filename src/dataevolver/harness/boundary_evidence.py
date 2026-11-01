"""Runtime-owned segmentation queries produce proposals, never edit authority."""
import copy
import json
import math
from pathlib import Path

from dataevolver.dataset.contracts import ArtifactRef, file_sha256, payload_sha256


QUERY_SCHEMA = {'type': 'object', 'additionalProperties': False,
    'required': ['question', 'subjects'], 'properties': {
        'question': {'type': 'string', 'minLength': 1, 'maxLength': 2000},
        'subjects': {'type': 'array', 'minItems': 1, 'maxItems': 2, 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': ['subject_id', 'description', 'box_xyxy_normalized', 'reason'],
            'properties': {'subject_id': {'type': 'string', 'minLength': 1},
                'description': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
                'box_xyxy_normalized': {'type': 'array', 'minItems': 4, 'maxItems': 4,
                    'items': {'type': 'number', 'minimum': 0, 'maximum': 1}},
                'reason': {'type': 'string', 'minLength': 1, 'maxLength': 2000}}}}}}

REVIEW_SCHEMA = {'type': 'object', 'additionalProperties': False,
    'required': ['assessments', 'reason'], 'properties': {
        'reason': {'type': 'string'}, 'assessments': {'type': 'array', 'items': {
            'type': 'object', 'additionalProperties': False,
            'required': ['subject_id', 'target_instance_id', 'identity_match', 'boundary_coverage',
                         'visible_apertures', 'preservation_conflict', 'reason'],
            'properties': {'subject_id': {'type': 'string'}, 'target_instance_id': {'type': 'string'},
                **{k: {'type': 'string', 'enum': ['supported', 'rejected', 'uncertain']} for k in
                   ('identity_match', 'boundary_coverage', 'visible_apertures')},
                'preservation_conflict': {'type': 'string', 'enum': ['observed', 'none_observed', 'uncertain']},
                'reason': {'type': 'string'}}}}}}


def compile_boundary_query(query, inputs):
    from jsonschema import validate
    from jsonschema.exceptions import ValidationError
    try:
        validate(query, QUERY_SCHEMA)
    except ValidationError as exc:
        raise ValueError('runtime boundary query is outside its frozen parameter schema') from exc
    old = inputs['original_segment_parameters']
    originals = {row['subject_id']: row for row in old['subjects']}
    chosen = []; seen = set()
    for row in query['subjects']:
        sid = row['subject_id']; box = row['box_xyxy_normalized']
        if (sid not in inputs['selected_subject_ids'] or sid in seen
                or not all(type(v) in (float, int) and math.isfinite(v) for v in box)
                or not box[0] < box[2] or not box[1] < box[3]):
            raise ValueError('probe needs distinct selected instances and finite observed boxes')
        seen.add(sid)
        chosen.append({'subject_id': sid, 'target_instance_id': originals[sid]['target_instance_id'],
                       'description': row['description'], 'box_xyxy_normalized': list(box)})
    # A new question/request ID alone cannot justify repeating completed SAM.
    if all(row == originals[row['subject_id']] for row in chosen):
        raise ValueError('identical completed segmentation parameters provide no new observation')
    return {**{k: old[k] for k in ('threshold', 'mask_threshold', 'min_box_iou')}, 'subjects': chosen}


def boundary_mask_diagnostics(inputs, query, result, store):
    import numpy as np
    from PIL import Image
    expected = {row['subject_id']: row for row in compile_boundary_query(query, inputs)['subjects']}
    found = result.get('instances', [])
    if len(found) != len(expected) or {row['subject_id'] for row in found} != set(expected):
        raise ValueError('probe changed the requested selected-instance mapping')
    refs = {row['label']: row['artifact'] for row in inputs['trial']['media']}
    def mask(raw):
        with Image.open(store.resolve(ArtifactRef.from_dict(raw))) as image:
            image.load(); a = np.asarray(image.convert('L'))
        if not np.isin(a, [0, 255]).all(): raise ValueError('boundary proposal must be an unmodified binary native mask')
        return a > 0
    rows = []
    for row in found:
        sid = row['subject_id']
        if row['target_instance_id'] != expected[sid]['target_instance_id']:
            raise ValueError('segmentation changed immutable target instance identity')
        old = mask(refs['instance mask '+sid]); proposed = mask(row['mask'])
        if old.shape != proposed.shape or not proposed.any(): raise ValueError('probe mask frame changed or is empty')
        rows.append({'subject_id': sid, 'target_instance_id': row['target_instance_id'],
            'native_proposal_mask': row['mask'], 'immutable_mask': refs['instance mask '+sid],
            'proposal_unique_pixels': int(proposed.sum()), 'added_vs_immutable_pixels': int((proposed & ~old).sum()),
            'removed_vs_immutable_pixels': int((old & ~proposed).sum()),
            'immutable_background_included_pixels': int((proposed & ~old).sum()),
            'all_immutable_background_was_preserved': not bool((proposed & ~old).any()),
            'proposal_is_edit_authority': False, 'boundary_witness_qualification': 'NOT_ASSESSED',
            'aperture_truth_established_by_mask_comparison': False})
    return rows


class BoundaryProbeDomain:
    def __init__(self, *, authority, observation_operator_id, admission):
        self.authority = authority
        self.binding = authority.identity()
        self.inputs = authority.probe_inputs()
        self.operator_id = observation_operator_id
        self.admission = copy.deepcopy(admission)
        a = self.admission
        from .operator_development import DevelopmentBudget
        budget = DevelopmentBudget.from_dict(a['budget'])
        original_deadline = self.inputs['contract']['production']['resource_deadline_epoch']
        feedback, _ = authority.read()
        increment = {'model_calls': 3, 'planner_calls': 1, 'operator_calls': 2,
                     'wall_seconds': 3600, 'output_bytes': 67108864, 'image_experiments': 0}
        if (a.get('schema_version') != 'dataevolver.boundary_probe_admission.v1'
                or a.get('feedback_sha256') != self.inputs['feedback_sha256']
                or a.get('new_planner_decisions') != 1 or a.get('max_segmentation_calls') != 1
                or a.get('max_visual_review_calls') != 1 or a.get('new_image_generations') != 0
                or a.get('resource_deadline_epoch') != original_deadline
                or budget.resource_deadline_epoch != original_deadline
                or any(budget.prior_usage[k] < feedback['lifecycle_total'][k] for k in increment)
                or any(budget.limits[k] != budget.prior_usage[k]+increment[k] for k in increment)):
            raise ValueError('one explicit, separately accounted boundary-probe admission is required')

    def config(self):
        return {'domain': 'registered-boundary-evidence-probe-v1', 'authority': self.binding,
                'operator_id': self.operator_id, 'admission': self.admission,
                'implementation_sha256': file_sha256(Path(__file__))}

    def context(self, state):
        if self.authority.identity() != self.binding: raise ValueError('boundary probe origin changed')
        feedback, _ = self.authority.read(); trial = self.inputs['trial']
        return {'task': 'Decide whether one distinct SAM3 boundary proposal and an independent runtime visual '
                'inspection could resolve a useful evidence gap on the actual media. Choose the selected IDs, '
                'descriptions and tight normalized boxes from observation yourself, or ABSTAIN if unsupported. '
                'Do not request the identical completed segmentation, invent labels, infer effect ownership '
                'from mask differences or change immutable masks. Native masks are unqualified proposals. '
                'Visible holes/background enclosed by an envelope remain ungranted. No generation, '
                'repair, support modification or adoption is allowed in this probe.',
            'allowed_actions': ['OBSERVE', 'ABSTAIN'], 'capability_requests_allowed': False,
            'available_evidence_ids': ['registered_production_diagnostic', 'original_instance_correspondence'],
            'evidence': {'registered_production_diagnostic': feedback,
                'original_instance_correspondence': {
                    'selected_ids': self.inputs['selected_subject_ids'],
                    'completed_segmentation_parameters': self.inputs['original_segment_parameters'],
                    'media_order': [x['label'] for x in self._media()],
                    'media_bindings': self._media(), 'source': trial['source'], 'candidate': trial['candidate']}},
            'allowed_observations': ['probe_instance_boundaries'], 'allowed_patches': [],
            'registered_observations': {'probe_instance_boundaries': {'parameters': QUERY_SCHEMA,
                'effect': 'one native SAM3 segmentation plus one independent runtime visual inspection; '
                          'all output is unqualified evidence, never editing permission'}},
            'compatible_experience_ids': [], 'resource_deadline_epoch': self.admission['resource_deadline_epoch']}

    def _media(self):
        trial = self.inputs['trial']
        return [*trial['media'][:3], *[row for row in trial['media'][3:]
                if row['label'].startswith(('reference ', 'instance mask '))]]

    def planner_artifacts(self, state):
        return [ArtifactRef.from_dict(row['artifact']) for row in self._media()]

    def compile_observation(self, decision, state):
        from .runtime_rsi import validate_decision
        validate_decision(decision, self.context(state))
        if not decision['evidence_ids']:
            raise ValueError('boundary probe must cite actual registered evidence')
        params = json.loads(decision['observation']['parameters_json'])
        compile_boundary_query(params, self.inputs)
        return self.operator_id, params

    def compile_intervention(self, decision, state):
        raise ValueError('boundary evidence probes cannot generate, repair or adopt images')
