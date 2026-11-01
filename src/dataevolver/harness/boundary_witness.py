"""Provenance-bound local observations, with explicit unqualified pixel evidence.

This compiler consumes verified registered feedback. It neither segments an
image nor invents pixel labels. Its validator checks conformance separately
from the evidence required for operational qualification.
"""
import copy

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from dataevolver.dataset.contracts import ArtifactRef, payload_sha256
from .boundary_evidence import QUERY_SCHEMA, REVIEW_SCHEMA


def _object(properties):
    return {'type': 'object', 'additionalProperties': False,
            'properties': properties, 'required': list(properties)}


_HASH = {'type': 'string', 'pattern': '^[0-9a-f]{64}$'}
_TEXT = {'type': 'string', 'minLength': 1}
_FALSE = {'const': False}
_NOT_ASSESSED = {'const': 'NOT_ASSESSED'}
_REF = _object({'schema_version': {'const': ArtifactRef.schema}, 'store': _TEXT,
                'path': _TEXT, 'sha256': _HASH, 'media_type': _TEXT})
_CORRESPONDENCE = _object({'subject_id': _TEXT, 'target_instance_id': _TEXT, 'immutable_mask': _REF})
_REFERENCE = _object({'report_sha256': _HASH, 'json_pointer': _TEXT,
                      'row_payload_sha256': _HASH, 'receipt_sha256': _HASH})
_ASSESSMENT = REVIEW_SCHEMA['properties']['assessments']['items']
_DIAGNOSTIC = _object({
    'subject_id': _TEXT, 'target_instance_id': _TEXT, 'immutable_mask': _REF,
    'native_proposal_mask': _REF,
    **{k: {'type': 'integer', 'minimum': 0} for k in (
        'proposal_unique_pixels', 'added_vs_immutable_pixels', 'removed_vs_immutable_pixels',
        'immutable_background_included_pixels')},
    'all_immutable_background_was_preserved': {'type': 'boolean'},
    'proposal_is_edit_authority': _FALSE, 'boundary_witness_qualification': _NOT_ASSESSED,
    'aperture_truth_established_by_mask_comparison': _FALSE})
_PROVENANCE = _object({
    **{k: _HASH for k in ('feedback_sha256', 'runtime_request_sha256', 'original_contract_sha256',
        'original_constraints_sha256', 'original_snapshot_sha256', 'frozen_contract_sha256',
        'probe_receipt_sha256', 'report_sha256', 'segmentation_request_sha256',
        'segmentation_receipt_sha256', 'visual_request_sha256', 'visual_receipt_sha256')},
    'segmentation_request_id': _TEXT, 'visual_request_id': _TEXT,
    'resource_deadline_epoch': {'type': 'number'}})

WITNESS_SCHEMA = _object({
    'schema_version': {'const': 'dataevolver.boundary_witness.v1'},
    'scope': {'const': 'registered_local_observations_only'}, 'source': _REF,
    'provenance': _PROVENANCE,
    'immutable_correspondence': {'type': 'array', 'minItems': 1, 'maxItems': 256, 'items': _CORRESPONDENCE},
    'runtime_scope': QUERY_SCHEMA,
    'findings': {'type': 'array', 'minItems': 1, 'maxItems': 2, 'items': _object({
        'assessment': _ASSESSMENT, 'assessment_reference': _REFERENCE,
        'mask_comparison': _DIAGNOSTIC, 'comparison_reference': _REFERENCE})},
    'unassessed_subject_ids': {'type': 'array', 'maxItems': 256, 'items': _TEXT},
    'pixel_evidence': _object({k: _NOT_ASSESSED for k in ('object', 'background', 'uncertain')}),
    'qualification': _object({
        'boundary_witness': _NOT_ASSESSED, 'effect_ownership': _NOT_ASSESSED,
        'production_qualification_passed': _FALSE, 'image_submission_allowed': _FALSE,
        'candidate_adoption_allowed': _FALSE, 'full_task_completion': _FALSE}),
    'mask_comparison_establishes_pixel_truth': _FALSE,
    'local_visual_support_establishes_pixel_truth': _FALSE,
    'algorithm_defect_established': _FALSE})


def compile_boundary_witness(feedback, context, *, request_binding):
    """Compile exact authenticated observations, including their original caveats."""
    r = request_binding
    if (feedback.get('schema_version') != 'dataevolver.boundary_probe_feedback.v1'
            or feedback.get('proposal_status') != 'OBSERVED_UNQUALIFIED'
            or feedback.get('image_submission_allowed') is not False
            or feedback.get('candidate_adoption_allowed') is not False
            or feedback.get('effect_attribution') != 'NOT_ASSESSED'
            or feedback.get('boundary_witness_qualification') != 'NOT_ASSESSED'
            or r['feedback_sha256'] != payload_sha256(feedback)
            or r['request']['request']['implementation_kind'] != 'artifact_contract'
            or any(r.get(k) is not True for k in (
                'frozen', 'replay', 'registered_effect_verified', 'native_and_receipt_verified'))):
        raise ValueError('boundary witness needs this actual frozen runtime artifact request')
    source = ArtifactRef.from_dict(context['source']).to_dict()
    masks = {x['label'].removeprefix('instance mask '): x['artifact'] for x in context['media']
             if x['label'].startswith('instance mask ')}
    originals = context['original_segment_parameters']['subjects']
    mapping = {x['subject_id']: x['target_instance_id'] for x in originals}
    selected = context['selected_subject_ids']
    if (len(mapping) != len(originals) or not selected or len(set(selected)) != len(selected)
            or not set(selected) <= set(mapping) or set(masks) != set(selected)):
        raise ValueError('immutable correspondence or selected mask inventory differs')
    correspondence = [{'subject_id': sid, 'target_instance_id': mapping[sid],
                       'immutable_mask': ArtifactRef.from_dict(masks[sid]).to_dict()} for sid in selected]
    visual = feedback['runtime_visual_assessment']['assessments']
    diagnostic = feedback['mask_diagnostics']
    by_id = {x['subject_id']: (i, x) for i, x in enumerate(diagnostic)}
    query_ids = [x['subject_id'] for x in feedback['runtime_query']['subjects']]
    if (len(by_id) != len(diagnostic) or len(visual) != len(by_id)
            or len(set(query_ids)) != len(query_ids) or set(query_ids) != set(by_id)
            or {x['subject_id'] for x in visual} != set(by_id)):
        raise ValueError('registered local scope or finding correspondence differs')
    origin = feedback['origin']
    if origin['resource_deadline_epoch'] != context['resource_deadline_epoch']:
        raise ValueError('boundary witness cannot extend the original resource deadline')
    def reference(pointer, row, receipt):
        return {'report_sha256': origin['report_sha256'], 'json_pointer': pointer,
                'row_payload_sha256': payload_sha256(row), 'receipt_sha256': receipt}
    findings = []
    for index, assessment in enumerate(visual):
        sid = assessment['subject_id']; di, comparison = by_id[sid]
        if (sid not in selected or assessment['target_instance_id'] != mapping[sid]
                or comparison['target_instance_id'] != mapping[sid] or comparison['immutable_mask'] != masks[sid]):
            raise ValueError('registered findings differ from immutable instance correspondence')
        ArtifactRef.from_dict(comparison['native_proposal_mask'])
        findings.append({'assessment': assessment,
            'assessment_reference': reference('/visual_review/receipt/output/assessments/'+str(index),
                                              assessment, feedback['visual_receipt_sha256']),
            'mask_comparison': comparison,
            'comparison_reference': reference('/mask_diagnostics/'+str(di), comparison,
                                              feedback['segmentation_receipt_sha256'])})
    value = {'schema_version': 'dataevolver.boundary_witness.v1',
        'scope': 'registered_local_observations_only', 'source': source,
        'provenance': {'feedback_sha256': payload_sha256(feedback), 'runtime_request_sha256': r['request_sha256'],
            'original_contract_sha256': context['original_contract_sha256'],
            'original_constraints_sha256': payload_sha256(context['original_task_constraints']),
            'original_snapshot_sha256': context['original_snapshot']['sha256'],
            'frozen_contract_sha256': origin['frozen_contract_sha256'],
            'probe_receipt_sha256': origin['receipt_sha256'], 'report_sha256': origin['report_sha256'],
            **{kind+'_'+suffix: (payload_sha256(feedback[kind+'_request']) if suffix == 'request_sha256'
                                else feedback[kind+'_request']['request_id'] if suffix == 'request_id'
                                else feedback[kind+'_receipt_sha256'])
               for kind in ('segmentation', 'visual') for suffix in ('request_id', 'request_sha256', 'receipt_sha256')},
            'resource_deadline_epoch': context['resource_deadline_epoch']},
        'immutable_correspondence': correspondence, 'runtime_scope': feedback['runtime_query'],
        'findings': findings, 'unassessed_subject_ids': [sid for sid in selected if sid not in by_id],
        'pixel_evidence': {k: 'NOT_ASSESSED' for k in ('object', 'background', 'uncertain')},
        'qualification': {'boundary_witness': 'NOT_ASSESSED', 'effect_ownership': 'NOT_ASSESSED',
            'production_qualification_passed': False, 'image_submission_allowed': False,
            'candidate_adoption_allowed': False, 'full_task_completion': False},
        'mask_comparison_establishes_pixel_truth': False,
        'local_visual_support_establishes_pixel_truth': False, 'algorithm_defect_established': False}
    try:
        Draft202012Validator(WITNESS_SCHEMA).validate(value)
    except ValidationError as exc:
        raise ValueError('registered boundary evidence violates the typed witness contract') from exc
    return copy.deepcopy(value)


def validate_boundary_witness(value, feedback, context, *, request_binding, available_receipts):
    """Rebuild findings from trusted evidence; no observational or model dispatch."""
    try:
        Draft202012Validator(WITNESS_SCHEMA).validate(value)
        expected = compile_boundary_witness(feedback, context, request_binding=request_binding)
        if value != expected:
            raise ValueError('source, scope, provenance or original findings changed')
    except (ValueError, KeyError, TypeError, ValidationError) as exc:
        return {'schema_version': 'dataevolver.boundary_witness_validation.v1',
            'contract_conformant': False, 'status': 'INVALID_EVIDENCE', 'reason': str(exc),
            'production_qualification_passed': False, 'image_submission_allowed': False,
            'candidate_adoption_allowed': False, 'new_model_calls': 0}
    missing = [kind+'_receipt' for kind in ('segmentation', 'visual')
               if available_receipts.get(feedback[kind+'_request']['request_id']) != feedback[kind+'_receipt_sha256']]
    missing += ['qualified_pixel_boundary_witness', 'independent_production_qualification', 'effect_ownership']
    conflicts = [row['assessment']['subject_id'] for row in value['findings']
                 if row['assessment']['preservation_conflict'] == 'observed']
    uncertain = [row['assessment']['subject_id'] for row in value['findings']
                 if 'uncertain' in row['assessment'].values()]
    return {'schema_version': 'dataevolver.boundary_witness_validation.v1',
        'contract_conformant': True, 'status': 'INSUFFICIENT_EVIDENCE',
        'witness_sha256': payload_sha256(value), 'missing_evidence': missing,
        'observed_preservation_conflicts': conflicts, 'uncertain_assessments': uncertain,
        'reason': 'Local visual observations remain unqualified; original reasons and scope are retained.',
        'boundary_witness_qualification': 'NOT_ASSESSED', 'effect_ownership': 'NOT_ASSESSED',
        'production_qualification_passed': False, 'image_submission_allowed': False,
        'candidate_adoption_allowed': False, 'full_task_completion': False,
        'algorithm_defect_established': False, 'new_model_calls': 0}
