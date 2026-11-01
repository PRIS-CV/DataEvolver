"""Diagnostic evidence sufficiency under immutable coverage/preservation rules.

Declarations describe admissible evidence and provenance requirements; they are
not witnesses, qualifications or permissions. A conflict among supplied sets
does not establish that the real task is impossible. No coordinates are emitted.
"""
import copy
from collections import Counter

from dataevolver.dataset.contracts import payload_sha256


def validate_evidence_declarations(value):
    if (set(value) != {'schema_version', 'entries'}
            or value['schema_version'] != 'dataevolver.evidence_requirement_registry.v1'
            or not isinstance(value['entries'], list) or len(value['entries']) > 64):
        raise ValueError('a bounded versioned evidence requirement registry is required')
    ids = set()
    for entry in value['entries']:
        if (set(entry) != {'reason', 'evidence_type', 'provenance_rules', 'declaration_status',
                          'implementation_qualification', 'grants_authority'}
                or not isinstance(entry['reason'], str) or not 1 <= len(entry['reason']) <= 128
                or entry['reason'] in ids or not isinstance(entry['evidence_type'], str)
                or not 1 <= len(entry['evidence_type']) <= 128
                or entry['declaration_status'] != 'DECLARED'
                or entry['implementation_qualification'] != 'NOT_ASSESSED'
                or entry['grants_authority'] is not False
                or not isinstance(entry['provenance_rules'], list) or not entry['provenance_rules']
                or len(entry['provenance_rules']) > 12
                or any(not isinstance(x, str) or not 1 <= len(x) <= 512 for x in entry['provenance_rules'])
                or len(set(entry['provenance_rules'])) != len(entry['provenance_rules'])):
            raise ValueError('evidence declarations cannot supply witnesses or execution permissions')
        ids.add(entry['reason'])
    return copy.deepcopy(value)


def compile_sufficiency_receipt(feedback, records, declarations, *, request_binding):
    registry = validate_evidence_declarations(declarations)
    if (feedback['schema_version'] != 'dataevolver.training_candidate_feedback.v1'
            or feedback['scope'] != 'registered_production_diagnostic'
            or feedback['candidate_status'] != 'unsupported'
            or feedback['image_submission_allowed'] is not False
            or feedback['candidate_adoption_allowed'] is not False
            or feedback['algorithm_defect_established'] is not False
            or records['schema_version'] != 'dataevolver.production_diagnostic_records.v1'
            or records['geometry_emitted'] is not False or records['new_permissions'] is not False
            or records['candidate_output_payload_sha256'] != feedback['candidate_output_payload_sha256']):
        raise ValueError('sufficiency receipt needs the exact blocked production diagnostic')
    if (request_binding['feedback_sha256'] != payload_sha256(feedback)
            or request_binding['request']['request']['implementation_kind'] != 'artifact_contract'
            or request_binding['frozen'] is not True or request_binding['replay'] is not True
            or request_binding['registered_effect_verified'] is not True
            or request_binding['native_and_receipt_verified'] is not True):
        raise ValueError('sufficiency contract was not requested from this actual runtime feedback')
    entries = {row['reason']: row for row in registry['entries']}
    gaps, conflicts = [], []
    if len(records['records']) > 4096:
        raise ValueError('too many diagnostic rows for this bounded receipt')
    for section in ('unresolved', 'conflicts'):
        rows = [row for row in records['records'] if row['section'] == section]
        if (len(rows) != feedback[section]['row_count']
                or [row['index'] for row in rows] != list(range(len(rows)))
                or dict(Counter(row['reason'] for row in rows)) != feedback[section]['reason_row_counts']):
            raise ValueError('diagnostic rows or overlap counts changed')
        for row in rows:
            ref = row['reference']
            if (ref['execution_sha256'] != feedback['origin']['execution_sha256']
                    or ref['json_pointer'] != '/output/'+section+'/'+str(row['index'])
                    or len(ref['row_payload_sha256']) != 64):
                raise ValueError('diagnostic row lost its original captured reference')
            declaration = entries.get(row['reason'])
            finding = {'reason': row['reason'], 'reference': ref,
                'original_identifiers': row['original_identifiers'],
                'witness_requirement': declaration, 'witness_satisfied': False,
                'declaration_missing': declaration is None}
            (gaps if section == 'unresolved' else conflicts).append(finding)
    if len(gaps)+len(conflicts) != len(records['records']):
        raise ValueError('unknown diagnostic record section')
    cpu_findings = records['cpu_findings']
    if len(cpu_findings) != 1:
        raise ValueError('one explicit declared-set feasibility check is required')
    finding = cpu_findings[0]
    if (finding['kind'] != 'declared_coverage_intersects_preservation'
            or finding['scope'] != 'supplied_declared_sets_only'
            or finding['task_feasibility'] != 'indeterminate'
            or finding['coverage_is_edit_authority'] is not False
            or any(type(finding[k]) is not int or finding[k] < 0 for k in (
                'declared_coverage_unique_pixels', 'preservation_unique_pixels', 'intersection_unique_pixels'))
            or finding['intersection_unique_pixels'] > min(finding['declared_coverage_unique_pixels'],
                                                           finding['preservation_unique_pixels'])):
        raise ValueError('declared-set finding cannot assert unobserved task feasibility')
    unknown = any(row['declaration_missing'] for row in gaps+conflicts)
    status = ('conflicting' if finding['intersection_unique_pixels'] or conflicts else
              'indeterminate' if unknown or not gaps else 'evidence_insufficient')
    return {'schema_version': 'dataevolver.evidence_sufficiency_receipt.v1',
        'scope': 'registered_production_diagnostic_only', 'status': status,
        'task_feasibility': 'indeterminate', 'actual_task_infeasibility_established': False,
        'original_candidate_status': 'unsupported', 'evidence_gaps': gaps,
        'captured_conflicts': conflicts, 'cpu_constraint_findings': cpu_findings,
        'row_counts_may_overlap': True, 'diagnostic_records_sha256': payload_sha256(records),
        'feedback_sha256': payload_sha256(feedback), 'declarations_sha256': payload_sha256(registry),
        'request_sha256': request_binding['request_sha256'], 'original_execution': feedback['origin'],
        'input_artifact': records['input_artifact'], 'input_payload_sha256': records['input_payload_sha256'],
        'implementation_qualification': 'NOT_ASSESSED', 'measured_method_gain': 'NOT_ASSESSED',
        'positive_production_behavior_verified': False, 'production_qualification_passed': False,
        'image_submission_allowed': False, 'candidate_adoption_allowed': False,
        'geometry_emitted': False, 'new_permissions': False, 'new_candidate_executions': 0,
        'new_model_calls': 0, 'new_accepted_records': 0, 'algorithm_defect_established': False}
