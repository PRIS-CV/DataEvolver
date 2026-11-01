"""Operational evidence assessment over independently qualified host receipts.

The compiler assigns no pixels. Positive findings require a separate completed
production qualification chain and a host-verified record for the exact source
and required scope. Existing local observations remain visible in every result.
"""
import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from dataevolver.dataset.contracts import ArtifactRef, file_sha256, payload_sha256
from .boundary_witness import WITNESS_SCHEMA, _object, _HASH, _TEXT, _REF
from .production_qualification import (
    ControllerProductionQualificationAuthority, ProductionQualificationExecution,
    assess_production_qualification)


CLAIMS = ('boundary_coverage', 'visible_apertures', 'background_preservation', 'effect_ownership')
EVIDENCE_SCHEMA = _object({
    'schema_version': {'const': 'dataevolver.boundary_operational_evidence.v1'},
    'source': _REF, 'witness_sha256': _HASH, 'original_contract_sha256': _HASH,
    'original_constraints_sha256': _HASH, 'immutable_correspondence_sha256': _HASH,
    'claims': {'type': 'array', 'minItems': 1, 'maxItems': 1024, 'items': _object({
        'subject_id': _TEXT, 'target_instance_id': _TEXT,
        'claim': {'enum': list(CLAIMS)}, 'finding': {'enum': ['supported', 'rejected', 'unresolved']},
        'scope': {'enum': ['full_required_scope', 'partial', 'unassessed']}, 'scope_sha256': _HASH,
        'basis': {'enum': ['independent_pixel_evidence', 'independent_effect_evidence']},
        'reason': _TEXT, 'evidence': {'type': 'array', 'minItems': 1, 'maxItems': 32, 'items': _REF},
        'resolves_prior_assessments': {'type': 'array', 'maxItems': 64, 'items': _HASH}})}})


def required_scope_sha256(witness, subject):
    return payload_sha256({'source': witness['source'], 'subject_id': subject['subject_id'],
        'target_instance_id': subject['target_instance_id'],
        'original_contract_sha256': witness['provenance']['original_contract_sha256'],
        'original_constraints_sha256': witness['provenance']['original_constraints_sha256']})


class RegisteredBoundaryQualificationEvidence:
    """Read the existing engineering/independent Controller authorities only.

    A package report, fixture signature, raw model output or caller-supplied
    JSON cannot implement this port. Evidence semantics remain the responsibility
    of the host production verifier; this reader checks its committed closure.
    """
    def __init__(self, *, execution=None, authorities=None):
        self.execution = execution
        self.authorities = dict(authorities or {})
        if self.authorities:
            if (type(execution) is not ProductionQualificationExecution
                    or set(self.authorities) != {'engineering', 'independent'}
                    or any(type(a) is not ControllerProductionQualificationAuthority
                           for a in self.authorities.values())):
                raise ValueError('boundary evidence needs both original production Controller authorities')
        elif execution is not None:
            raise ValueError('a production execution needs both qualified stage authorities')
        self.binding = self.identity()

    def identity(self):
        return {'schema_version': 'dataevolver.registered_boundary_qualification_evidence.v1',
            'reader_sha256': file_sha256(Path(__file__)),
            'execution_sha256': self.execution.compute_hash() if self.execution else None,
            'authorities': {stage: a.base.identity() for stage, a in self.authorities.items()}}

    def read(self, witness):
        if self.identity() != self.binding:
            raise ValueError('boundary qualification evidence authority changed')
        if not self.authorities:
            return {'qualified_chain': None, 'records': [], 'artifact_hashes': {},
                    'missing_prerequisites': ['independent_production_qualification_authority'],
                    'new_model_calls': 0, 'new_operator_executions': 0}
        chain = assess_production_qualification(self.execution, self.authorities)
        records, hashes = [], {}
        for stage, authority in self.authorities.items():
            controller = authority.base.controller
            event = next(e for e in controller.store.events.read() if e.event_type == 'ACTION_COMMITTED'
                         and e.payload['call_id'] == authority.base.call_id)
            closed = event.payload['effect_verification']['artifact_hashes']
            abi = event.payload['receipt']['stage_result']['production_evidence']
            manifest_ref = ArtifactRef.from_dict(abi['artifacts']['quality_checks'])
            path = authority.store.resolve(manifest_ref)
            if closed.get(str(path)) != manifest_ref.sha256:
                raise ValueError('boundary evidence manifest is outside the committed host closure')
            manifest = json.loads(path.read_text())
            if set(manifest) != {'role', 'cases'} or manifest['role'] != 'quality_checks':
                raise ValueError('boundary evidence must use the registered production role manifest')
            sources = {s['family_id']: s['artifact'] for s in self.execution.execution_protocol().sources
                       if s['family_id'] in self.execution.execution_protocol().families(stage)}
            seen = set()
            for case in manifest['cases']:
                if (set(case) != {'family_id', 'artifact', 'receipt_sha256'}
                        or case['family_id'] not in sources or case['family_id'] in seen):
                    raise ValueError('boundary evidence family inventory changed')
                seen.add(case['family_id'])
                ref = ArtifactRef.from_dict(case['artifact']); file = authority.store.resolve(ref)
                if closed.get(str(file)) != ref.sha256:
                    raise ValueError('boundary evidence record is outside the committed host closure')
                record = json.loads(file.read_text())
                # Other sources qualify the method; they cannot label this source.
                if sources[case['family_id']] != witness['source']:
                    continue
                _validate_record(record)
                if record['source'] != sources[case['family_id']]:
                    raise ValueError('boundary evidence belongs to a different original source')
                for claim in record['claims']:
                    for raw in claim['evidence']:
                        evidence = ArtifactRef.from_dict(raw); location = authority.store.resolve(evidence)
                        if closed.get(str(location)) != evidence.sha256:
                            raise ValueError('pixel or effect evidence lacks committed independent provenance')
                records.append({'stage': stage, 'family_id': case['family_id'], 'record': record,
                    'artifact': ref.to_dict(), 'host_receipt_sha256': case['receipt_sha256']})
            if seen != set(sources):
                raise ValueError('boundary evidence omitted a predeclared production family')
            hashes.update(closed)
        return {'qualified_chain': chain, 'records': records, 'artifact_hashes': hashes,
                'missing_prerequisites': [], 'new_model_calls': 0, 'new_operator_executions': 0}


def _validate_record(value):
    try:
        Draft202012Validator(EVIDENCE_SCHEMA).validate(value)
        ArtifactRef.from_dict(value['source'])
        for row in value['claims']:
            for ref in row['evidence']:
                ArtifactRef.from_dict(ref)
            if len(set(row['resolves_prior_assessments'])) != len(row['resolves_prior_assessments']):
                raise ValueError('duplicate prior assessment resolution')
    except (ValidationError, KeyError, TypeError) as exc:
        raise ValueError('invalid typed independent boundary evidence') from exc


def qualify_boundary_witness(witness, consumed):
    """Compute claim findings from a host-verified evidence snapshot.

    This pure function supports independent engineering checks. Production
    callers must obtain consumed from RegisteredBoundaryQualificationEvidence;
    a successful pure-function fixture establishes no production qualification.
    """
    try:
        Draft202012Validator(WITNESS_SCHEMA).validate(witness)
    except ValidationError as exc:
        raise ValueError('operational qualification requires the typed original witness') from exc
    mapping = {s['subject_id']: s for s in witness['immutable_correspondence']}
    if len(mapping) != len(witness['immutable_correspondence']):
        raise ValueError('boundary qualification cannot merge original instances')
    prior = {row['assessment']['subject_id']: row for row in witness['findings']}
    if len(prior) != len(witness['findings']) or not set(prior) <= set(mapping):
        raise ValueError('boundary witness observed correspondence changed')
    expected = {'source': witness['source'], 'witness_sha256': payload_sha256(witness),
        'original_contract_sha256': witness['provenance']['original_contract_sha256'],
        'original_constraints_sha256': witness['provenance']['original_constraints_sha256'],
        'immutable_correspondence_sha256': payload_sha256(witness['immutable_correspondence'])}
    chain = consumed['qualified_chain']
    if chain is None and consumed['records']:
        raise ValueError('unqualified reports cannot supply operational boundary claims')
    if chain is not None and (chain['production_engineering_qualification'] != 'VERIFIED'
            or chain['production_independent_qualification'] != 'VERIFIED'
            or chain['production_adoption_allowed'] is not False
            or len({p['campaign_id'] for p in chain['qualification_evidence'].values()}) != 2):
        raise ValueError('boundary claims need distinct actual qualified production stages')
    candidates = {}
    disallowed = {witness['source']['sha256'],
                  *(s['immutable_mask']['sha256'] for s in mapping.values()),
                  *(r['mask_comparison']['native_proposal_mask']['sha256'] for r in witness['findings'])}
    for entry in consumed['records']:
        value = entry['record']; _validate_record(value)
        if any(value[k] != v for k, v in expected.items()):
            raise ValueError('independent boundary evidence source, constraints or witness differs')
        for row in value['claims']:
            sid = row['subject_id']
            if (sid not in mapping or row['target_instance_id'] != mapping[sid]['target_instance_id']
                    or row['scope_sha256'] != required_scope_sha256(witness, mapping[sid])):
                raise ValueError('independent evidence changed the required original instance scope')
            if (row['basis'] != ('independent_effect_evidence' if row['claim'] == 'effect_ownership'
                                else 'independent_pixel_evidence')
                    or any(ref['sha256'] in disallowed for ref in row['evidence'])):
                raise ValueError('source, immutable masks and proposals are not independent pixel or effect evidence')
            candidates.setdefault((sid, row['claim']), []).append({
                'claim': row, 'record_artifact': entry['artifact'], 'stage': entry['stage'],
                'family_id': entry['family_id'], 'host_receipt_sha256': entry['host_receipt_sha256']})
    findings = []
    for sid, subject in mapping.items():
        for kind in CLAIMS:
            values = candidates.get((sid, kind), [])
            missing = list(consumed['missing_prerequisites'])
            full = [r for r in values if r['claim']['scope'] == 'full_required_scope']
            original = prior.get(sid)
            needs_resolution = original and (
                (kind in ('boundary_coverage', 'visible_apertures') and original['assessment'][kind] != 'supported')
                or (kind == 'background_preservation' and original['assessment']['preservation_conflict'] != 'none_observed'))
            if needs_resolution:
                digest = original['assessment_reference']['row_payload_sha256']
                full = [r for r in full if digest in r['claim']['resolves_prior_assessments']]
                if not full:
                    missing.append('prior_registered_assessment_not_independently_resolved')
            if not full:
                missing.append('independent_effect_evidence' if kind == 'effect_ownership' else 'independent_pixel_evidence')
                if values:
                    missing.append('complete_required_scope')
            statuses = {r['claim']['finding'] for r in full}
            if not full or 'unresolved' in statuses or len(statuses) != 1:
                status = 'unresolved'
                if len(statuses) > 1:
                    missing.append('conflicting_independent_evidence')
                if 'unresolved' in statuses:
                    missing.append('independent_assessment_unresolved')
            else:
                status = next(iter(statuses))
            findings.append({'subject_id': sid, 'target_instance_id': subject['target_instance_id'],
                'claim': kind, 'finding': status, 'missing_prerequisites': sorted(set(missing)),
                'original_observation_status': 'OBSERVED_UNQUALIFIED' if original else 'NOT_ASSESSED',
                'evidence': full, 'original_assessment': copy.deepcopy(original['assessment']) if original else None})
    status = ('REJECTED' if any(r['finding'] == 'rejected' for r in findings) else
              'SUPPORTED' if all(r['finding'] == 'supported' for r in findings) else 'INSUFFICIENT_EVIDENCE')
    return {'schema_version': 'dataevolver.boundary_operational_qualification.v1',
        'status': status, 'source': witness['source'], 'witness_sha256': payload_sha256(witness),
        'original_contract_sha256': expected['original_contract_sha256'],
        'original_constraints_sha256': expected['original_constraints_sha256'],
        'immutable_correspondence_sha256': expected['immutable_correspondence_sha256'],
        'findings': findings, 'prior_observations': copy.deepcopy(witness['findings']),
        'original_unassessed_subject_ids': witness['unassessed_subject_ids'],
        'qualified_production_chain': copy.deepcopy(chain),
        'production_qualification_passed': False, 'image_submission_allowed': False,
        'candidate_adoption_allowed': False, 'full_task_completion': False,
        'algorithm_defect_established': False, 'new_model_calls': 0, 'new_probe_executions': 0,
        'new_accepted_records': 0, 'method_improvement_verified': False}
