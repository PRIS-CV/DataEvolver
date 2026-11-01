"""Import already produced evidence through qualified host provenance.

This port has no inference or provider-discovery backend. A production provider
is the existing pair of frozen Controller qualification authorities, including
their host-verified native audit. A caller declaration cannot establish trust.
"""
import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from dataevolver.dataset.contracts import ArtifactRef, file_sha256, payload_sha256
from .boundary_witness import _object, _HASH, _TEXT, _REF
from .boundary_qualification import (RegisteredBoundaryQualificationEvidence,
                                     qualify_boundary_witness)


PROVIDER_SCHEMA = _object({
    'schema_version': {'const': 'dataevolver.independent_evidence_provider.v1'},
    'provider_id': _TEXT, 'trust_profile_sha256': _HASH,
    'execution_contract_sha256': _HASH, 'source': _REF,
    'witness_sha256': _HASH, 'original_contract_sha256': _HASH,
    'original_constraints_sha256': _HASH, 'immutable_correspondence_sha256': _HASH,
    'record': _REF, 'host_receipt_sha256': _HASH,
    'claims': {'type': 'array', 'minItems': 1, 'maxItems': 1024, 'items': _object({
        'claim_sha256': _HASH,
        'independence': {'enum': ['independent', 'dependent', 'unassessed']},
        'authority_scope': {'enum': ['authorized', 'unavailable']},
        'reason': _TEXT,
        'production_receipts': {'type': 'array', 'minItems': 1, 'maxItems': 32, 'items': _REF},
        'assessment_dependencies': {'type': 'array', 'maxItems': 1024, 'items': _HASH}})}})


def _select_provider_claims(witness, entry, provenance, *, execution_sha256, trust_profile_sha256):
    """Pure validation for host-bound records; fixtures establish interface behavior only."""
    try:
        Draft202012Validator(PROVIDER_SCHEMA).validate(provenance)
    except ValidationError as exc:
        raise ValueError('invalid independent provider provenance') from exc
    record = entry['record']
    expected = {k: record[k] for k in ('source', 'witness_sha256', 'original_contract_sha256',
        'original_constraints_sha256', 'immutable_correspondence_sha256')}
    expected.update(record=entry['artifact'], host_receipt_sha256=entry['host_receipt_sha256'],
                    execution_contract_sha256=execution_sha256, trust_profile_sha256=trust_profile_sha256)
    if any(provenance[k] != v for k, v in expected.items()):
        raise ValueError('provider provenance changed source, record, scope or external authority')
    prior = {witness['source']['sha256'], payload_sha256(witness),
             *(s['immutable_mask']['sha256'] for s in witness['immutable_correspondence'])}
    for finding in witness['findings']:
        prior.add(finding['assessment_reference']['row_payload_sha256'])
        prior.add(finding['mask_comparison']['native_proposal_mask']['sha256'])
        # The original witness contains the exact local evidence bindings.
        def collect(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    if 'sha256' in k and isinstance(v, str):
                        prior.add(v)
                    else:
                        collect(v)
            elif isinstance(value, list):
                for v in value:
                    collect(v)
        collect(finding)
    claims = {payload_sha256(row): row for row in record['claims']}
    if len(claims) != len(record['claims']):
        raise ValueError('duplicate provider claim records')
    seen, accepted, gaps, receipts = set(), [], [], []
    for row in provenance['claims']:
        key = row['claim_sha256']
        if key not in claims or key in seen:
            raise ValueError('provider omitted, duplicated or substituted an evidence claim')
        seen.add(key)
        for raw in row['production_receipts']:
            ArtifactRef.from_dict(raw)
        claim = claims[key]
        reason = None
        if row['authority_scope'] != 'authorized':
            reason = 'provider_claim_authority_unavailable'
        elif row['independence'] != 'independent' or prior.intersection(row['assessment_dependencies']):
            reason = 'provider_assessment_not_independent'
        elif any(ref['sha256'] in prior for ref in row['production_receipts']):
            reason = 'prior_observation_is_not_independent_production_receipt'
        if reason:
            gaps.append({'subject_id': claim['subject_id'], 'claim': claim['claim'], 'prerequisite': reason})
        else:
            accepted.append(copy.deepcopy(claim))
            receipts.extend(copy.deepcopy(row['production_receipts']))
    if seen != set(claims):
        raise ValueError('provider provenance does not cover its complete claim inventory')
    return accepted, gaps, receipts


class RegisteredIndependentEvidenceProvider:
    """Host-owned read port, restricted to the existing production authority ABI."""
    def __init__(self, *, evidence_authority):
        if type(evidence_authority) is not RegisteredBoundaryQualificationEvidence:
            raise ValueError('evidence acquisition requires the registered production evidence reader')
        self.evidence_authority = evidence_authority
        self.binding = self.identity()

    def identity(self):
        return {'schema_version': 'dataevolver.registered_independent_evidence_provider.v1',
            'reader_sha256': file_sha256(Path(__file__)),
            'evidence_authority': self.evidence_authority.identity(),
            'mode': 'import_completed_host_evidence', 'dispatch_allowed': False}

    def acquire(self, witness, *, import_allowed=True):
        if type(import_allowed) is not bool or self.identity() != self.binding:
            raise ValueError('evidence acquisition permissions or provider binding changed')
        # This uses the original qualifier to validate every source/scope/basis
        # binding before provenance filtering; no new qualification rules replace it.
        consumed = (self.evidence_authority.read(witness) if import_allowed else {
            'qualified_chain': None, 'records': [], 'artifact_hashes': {},
            'missing_prerequisites': ['evidence_import_permission'],
            'new_model_calls': 0, 'new_operator_executions': 0})
        qualify_boundary_witness(witness, consumed)
        pixels, effects = set(), set()
        for entry in consumed['records']:
            for row in entry['record']['claims']:
                target = effects if row['claim'] == 'effect_ownership' else pixels
                target.update(raw['sha256'] for raw in row['evidence'])
        if pixels.intersection(effects):
            raise ValueError('effect ownership needs evidence distinct from pixel agreement')
        accepted, gaps, providers, receipts = [], [], [], []
        authority = self.evidence_authority
        for entry in consumed['records']:
            host = authority.authorities[entry['stage']]
            event = next(e for e in host.base.controller.store.events.read()
                         if e.event_type == 'ACTION_COMMITTED' and e.payload['call_id'] == host.base.call_id)
            closed = event.payload['effect_verification']['artifact_hashes']
            abi = event.payload['receipt']['stage_result']['production_evidence']
            def resolve(raw):
                ref = ArtifactRef.from_dict(raw); path = host.store.resolve(ref)
                if closed.get(str(path)) != ref.sha256:
                    raise ValueError('provider provenance or receipt is outside the committed host closure')
                return path
            manifest = json.loads(resolve(abi['artifacts']['native_audit']).read_text())
            if set(manifest) != {'role', 'cases'} or manifest['role'] != 'native_audit':
                raise ValueError('provider independence needs the original host native-audit manifest')
            inventory = {s['family_id'] for s in authority.execution.execution_protocol().sources
                         if s['family_id'] in authority.execution.execution_protocol().families(entry['stage'])}
            by_family = {}
            for case in manifest['cases']:
                if (set(case) != {'family_id', 'artifact', 'receipt_sha256'}
                        or case['family_id'] not in inventory or case['family_id'] in by_family):
                    raise ValueError('provider native-audit family inventory changed')
                by_family[case['family_id']] = case
                resolve(case['artifact'])
            if set(by_family) != inventory:
                raise ValueError('provider native-audit omitted an original production family')
            case = by_family[entry['family_id']]
            if case['receipt_sha256'] != entry['host_receipt_sha256']:
                raise ValueError('provider audit belongs to a different actual host execution')
            provenance = json.loads(resolve(case['artifact']).read_text())
            rows, missing, native = _select_provider_claims(witness, entry, provenance,
                execution_sha256=authority.execution.compute_hash(),
                trust_profile_sha256=authority.execution.runtime['trust_profile_sha256'])
            for ref in native:
                resolve(ref)
            gaps.extend(missing); receipts.extend(native)
            providers.append({'provider_id': provenance['provider_id'], 'stage': entry['stage'],
                'authority_sha256': payload_sha256(host.base.identity()),
                'provenance': case['artifact'], 'host_receipt_sha256': case['receipt_sha256'],
                'accepted_claims': len(rows)})
            if rows:
                accepted.append({**entry, 'record': {**entry['record'], 'claims': rows}})
        filtered = {**consumed, 'records': accepted}
        qualification = qualify_boundary_witness(witness, filtered)
        for gap in gaps:
            row = next(r for r in qualification['findings'] if
                       (r['subject_id'], r['claim']) == (gap['subject_id'], gap['claim']))
            if row['finding'] == 'unresolved':
                row['missing_prerequisites'] = sorted(set(row['missing_prerequisites']+[gap['prerequisite']]))
        if not providers:
            for row in qualification['findings']:
                row['missing_prerequisites'] = sorted(set(row['missing_prerequisites']+['authorized_independent_evidence_provider']))
        return {'schema_version': 'dataevolver.independent_evidence_acquisition.v1',
            'source': witness['source'], 'witness_sha256': payload_sha256(witness),
            'provider_binding_sha256': payload_sha256(self.binding),
            'provider_availability': 'AVAILABLE' if providers else 'UNAVAILABLE',
            'providers': providers, 'production_receipts': receipts, 'provider_gaps': gaps,
            'accepted_evidence_records': len(accepted), 'qualification': qualification,
            'artifact_hashes': copy.deepcopy(consumed['artifact_hashes']),
            'new_model_calls': 0, 'new_probe_executions': 0, 'new_operator_executions': 0,
            'image_submission_allowed': False, 'candidate_adoption_allowed': False,
            'production_qualification_passed': False, 'new_accepted_records': 0}
