"""Persistent identities over the existing pixel/dHash/declared-source relation."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .contracts import identifier, payload_sha256
from .families import assign_families, duplicate_method
from dataevolver.harness.state_store import CampaignLease, atomic_write_json


class FamilyLedger:
    def __init__(self, path, *, namespace, hamming_distance):
        self.path = Path(path).absolute()
        identifier(namespace)
        assign_families([], hamming_distance)
        self.namespace, self.distance = namespace, hamming_distance

    def _read(self):
        if any(p.is_symlink() for p in (self.path, *self.path.parents)):
            raise ValueError('family ledger cannot follow symlinks')
        if not self.path.exists():
            return {'schema_version': 'dataevolver.family_ledger.v1', 'namespace': self.namespace,
                    'hamming_distance': self.distance, 'members': {}, 'aliases': {},
                    'conflicted': [], 'releases': {}, 'sequence': 0}
        value = json.loads(self.path.read_text())
        digest = value.pop('sha256')
        if (digest != payload_sha256(value) or value['namespace'] != self.namespace
                or value['hamming_distance'] != self.distance
                or value['schema_version'] != 'dataevolver.family_ledger.v1'):
            raise ValueError('family ledger binding or content changed')
        return value

    def _write(self, state):
        atomic_write_json(self.path, {**state, 'sha256': payload_sha256(state)})

    @staticmethod
    def _canonical(state, family):
        seen = set()
        while family in state['aliases']:
            if family in seen:
                raise ValueError('family alias cycle')
            seen.add(family)
            family = state['aliases'][family]
        return family

    def assign(self, rows):
        """Old IDs/splits stay fixed; aliases describe later cluster mergers."""
        with CampaignLease(self.path.with_suffix('.lease'), ttl_seconds=300):
            state = self._read()
            before = copy.deepcopy(state)
            incoming = {}
            keys = ('id', 'pixel_sha256', 'dhash', 'source_dataset', 'source_family')
            for row in rows:
                identifier(row['id'])
                if row['id'] in incoming or not row.get('split'):
                    raise ValueError('family input needs unique IDs and explicit split')
                record = {k: row[k] for k in keys}
                old = state['members'].get(row['id'])
                if old and any(old[k] != record[k] for k in keys):
                    raise ValueError('existing family member content changed')
                incoming[row['id']] = record
            all_rows = [{k: r[k] for k in keys} for r in state['members'].values()]
            all_rows += [r for k, r in incoming.items() if k not in state['members']]
            groups = {}
            for row in assign_families(all_rows, self.distance):
                groups.setdefault(row['family_id'], []).append(row['id'])
            proposed = {r['id']: r['split'] for r in rows}
            for ids in groups.values():
                old = [state['members'][i] for i in ids if i in state['members']]
                if old:
                    anchor = min(old, key=lambda x: (x['created'], x['family_id']))
                    family = self._canonical(state, anchor['family_id'])
                    splits = {r['split'] for r in old}
                    for r in old:
                        other = self._canonical(state, r['family_id'])
                        if other != family:
                            state['aliases'][other] = family
                else:
                    family = 'family-' + payload_sha256([self.namespace, sorted(ids)[0]])[:24]
                    splits = {proposed[i] for i in ids}
                split = next(iter(splits)) if len(splits) == 1 else None
                if len(splits) != 1 or any(self._canonical(state, f) == family for f in state['conflicted']):
                    state['conflicted'] = sorted(set(state['conflicted']) | {family})
                for i in ids:
                    if i not in state['members']:
                        state['members'][i] = {**incoming[i], 'family_id': family,
                            'split': split or proposed[i], 'created': len(state['members'])}
            if state != before:
                state['sequence'] += 1
                self._write(state)
            conflicted = {self._canonical(state, f) for f in state['conflicted']}
            result = []
            for row in rows:
                old = state['members'][row['id']]
                canonical = self._canonical(state, old['family_id'])
                result.append({**copy.deepcopy(row), 'family_id': old['family_id'],
                    'canonical_family_id': canonical, 'split': old['split'],
                    'family_status': 'split_conflict' if canonical in conflicted else 'clean'})
            return {'rows': result, 'ledger_sha256': payload_sha256(state),
                    'duplicate_method': duplicate_method(self.distance)}

    def register_release(self, release_id, member_ids):
        identifier(release_id)
        with CampaignLease(self.path.with_suffix('.lease'), ttl_seconds=300):
            state = self._read()
            members = sorted(member_ids)
            if not members or len(set(members)) != len(members) or set(members) - set(state['members']):
                raise ValueError('release must bind existing unique family members')
            previous = state['releases'].get(release_id)
            if previous is not None and previous != members:
                raise ValueError('family release membership is immutable')
            if previous is None:
                state['releases'][release_id] = members
                state['sequence'] += 1
                self._write(state)
        return self.release_status(release_id)

    def release_status(self, release_id):
        identifier(release_id)
        state = self._read()
        conflicts = {self._canonical(state, f) for f in state['conflicted']}
        affected = [i for i in state['releases'][release_id]
                    if self._canonical(state, state['members'][i]['family_id']) in conflicts]
        return {'release_id': release_id, 'clean': not affected, 'affected_members': affected,
                'ledger_sha256': payload_sha256(state)}

    def require_release(self, release_id, member_ids=None, *, unique_families=False):
        """Recheck previously delivered members after later cross-batch merges."""
        state = self._read()
        if member_ids is not None and sorted(member_ids) != state['releases'].get(release_id):
            raise ValueError('consumer membership differs from the registered release')
        result = self.release_status(release_id)
        if not result['clean']:
            raise ValueError('family release is contaminated by a cross-split merger')
        families = [self._canonical(state, state['members'][i]['family_id'])
                    for i in state['releases'][release_id]]
        if unique_families and len(set(families)) != len(families):
            raise ValueError('family release now contains duplicate canonical families')
        return result


def ledger_for_context(ctx):
    """A declared ledger always lives inside the graph's writable dataset store."""
    from pathlib import PurePosixPath
    relative = ctx.parameters['ledger_path']
    path = PurePosixPath(relative)
    if (not relative or path.is_absolute() or '..' in path.parts or str(path) != relative
            or '\\' in relative or str(path) == '.'):
        raise ValueError('family ledger needs a normalized dataset-relative path')
    target = ctx.graph.dataset_root / relative
    if not target.resolve().is_relative_to(ctx.graph.dataset_root):
        raise ValueError('family ledger escaped its writable store')
    return FamilyLedger(target, namespace=ctx.parameters['namespace'],
                        hamming_distance=ctx.parameters['hamming_distance'])


def assign_persistent(ctx):
    value = ctx.read('catalog')
    rows = [{**r, 'split': r.get('split', ctx.parameters['split'])} for r in value['rows']]
    result = ledger_for_context(ctx).assign(rows)
    return {'catalog': ctx.write_json('families.json', {**value, **result})}


def register(registry):
    """Standard SDK/CLI binding shared by image and structured catalog graphs."""
    from dataevolver.harness.operator_sdk import declare_step
    registry.register_step(declare_step('images.persistent_families', assign_persistent,
        inputs={'catalog': {'type': 'object', 'required': ['rows']}},
        outputs={'catalog': {'type': 'object', 'required': ['rows', 'ledger_sha256', 'duplicate_method']}},
        parameters={'type': 'object', 'additionalProperties': False, 'properties': {
            'ledger_path': {'type': 'string', 'minLength': 1}, 'namespace': {'type': 'string', 'minLength': 1},
            'hamming_distance': {'type': 'integer', 'minimum': 0, 'maximum': 64},
            'split': {'type': 'string', 'enum': ['train', 'validation', 'test', 'pilot', 'dev', 'eval']}},
            'required': ['ledger_path', 'namespace', 'hamming_distance', 'split']}, modalities=('json', 'image')))


def persistent_family_step(ledger):
    """Inject the same host-owned ledger into either domain's unified Registry."""
    from dataevolver.harness.operator_sdk import declare_step
    def execute(ctx):
        value = ctx.read('catalog')
        if ctx.parameters['hamming_distance'] != ledger.distance:
            raise ValueError('family relation differs from the frozen ledger')
        rows = [{**r, 'split': r.get('split', ctx.parameters['split'])} for r in value['rows']]
        result = ledger.assign(rows)
        return {'catalog': ctx.write_json('families.json', {**value, **result})}
    step = declare_step('images.families', execute, version='3',
        inputs={'catalog': {'type':'object'}}, outputs={'catalog': {'type':'object'}},
        parameters={'type':'object', 'additionalProperties':False,
            'properties':{'hamming_distance':{'type':'integer','minimum':0,'maximum':64},
                          'split':{'type':'string','enum':['train','validation','test','pilot','dev','eval']}},
            'required':['hamming_distance','split']}, kind='data', modalities=('json','image'))
    from dataclasses import replace
    spec = replace(step.spec, model_config_hash=payload_sha256({
        'ledger': str(ledger.path), 'namespace': ledger.namespace, 'hamming_distance': ledger.distance}))
    return replace(step, spec=spec, contract=replace(step.contract, spec_sha256=spec.sha256()))
