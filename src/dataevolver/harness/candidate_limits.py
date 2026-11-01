"""Atomic R/M candidate-slot accounting, separate from native request units."""
import json
from pathlib import Path

from dataevolver.dataset.contracts import identifier, payload_sha256, canonical_json
from .state_store import CampaignLease, atomic_write_json


class CandidateSlotLimits:
    def __init__(self, *, root, round_id, quotas, closed_history):
        self.root=Path(root).resolve();identifier(round_id)
        if (set(quotas)!={'R','M'} or any(type(n) is not int or n<0 for n in quotas.values())
                or not closed_history):
            raise ValueError('candidate slots require a frozen round, limits and closed-history identity')
        canonical_json(closed_history)
        self.contract={'round_id':round_id,'quotas':dict(quotas),'closed_history':json.loads(canonical_json(closed_history))}

    def identity(self):
        return json.loads(canonical_json({'root':str(self.root),**self.contract}))

    def _read(self):
        path=self.root/'candidates.json'
        if any(p.is_symlink() for p in (path,self.root,*self.root.parents)):
            raise ValueError('candidate accounting cannot contain symlinks')
        if not path.exists():return {'contract':self.contract,'reservations':[]}
        value=json.loads(path.read_text())
        if value['contract']!=self.contract:raise ValueError('candidate round, quota or closed history changed')
        keys=set()
        for row in value['reservations']:
            if (set(row)!={'key','layer','candidate_id','binding'} or row['key'] in keys
                    or row['layer'] not in self.contract['quotas']
                    or row['key']!=payload_sha256([self.contract['round_id'],row['layer'],row['candidate_id']])):
                raise ValueError('candidate reservation identity changed')
            keys.add(row['key'])
        if any(sum(r['layer']==k for r in value['reservations'])>n for k,n in self.contract['quotas'].items()):
            raise ValueError('candidate slot quota exceeded')
        return value

    def reserve(self, *, layer, candidate_id, binding):
        identifier(candidate_id);canonical_json(binding)
        if layer not in self.contract['quotas']:raise ValueError('unknown evolution layer')
        with CampaignLease(self.root/'.lease',ttl_seconds=60):
            value=self._read();key=payload_sha256([self.contract['round_id'],layer,candidate_id])
            row={'key':key,'layer':layer,'candidate_id':candidate_id,'binding':binding}
            previous=[r for r in value['reservations'] if r['key']==key]
            if previous:
                if previous!=[row]:raise ValueError('reserved candidate binding changed')
                return row
            if sum(r['layer']==layer for r in value['reservations'])>=self.contract['quotas'][layer]:
                raise ValueError('candidate slot quota exhausted')
            value['reservations'].append(row);atomic_write_json(self.root/'candidates.json',value)
            return row

    def usage(self):
        rows=self._read()['reservations']
        return {layer:sum(row['layer']==layer for row in rows) for layer in self.contract['quotas']}

    def lookup(self, *, layer, candidate_id):
        rows=[r for r in self._read()['reservations'] if r['layer']==layer and r['candidate_id']==candidate_id]
        if len(rows)!=1:raise ValueError('exact candidate reservation is absent')
        return rows[0]
