"""Subquotas projected from the existing ModelGateway attempt journal."""
from dataclasses import dataclass, field
from pathlib import Path
import json



@dataclass(frozen=True)
class ModelAttemptLimits:
    quotas: dict[str, int]
    role_buckets: dict[str, str]
    max_wall_seconds: int = 7200
    ledger_roots: tuple[str, ...] = ()
    unit_rules: dict[str, str] = field(default_factory=dict)
    reservation_lock: str = ''

    def __post_init__(self):
        if (not self.quotas or any(type(n) is not int or n < 0 for n in self.quotas.values())
                or set(self.role_buckets.values()) - set(self.quotas)
                or type(self.max_wall_seconds) is not int or self.max_wall_seconds <= 0
                or set(self.unit_rules) - set(self.role_buckets)
                or any(v not in {'image_subject', 'video_frame_subject'} for v in self.unit_rules.values())):
            raise ValueError('invalid model-attempt subquotas')
        if self.reservation_lock and not Path(self.reservation_lock).is_absolute():
            raise ValueError('shared model reservation lock must be an explicit absolute host path')
        if len({Path(p).resolve() for p in self.ledger_roots}) != len(self.ledger_roots):
            raise ValueError('linked model ledger roots must be distinct')

    def to_dict(self):
        value = {'quotas': dict(self.quotas), 'role_buckets': dict(self.role_buckets),
                'max_wall_seconds': self.max_wall_seconds, 'ledger_roots': list(self.ledger_roots)}
        if self.unit_rules:
            value['unit_rules'] = dict(self.unit_rules)
        if self.reservation_lock:
            value['reservation_lock'] = self.reservation_lock
        return value

    def reservation_lease(self):
        from contextlib import nullcontext
        from .state_store import CampaignLease
        return CampaignLease(Path(self.reservation_lock), ttl_seconds=300) if self.reservation_lock else nullcontext()

    def units(self, request, *, artifact_store=None):
        """Host-derived reservation; a candidate cannot declare its own price."""
        if request.role not in self.unit_rules:
            return 1
        parameters = json.loads(request.messages[0]['content'])
        subjects = parameters.get('subjects')
        if len(request.artifacts) != 1 or not isinstance(subjects, list) or not subjects:
            raise ValueError('SAM native units require one media artifact and explicit subjects')
        identities = [s['subject_id'] for s in subjects]
        if len(set(identities)) != len(identities):
            raise ValueError('SAM native subjects must be distinct')
        if self.unit_rules[request.role] == 'image_subject':
            if not request.artifacts[0].media_type.startswith('image/'):
                raise ValueError('image-subject accounting cannot price video frames')
            return len(subjects)
        if artifact_store is None or request.artifacts[0].media_type != 'video/mp4':
            raise ValueError('video frame-subject accounting requires its verified artifact store')
        import av
        count = 0
        with av.open(str(artifact_store.resolve(request.artifacts[0]))) as video:
            for _ in video.decode(video=0):
                count += 1
                if count > 256:
                    raise ValueError('video frame-subject accounting exceeds the registered frame ceiling')
        if not count:
            raise ValueError('video frame-subject accounting requires decodable frames')
        return count * len(subjects)

    def usage(self, records):
        usage = {k: 0 for k in self.quotas}
        for row in records:
            role = row.get('role')
            if role not in self.role_buckets:
                raise ValueError('model ledger contains an unclassified role')
            units = row.get('native_units_per_attempt', 1)
            if type(units) is not int or units < 1:
                raise ValueError('invalid native-unit reservation')
            if role in self.unit_rules and 'native_units_per_attempt' not in row:
                raise ValueError('unit-priced role has an unpriced historical attempt')
            usage[self.role_buckets[role]] += row['attempts'] * units
        return usage

    def check(self, request, records, *, now, current_root=None, artifact_store=None):
        records = list(records)
        for root in self.ledger_roots:
            path = Path(root).resolve()
            if current_root is not None and path == Path(current_root).resolve():
                continue
            for p in path.glob('*.intent.json'):
                row = json.loads(p.read_text())
                if type(row.get('attempts')) is not int or row['attempts'] < 1:
                    raise ValueError('invalid linked Gateway attempt record')
                records.append(row)
        if request.role not in self.role_buckets:
            raise ValueError('model request has no frozen subquota')
        usage = self.usage(records)
        bucket = self.role_buckets[request.role]
        if any(usage[k] > n for k, n in self.quotas.items()) or usage[bucket] + self.units(request, artifact_store=artifact_store) > self.quotas[bucket]:
            raise ValueError('model-attempt subquota exhausted: ' + bucket)
        started = min((r.get('first_started_at', r['started_at']) for r in records), default=now)
        if now - started >= self.max_wall_seconds:
            raise ValueError('online model window exhausted')
        if request.timeout_seconds > self.max_wall_seconds - (now - started):
            raise ValueError('request timeout exceeds the remaining online window')
