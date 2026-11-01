"""Read a committed training trial using its original frozen Controller.

The reader cannot run a Controller, operator or model. Failed quality remains
failed; this authority only establishes where the real training evidence came
from. Neither a candidate implementation nor a candidate-owned module is loaded.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256


def _verify_copy(source, destination):
    """The historical compiler may verify an existing copy, never create it."""
    if file_sha256(Path(source)) != file_sha256(Path(destination)):
        raise ValueError('original compiler artifact copy changed')


def _verify_json(path, value):
    if json.loads(Path(path).read_text()) != value:
        raise ValueError('original compiler artifact changed')


def scoped_reference_closure(values, scopes):
    """Close historical refs in their declared stores, preserving raw refs.

    A historical recipe may refer to an earlier candidate with the same logical
    store/path and a different hash. Flattening all those namespaces onto the
    final candidate's store is invalid. Record every exact declared match and
    recursively close JSON bytes without rewriting historical request records.
    """
    from dataevolver.dataset.contracts import ArtifactRef
    pending=[values];seen=set();rows=[]
    while pending:
        value=pending.pop()
        if isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, dict):
            if value.get('schema_version')!=ArtifactRef.schema:
                pending.extend(value.values());continue
            ref=ArtifactRef.from_dict(value);key=(ref.store,ref.path,ref.sha256)
            if key in seen:continue
            seen.add(key);matches={}
            for scope,store in scopes.items():
                try:path=store.resolve(ref)
                except (OSError,ValueError):continue
                matches.setdefault(str(path),[]).append(scope)
            if not matches:
                raise ValueError('historical reference has no exact declared scope: '+json.dumps(value))
            rows.append({'original_reference_json':json.dumps(ref.to_dict(),sort_keys=True),
                         'sha256':ref.sha256,'media_type':ref.media_type,
                         'matches':[{'path':p,'scopes':sorted(names)} for p,names in sorted(matches.items())]})
            if ref.media_type=='application/json':
                pending.append(json.loads(Path(next(iter(matches))).read_text()))
    return sorted(rows,key=lambda row:row['original_reference_json'])


def declared_predecessor_scopes(contract):
    """Include original store namespaces for explicitly declared old evidence."""
    from dataevolver.dataset.contracts import ArtifactStore
    scopes={};hashes={}
    for i,entry in enumerate(contract.get('production',{}).get('preceding_runs',[])):
        if not entry.get('config_dir'):continue
        cfg=Path(entry['config_dir']);config=cfg/'config.json';frozen=cfg/'frozen.json'
        if not entry.get('config_sha256') or file_sha256(config)!=entry['config_sha256']:
            raise ValueError('declared historical config identity changed')
        value=json.loads(frozen.read_text())
        if value.get('config_sha256')!=entry['config_sha256'] or not isinstance(value.get('roots'),dict):
            raise ValueError('declared historical store binding changed')
        scopes['predecessor-'+str(i)]=ArtifactStore(value['roots'])
        hashes.update({str(config):file_sha256(config),str(frozen):file_sha256(frozen)})
    return scopes,hashes


def _read_only_event(event, args):
    # This is a trusted framework reader, not a sandbox for generated code.
    # The audit hook prevents accidental mutations, provider/network calls and
    # child execution by the original framework during evidence verification.
    if event == 'open':
        path, mode, flags = args
        if path == os.devnull:
            return  # Git's stderr sink does not mutate stored evidence.
        if (mode and any(c in mode for c in 'wax+')) or flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise RuntimeError('historical reader cannot write files')
    elif event == 'os.mkdir':
        if not Path(args[0]).is_dir():
            raise RuntimeError('historical reader cannot create directories')
    elif event == 'subprocess.Popen' and len(args)>1:
        executable, argv = args[:2]
        # FrozenExecutionContract verifies these two read-only Git queries.
        if (executable == 'git' and isinstance(argv, (list, tuple))
                and (list(argv) == ['git', 'rev-parse', 'HEAD']
                     or list(argv[:5]) == ['git', 'diff', '--binary', 'HEAD', '--'])):
            return
        raise RuntimeError('historical reader cannot dispatch effects')
    elif event in {'os.remove', 'os.rename', 'os.rmdir', 'os.link', 'os.symlink',
                   'os.chmod', 'os.chown', 'os.truncate', 'os.utime', 'os.system',
                   'os.exec', 'os.posix_spawn', 'subprocess.Popen', 'socket.__new__'}:
        raise RuntimeError('historical reader cannot mutate or dispatch effects')


class FrozenTrainingTrialAuthority:
    def __init__(self, *, config, call_id, python):
        self.config = Path(config).resolve()
        self.call_id = call_id
        self.python = Path(python).absolute()
        self._identity = self._current_identity()

    def _current_identity(self):
        cfg = self.config
        goal = json.loads((cfg/'goal.json').read_text())
        campaign = cfg/'campaigns'/goal['campaign_id']
        return {'config':str(cfg), 'call_id':self.call_id, 'python':str(self.python),
                'python_sha256':file_sha256(self.python), 'reader_sha256':file_sha256(Path(__file__)),
                'source_commit':json.loads((cfg/'frozen.json').read_text())['source_commit'],
                'goal_sha256':file_sha256(cfg/'goal.json'), 'freeze_sha256':file_sha256(cfg/'frozen.json'),
                'env_sha256':file_sha256(cfg/'env.sh'),
                'frozen_contract_sha256':file_sha256(campaign/'frozen_contract.json'),
                'events_sha256':file_sha256(campaign/'events.jsonl'),
                'state_sha256':file_sha256(campaign/'state.json')}

    def identity(self):
        if self._current_identity() != self._identity:
            raise ValueError('original training trial authority changed')
        return dict(self._identity)

    def read(self):
        identity = self.identity()
        before = {str(p): file_sha256(p) for p in self.config.rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts}
        env = {**os.environ, 'PYTHONPATH':str(self.config/'source/src'),
               'PYTHONDONTWRITEBYTECODE':'1', 'GIT_OPTIONAL_LOCKS':'0', 'CUDA_VISIBLE_DEVICES':''}
        # The approved historical env supplies model/store identity paths. Its
        # source path is overridden explicitly before any framework import.
        args = ['bash','-ec','set -a; source "$1"; export PYTHONPATH="$2" CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1; cd "$5/source"; exec "$3" -B "$4" --read "$5" "$6"',
                'read-frozen-training', str(self.config/'env.sh'), str(self.config/'source/src'),
                str(self.python), str(Path(__file__).resolve()), str(self.config), self.call_id]
        result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=180)
        after = {str(p): file_sha256(p) for p in self.config.rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts}
        if after != before:
            raise ValueError('historical training verification modified its original evidence')
        if result.returncode:
            raise ValueError('original training verification failed: '+result.stderr[-2000:])
        value = json.loads(result.stdout)
        if (value['call']['call_id'] != self.call_id or value['config'] != str(self.config)
                or value['source_commit'] != identity['source_commit']
                or not value['frozen'] or not value['replay']):
            raise ValueError('original training reader returned different evidence')
        self.identity()
        return value


def _read(config, call_id):
    from dataevolver.harness.cli import _existing_controller
    from dataevolver.harness.controller import EvolutionController
    from dataevolver.harness.models import OperatorCall, OperatorReceipt, EvidenceStatus
    from dataevolver.harness.adapters.training_rsi import TrainingRSITrialOperator
    from dataevolver.harness.adapters.training_supervision import TrainingSupervisionOperator
    from dataevolver.harness.adapters import training_rsi
    from dataevolver.adapters.model_gateway import ModelGateway
    from dataevolver.adapters.model_contracts import ModelRequest
    from dataevolver.dataset.contracts import payload_sha256, ArtifactRef

    def forbidden(*args, **kwargs):
        raise RuntimeError('training evidence reader cannot execute or recover effects')
    EvolutionController.run = EvolutionController.tick = forbidden
    ModelGateway.invoke = ModelGateway.adopt_completed = forbidden
    TrainingRSITrialOperator.execute = TrainingRSITrialOperator.recover = forbidden
    TrainingSupervisionOperator.execute = TrainingSupervisionOperator.recover = forbidden
    TrainingSupervisionOperator._wait = forbidden
    training_rsi._copy_exact = _verify_copy
    training_rsi._write_immutable = _verify_json
    # Recover only a receipt that already exists, without acquiring/writing a
    # Gateway lease or falling back to a provider's native recovery.
    def existing_receipt(gateway, request):
        if not (gateway.root / (request.request_id + '.receipt.json')).is_file():
            raise ValueError('original Gateway receipt is missing; recovery is forbidden')
        return gateway._recover(request, gateway._prepare(request))
    ModelGateway.recover = existing_receipt
    ModelGateway._provider = forbidden
    sys.addaudithook(_read_only_event)
    cfg = Path(config).resolve()
    frozen = json.loads((cfg/'frozen.json').read_text())
    if file_sha256(cfg/'goal.json') != frozen['goal_sha256']:
        raise ValueError('original goal changed')
    if file_sha256(cfg/'source.tar') != frozen['source_archive_sha256']:
        raise ValueError('original source archive changed')
    for name, digest in frozen['source_hashes'].items():
        if file_sha256(cfg/'source'/name) != digest:
            raise ValueError('original frozen implementation changed')
    goal = json.loads((cfg/'goal.json').read_text())
    c = _existing_controller(cfg/'campaigns', goal['campaign_id'])
    if c.status().state.value not in {'ABSTAIN','COMPLETE'} or not c._frozen()[0]:
        raise ValueError('original Controller is not a verified terminal run')
    if c.replay().to_dict() != c.status().to_dict():
        raise ValueError('original training replay differs')
    events = c.store.events.read()
    planned = [e for e in events if e.event_type=='ACTION_PLANNED' and e.payload['call_id']==call_id]
    committed = [e for e in events if e.event_type=='ACTION_COMMITTED' and e.payload['call_id']==call_id]
    if len(planned)!=1 or len(committed)!=1:
        raise ValueError('one planned and committed training trial is required')
    call = OperatorCall.from_dict(planned[0].payload['operator_call'])
    receipt = OperatorReceipt.from_dict(committed[0].payload['receipt'])
    op = c.registry.get(call.operator_id)
    if type(op) is not TrainingRSITrialOperator:
        raise ValueError('only the original registered training trial is supported')
    verification = op.verify_effect(call, receipt)
    if verification.status is not EvidenceStatus.VERIFIED:
        raise ValueError(verification.reason)
    trial_path = Path(call.work_dir)/'trial.json'
    trial = json.loads(trial_path.read_text())
    contract, store, inner = op._prepare(call)
    inner_call = call.with_updates(operator_id='data.training_supervision', parameters=contract,
                                  work_dir=str(Path(call.work_dir)/'production'))
    inner_verification = inner.verify_effect(inner_call, None)
    if inner_verification.status is not EvidenceStatus.VERIFIED:
        raise ValueError(inner_verification.reason)
    production_path = Path(inner_call.work_dir)/'production.json'
    production = json.loads(production_path.read_text())
    family_path = Path(inner_call.work_dir)/'families'/(trial['source_family']+'.json')
    family = json.loads(family_path.read_text())
    gateway = inner.gateway(inner_call)
    for record in family['model_requests']:
        cached = gateway.recover(ModelRequest.from_dict(record['request']))
        if cached is None or cached.without_reuse_hash()!=record['receipt_hash']:
            raise ValueError('training family lost its exact Gateway receipt')
    for item in trial['media']:
        store.resolve(ArtifactRef.from_dict(item['artifact']))
    predecessors,predecessor_hashes=declared_predecessor_scopes(contract)
    scoped=scoped_reference_closure([trial,contract,family['model_requests']],
        {**predecessors,**{name:op.domain.scope(name) for name in op.domain.value['scopes']}})
    paths = {trial_path, production_path, family_path,
             cfg/'goal.json', cfg/'frozen.json', c.layout.campaign_dir/'state.json',
             c.layout.campaign_dir/'events.jsonl', c.layout.campaign_dir/'frozen_contract.json'}
    paths.update(Path(name) for name in verification.artifact_hashes)
    effect = json.loads((Path(call.work_dir)/'runtime-effect.json').read_text())
    paths.add(Path(call.work_dir)/'runtime-effect.json')
    paths.update(Path(name) for name in effect['files'])
    for name in ('model_receipts', 'pe_model_receipts', 'native'):
        paths.update(p for p in (Path(inner_call.work_dir)/name).rglob('*') if p.is_file())
    paths.update(store.resolve(ArtifactRef.from_dict(item['artifact'])) for item in trial['media'])
    paths.update(Path(match['path']) for row in scoped for match in row['matches'])
    paths.update(Path(name) for name in predecessor_hashes)
    return {'config':str(cfg), 'source_commit':frozen['source_commit'],
            'frozen':True, 'replay':True, 'state':c.status().state.value,
            'call':call.to_dict(), 'receipt_sha256':receipt.sha256(),
            'trial_path':str(trial_path), 'trial':trial, 'contract':contract,
            'store_roots':{k:str(v) for k,v in store.roots.items()},
            'historical_usage':production['usage'], 'family_records':family['model_requests'],
            'scoped_reference_closure':scoped,
            'artifact_hashes':{str(path):file_sha256(path) for path in paths},
            'new_model_calls':0, 'new_image_calls':0,
            'trial_sha256':payload_sha256(trial)}


if __name__=='__main__':
    import sys
    if len(sys.argv)!=4 or sys.argv[1]!='--read':
        raise SystemExit(2)
    print(json.dumps(_read(sys.argv[2], sys.argv[3])))
