"""Read-only acceptance and portable collection assembly for the Harness.

This module never invokes a model or edits an existing request or artifact.
"""
from __future__ import annotations

import collections
import json
import shutil
import subprocess
import sys
from pathlib import Path

from dataevolver.dataset.contracts import file_sha256, payload_sha256
from dataevolver.dataset.supervision import load_dataset
from dataevolver.dataset.training_pipeline import QUALITY_CHECKS
from .state_store import atomic_write_json


def merge_usage(*values):
    attempts={}
    for value in values:
        for item in value.get('attempts',[]):
            key=item['request_id']
            old=attempts.get(key)
            if old is not None:
                if old['request_hash']!=item['request_hash']:
                    raise ValueError('request ID reused with a different request hash')
                if old.get('wall_seconds') is not None and item.get('wall_seconds') is not None and old['wall_seconds']!=item['wall_seconds']:
                    raise ValueError('historical request duration changed')
                if item.get('wall_seconds') is None:continue
            attempts[key]=item
    rows=sorted(attempts.values(),key=lambda x:x['request_id'])
    return {'model_calls':sum(x.get('attempts',1) for x in rows),'attempts':rows,
        'observed_wall_seconds':sum(x.get('wall_seconds') or 0 for x in rows),
        'unknown_duration_attempts':sum(x.get('wall_seconds') is None for x in rows),
        'monetary_cost':None if any(x.get('monetary_cost') is None for x in rows) else sum(x['monetary_cost'] for x in rows)}


def journal_hashes(roots):
    return {str(path):file_sha256(path) for root in roots for path in Path(root).rglob('*.json')
            if any(part in {'model_receipts','pe_model_receipts'} for part in path.parts)}


def capture_execution_audit(roots,out,*,config_roots=(),max_output_bytes=None):
    """Union actual reservations, including failures, and copy exact native evidence."""
    if max_output_bytes is not None and (type(max_output_bytes) is not int or max_output_bytes < 1):
        raise ValueError('execution audit output bound must be a positive integer')
    queue=list(map(Path,roots));directories={};config_index={}
    for parent in config_roots:
        for path in sorted(Path(parent).glob('*/config.json')):
            config_index.setdefault(file_sha256(path),[]).append(path.parent)
    def declared_sources(value):
        if isinstance(value,dict):
            if isinstance(value.get('source_config_sha256'),str):queue.extend(config_index.get(value['source_config_sha256'],[]))
            for item in value.values():declared_sources(item)
        elif isinstance(value,list):
            for item in value:declared_sources(item)
    while queue:
        root=queue.pop().resolve()
        if str(root) in directories:continue
        directories[str(root)]=root
        contract=root/'config.json'
        if contract.exists():
            value=json.loads(contract.read_text());declared_sources(value)
            for entry in value.get('production',{}).get('preceding_runs',[]):
                if entry.get('config_dir'):queue.append(Path(entry['config_dir']))
        goal=root/'goal.json'
        if goal.exists():
            d=json.loads(goal.read_text()).get('planner',{}).get('runtime',{}).get('domain',{})
            declared_sources(d.get('production_contract',{}))
            for entry in d.get('production_contract',{}).get('production',{}).get('preceding_runs',[]):
                if entry.get('config_dir'):
                    parent=Path(entry['config_dir'])
                    if entry.get('config_sha256') and file_sha256(parent/'config.json')!=entry['config_sha256']:
                        raise ValueError('historical configuration changed before audit')
                    queue.append(parent)
            capability=d.get('surface_capability')
            if capability:
                request=Path(capability['request_path'])
                if file_sha256(request)!=capability['request_sha256']:
                    raise ValueError('implemented capability request changed before execution audit')
                queue.append(Path(capability['diagnostic_config']))
            if d.get('protocol_observation_recovery'):
                recovery=d['protocol_observation_recovery']
                for name,sha in recovery['file_hashes'].items():
                    if file_sha256(Path(name))!=sha:raise ValueError('protocol recovery evidence changed before audit')
                queue.append(Path(recovery['diagnostic_config']))
            if d.get('prior_runtime'):queue.append(Path(d['prior_runtime']['config_dir']))
            if d.get('seed',{}).get('work_dir'):queue.append(Path(d['seed']['work_dir']))
            for p in d.get('seed',{}).get('additional_journals',{}):queue.append(Path(p).parent.parent)
    before=journal_hashes(directories);usage=[];requests={};files={};preflight={}
    for i,root in enumerate(directories.values()):
        for p in root.rglob('*.intent.json'):
            if p.parent.name not in {'model_receipts','pe_model_receipts'}:continue
            record=json.loads(p.read_text());rid=p.name[:-len('.intent.json')]
            receipt=p.with_name(rid+'.receipt.json');observed={};state=record.get('state','UNKNOWN')
            if receipt.exists():
                wrapper=json.loads(receipt.read_text())
                if wrapper['sha256']!=payload_sha256(wrapper['receipt']):raise ValueError('invalid audit receipt')
                observed=wrapper['receipt']['usage'];state='COMPLETED'
            if state in {'STARTED','AMBIGUOUS','UNKNOWN'}:
                raise ValueError('unknown model effect must be reconciled before delivery')
            row={'request_id':rid,'request_hash':record['request_hash'],'attempts':record['attempts'],
                'wall_seconds':observed.get('wall_seconds',record.get('wall_seconds')),
                'monetary_cost':observed.get('cost'),'state':state,'error_code':record.get('error_code'),
                'role':record.get('role'),'usage':observed}
            usage.append({'attempts':[row]});requests.setdefault(rid,[]).append(str(p))
            if state=='RETRY_SAFE' and record.get('error_code') in {'INVALID_INPUT','CONFIG_MISSING','RESOURCE_BUSY','RESOURCE_UNAVAILABLE'}:
                preflight[rid]={'intent':str(p),'sha256':file_sha256(p),'provider_declared_retry_safe':True,
                                'classification':'provider_preflight_rejected','error_code':record['error_code']}
        # Exact bytes only. Do not export authentication env files or rewrite refs.
        for p in root.rglob('*'):
            if not p.is_file():continue
            relative=p.relative_to(root)
            if (any(part in {'model_receipts','pe_model_receipts','native','attempts','capability-requests'} for part in relative.parts) or
                    p.name in {'config.json','goal.json','state.json','events.jsonl','frozen_contract.json',
                        'trace.json','observation.json','trial.json','matched_trial.json','decision.json','proposal.json',
                        'evaluation.json','validation.json','production.json','node_run.json','stage_result.json'}):
                rel=Path('runs')/f'run{i:03d}'/relative
                files[str(p)]=(rel,file_sha256(p))
    union=merge_usage(*usage)
    if union['unknown_duration_attempts']:raise ValueError('unknown attempt durations need reconciliation before delivery')
    if max_output_bytes is not None:
        # Reserve manifest overhead before creating/copying any audit files.
        estimate=sum(Path(p).stat().st_size for p in files)+len(json.dumps({
            'files':[(p,str(rel),sha) for p,(rel,sha) in files.items()],
            'usage':union,'requests':requests,'preflight':preflight}).encode())*3+65536
        if estimate>max_output_bytes:
            raise ValueError('execution audit exceeds its finite output budget')
    out=Path(out)
    if out.exists():
        saved=json.loads((out/'audit.json').read_text())
        if saved['source_hashes']!={p:sha for p,(_,sha) in files.items()}:raise ValueError('execution audit inputs changed')
        return saved
    out.mkdir(parents=True)
    for source,(rel,sha) in files.items():
        target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if file_sha256(target)!=sha:raise ValueError('audit byte copy failed')
    native_paths=[Path(p) for p in files if 'native' in Path(p).parts]
    accepted_preflight={}
    for rid in requests:
        if not any(p.name.startswith(rid+'.') or rid in p.parts for p in native_paths):
            if rid not in preflight:raise ValueError('missing exact native evidence for model attempt '+rid)
            accepted_preflight[rid]=preflight[rid]
    if journal_hashes(directories)!=before:raise ValueError('audit modified model journals')
    value={'schema_version':'dataevolver.execution_audit.v1','usage':union,
        'source_hashes':{p:sha for p,(_,sha) in files.items()},'requests':requests,
        'files':[{'path':str(rel),'sha256':sha} for rel,sha in files.values()],
        'preflight_rejections_without_native_forward':accepted_preflight,
        'source_journals_unchanged':True,'model_calls_submitted_by_audit':0,
        'original_request_receipt_artifact_refs_rewritten':False}
    atomic_write_json(out/'audit.json',value);return value


def require_visual_acceptance(trial):
    if trial.get('qualified') is not True or trial.get('runtime_visual_actor')!='runtime_agent':
        raise ValueError('a runtime Agent must actually accept the sample')
    for review in (trial['model_review'],trial['runtime_visual_review']):
        if review.get('status')!='pass' or any(review['criteria'][k]['status']!='pass' for k in QUALITY_CHECKS):
            raise ValueError('all original visual quality checks must pass')


def verify_trial_bundle(trial, out, *, receipt_roots=(), allow_fixture=False):
    require_visual_acceptance(trial)
    out=Path(out).resolve();bundle=Path(trial['bundle']).resolve()
    before=journal_hashes(receipt_roots)
    identity={'manifest_sha256':file_sha256(bundle/'manifest.json'),'checksums_sha256':file_sha256(bundle/'checksums.json')}
    proof_path=out/'independent-validation.json'
    if proof_path.exists():
        value=json.loads(proof_path.read_text())
        if not value['accepted'] or any(value[k]!=v for k,v in identity.items()):
            raise ValueError('existing independent proof differs from this accepted trial')
        load_dataset(value['independent_root'],allow_fixture=allow_fixture)
        return value
    if out.exists():raise ValueError('refusing a partially populated independent validation directory')
    out.mkdir(parents=True)
    copied=out/'relocated-bundle';shutil.copytree(bundle,copied)
    loaded=load_dataset(copied,allow_fixture=allow_fixture)
    if len(loaded)!=1:raise ValueError('one workflow source must export exactly one sample')
    row=loaded[0][0]
    if (set(row['input'])!={'background','subjects','instruction'} or
            row['target']['image']['sha256']!=trial['source']['sha256'] or
            row['input']['background']['sha256']!=trial['candidate']['sha256']):
        raise ValueError('exported supervision does not bind the actual accepted source and background')
    from PIL import Image
    decoded=[]
    for path in copied.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.png','.jpg','.jpeg','.webp'}:
            with Image.open(path) as image:image.load();size=list(image.size)
            decoded.append({'path':path.relative_to(copied).as_posix(),'sha256':file_sha256(path),'size':size})
    if journal_hashes(receipt_roots)!=before:raise ValueError('verification mutated or added a model receipt')
    value={'schema_version':'dataevolver.runtime_training_proof.v1','accepted':True,**identity,
        'independent_root':str(copied),'loaded_records':1,'decoded_images':decoded,
        'cases':[{'family_id':row['family_id'],'split':row['split'],'target_sha256':trial['source']['sha256'],
            'background_sha256':trial['candidate']['sha256'],'reviewer_type':'runtime_agent','human_review_performed':False}],
        'usage':trial.get('model_usage',{}),'journal_unchanged':True,'new_model_requests':0,
        'runtime_visual_review':trial['runtime_visual_review'],'trial_sha256':payload_sha256(trial)}
    atomic_write_json(proof_path,value)
    return value


STANDALONE_LOADER='''"""Python 3 + Pillow. Run from any directory; repository and model access unnecessary."""
from pathlib import Path
import hashlib,json
from PIL import Image
root=Path(__file__).resolve().parent
index=json.loads((root/'collection.json').read_text())
assert index['schema_version']=='dataevolver.supervision_collection.v1'
families=set();sources=set();counts={}
for part in index['parts']:
    bundle=(root/part['bundle']).resolve();assert bundle.is_relative_to(root)
    assert hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest()==part['manifest_sha256']
    assert hashlib.sha256((bundle/'checksums.json').read_bytes()).hexdigest()==part['checksums_sha256']
    for rel,sha in json.loads((bundle/'checksums.json').read_text()).items():
        p=(bundle/rel).resolve();assert p.is_relative_to(bundle)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
    stores={k:(bundle/v).resolve() for k,v in json.loads((bundle/'stores.json').read_text()).items()}
    def resolve(ref):
        p=(stores[ref['store']]/ref['path']).resolve();assert p.is_relative_to(bundle)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256'];return p
    seen=set()
    def closure(value):
        if isinstance(value,list):
            for v in value:closure(v)
        elif isinstance(value,dict):
            if {'store','path','sha256'}<=set(value):
                key=(value['store'],value['path'],value['sha256'])
                if key in seen:return
                seen.add(key);p=resolve(value)
                if value.get('media_type')=='application/json':closure(json.loads(p.read_text()))
                elif str(value.get('media_type','')).startswith('image/'):
                    with Image.open(p) as image:image.load()
            else:
                for v in value.values():closure(v)
    for row in json.loads((bundle/'manifest.json').read_text())['records']:
        closure(row);assert set(row['input'])=={'background','subjects','instruction'}
        assert row['family_id'] not in families;families.add(row['family_id'])
        sha=row['target']['image']['sha256'];assert sha not in sources;sources.add(sha)
        counts[row['split']]=counts.get(row['split'],0)+1
        print(json.dumps({'id':row['id'],'family_id':row['family_id'],'split':row['split'],'subjects':[s['id'] for s in row['input']['subjects']]}))
assert len(families)==index['records'] and counts=={k:v for k,v in index['split_counts'].items() if v}
if index.get('execution_audit'):
    ref=index['execution_audit'];p=(root/ref['path']).resolve();assert p.is_relative_to(root)
    assert hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256']
    for entry in json.loads(p.read_text())['files']:
        q=(p.parent/entry['path']).resolve();assert q.is_relative_to(p.parent)
        assert hashlib.sha256(q.read_bytes()).hexdigest()==entry['sha256']
terminal=root/'terminal-harness-proof.json'
if terminal.exists():
    proof=json.loads(terminal.read_text());assert proof['accepted']
    assert hashlib.sha256((root/'collection.json').read_bytes()).hexdigest()==proof['collection_sha256']
    for rel,sha in proof['files'].items():
        p=(root/rel).resolve();assert p.is_relative_to(root)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
'''


def attach_terminal_proof(directory,controller):
    """Attach the terminal Controller snapshot after its final event is committed."""
    state=controller.status();valid,_=controller._frozen()
    if state.state.value!='COMPLETE' or not valid or controller.replay().to_dict()!=state.to_dict():
        raise ValueError('terminal package requires actual frozen Controller completion and replay')
    root=Path(controller.planner.domain.value['delivery_root']);directory=Path(directory)
    index=json.loads((root/'collection.json').read_text())
    if index['records']!=30 or index['split_counts']!={'train':18,'validation':6,'test':6} or len(state.data.get('accepted_assets',[]))!=30:
        raise ValueError('terminal source/split counts differ from the original goal')
    source={'goal.json':directory/'goal.json','evaluator.json':directory/'evaluator.json',
        'source.tar':directory/'source.tar','state.json':controller.layout.state_path,
        'events.jsonl':controller.layout.events_path,'frozen_contract.json':controller.layout.frozen_contract_path}
    hashes={}
    for name,path in source.items():
        relative='terminal-harness/'+name;target=root/relative
        if target.exists() and file_sha256(target)!=file_sha256(path):raise ValueError('terminal proof input changed')
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(path,target)
        hashes[relative]=file_sha256(target)
    proof={'accepted':True,'collection_sha256':file_sha256(root/'collection.json'),'files':hashes,
        'harness_state':'COMPLETE','frozen_contract_valid':True,'replay_matches_snapshot':True,
        'new_model_requests':0,'method_improvement_verified':False,'training_benefit_verified':False}
    path=root/'terminal-harness-proof.json'
    if path.exists() and json.loads(path.read_text())!=proof:raise ValueError('terminal proof changed')
    atomic_write_json(path,proof)
    subprocess.run([sys.executable,'-I',str(root/'load_samples.py')],check=True,capture_output=True,text=True)
    return {**proof,'proof_sha256':file_sha256(path)}


def assemble_collection(proofs,out,expected_counts,*,receipt_roots=(),allow_fixture=False):
    out=Path(out).resolve()
    if out.exists():
        existing=json.loads((out/'delivery-validation.json').read_text())
        if existing['input_proof_hashes']!=[payload_sha256(p) for p in proofs]:raise ValueError('collection inputs changed')
        return existing
    before=journal_hashes(receipt_roots);rows=[];bundles=[]
    for proof in proofs:
        if not proof['accepted'] or not proof['journal_unchanged'] or proof['new_model_requests']!=0:
            raise ValueError('collection needs independently accepted source proofs')
        bundle=Path(proof['independent_root'])
        if any(file_sha256(bundle/name)!=proof[key] for name,key in [('manifest.json','manifest_sha256'),('checksums.json','checksums_sha256')]):
            raise ValueError('accepted portable bundle changed')
        loaded=load_dataset(bundle,allow_fixture=allow_fixture);rows.extend(r for r,_ in loaded);bundles.append(bundle)
    if dict(collections.Counter(r['split'] for r in rows))!={k:v for k,v in expected_counts.items() if v}:
        raise ValueError('collection does not meet the frozen split counts')
    for key in ('family_id',):
        if len({r[key] for r in rows})!=len(rows):raise ValueError('duplicate source family')
    if len({r['target']['image']['sha256'] for r in rows})!=len(rows):raise ValueError('derived sources cross delivery splits')
    out.mkdir(parents=True);parts=[]
    for i,bundle in enumerate(bundles):
        rel=f'parts/part{i:03d}/bundle';shutil.copytree(bundle,out/rel)
        parts.append({'bundle':rel,'manifest_sha256':file_sha256(out/rel/'manifest.json'),
            'checksums_sha256':file_sha256(out/rel/'checksums.json'),'records':len(load_dataset(out/rel,allow_fixture=allow_fixture))})
    atomic_write_json(out/'collection.json',{'schema_version':'dataevolver.supervision_collection.v1','parts':parts,
        'records':len(rows),'split_counts':expected_counts,'preview':False,'training_benefit_verified':False})
    (out/'load_samples.py').write_text(STANDALONE_LOADER)
    result=subprocess.run([sys.executable,'-I',str(out/'load_samples.py')],capture_output=True,text=True,check=True)
    (out/'independent-load.jsonl').write_text(result.stdout)
    if journal_hashes(receipt_roots)!=before:raise ValueError('delivery changed the model ledger')
    proof={'accepted':True,'records':len(rows),'split_counts':expected_counts,'collection_sha256':file_sha256(out/'collection.json'),
        'input_proof_hashes':[payload_sha256(p) for p in proofs],'journal_unchanged':True,'new_model_requests':0,
        'loaded_independently':True,'usage':merge_usage(*(p.get('usage',p.get('production_validation',{}).get('usage',{})) for p in proofs))}
    atomic_write_json(out/'delivery-validation.json',proof)
    return proof
