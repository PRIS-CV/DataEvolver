"""Receipt/media fixtures test the baseline bridge, never production quality."""
import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from dataevolver.dataset.contracts import ArtifactRef, ArtifactStore, file_sha256, payload_sha256
from dataevolver.dataset.training_pipeline import QUALITY_CHECKS
from dataevolver.harness.adapters.training_production_case import TrainingProductionCaseOperator
from dataevolver.harness.models import EvidenceStatus, OperatorCall, RecoveryStatus
from dataevolver.harness.training_trial_authority import (
    FrozenTrainingTrialAuthority, _verify_copy, _verify_json, _read_only_event,
    scoped_reference_closure, declared_predecessor_scopes)
from dataevolver.harness.training_workflow_delivery import capture_execution_audit
from dataevolver.testing.test_operator_development import fixture, write_ref


class RequestFixture:
    def __init__(self, root):
        self.config_root = root

    def identity(self):
        return {'fixture_only': True}

    def verify(self, plan, store):
        store.resolve(ArtifactRef.from_dict(plan.capability_request))


class TrialFixture:
    def __init__(self, value):
        self.value = value

    def identity(self):
        return {'fixture_only': True}

    def read(self):
        return copy.deepcopy(self.value)


def setup_case(tmp_path):
    f = fixture(tmp_path)
    _, _, plan, package, old_store, protocol, _ = f
    roots = {**old_store.roots, 'media': tmp_path/'original-media', 'evidence': tmp_path/'case-output'}
    roots['media'].mkdir()
    def media(name, image):
        p = roots['media']/(name+'.png'); image.save(p)
        return ArtifactRef('media', p.name, file_sha256(p), 'image/png').to_dict()
    original = Image.new('RGB', (16, 16), 'white')
    proposal = Image.new('RGB', (16, 16), 'blue')
    mask = Image.new('L', (16, 16), 255)
    refs = [media('original', original), media('candidate', Image.composite(proposal, original, mask)),
            media('raw', proposal), media('region', mask), media('blend', mask), media('condition', original)]
    reviews = {'status': 'fail', 'criteria': {k: {'status': 'fail' if i == 0 else 'pass'}
                                           for i, k in enumerate(QUALITY_CHECKS)}}
    trial = {'source': refs[0], 'candidate': refs[1], 'media': [{'artifact': r} for r in refs],
             'model_review': reviews, 'runtime_visual_review': reviews,
             'runtime_visual_actor': 'runtime_agent', 'qualified': False}
    request = write_ref(roots['protocol'], 'protocol', 'request.json',
                        {'source': refs[0], 'rejected_candidate': refs[1]})
    regression = write_ref(roots['protocol'], 'protocol', 'baseline.json',
                           {'kind': 'existing-failed-source-regression', 'request_sha256': request['sha256']})
    matched = write_ref(roots['protocol'], 'protocol', 'matched.json', {'fixture_only': True})
    sources = tuple({**s, 'artifact': regression} if s['family_id'] == 'regression' else s
                    for s in protocol.sources)
    protocol = replace(protocol, sources=sources, matched_context_sha256=matched['sha256'])
    protocol_ref = write_ref(roots['protocol'], 'protocol', 'baseline-protocol.json', protocol.to_dict())
    plan = replace(plan, capability_request=request, qualification_protocol=protocol_ref)
    package = replace(package, development_plan_sha256=plan.compute_hash())
    journal = tmp_path/'history'; journal.mkdir()
    write_ref(journal, 'history', 'config.json', {'fixture_only': True})
    # A native audit fixture records a completed historical failure and its cost.
    write_ref(journal, 'history', 'model_receipts/old.intent.json',
              {'request_hash': 'a'*64, 'attempts': 1, 'state': 'FAILED', 'wall_seconds': 7., 'role': 'fixture'})
    write_ref(journal, 'history', 'native/old.error.json', {'error': 'fixture failed effect'})
    authority = TrialFixture({'config': str(journal), 'trial': trial, 'trial_sha256': payload_sha256(trial),
        'historical_usage': {'model_calls': 1, 'observed_wall_seconds': 7.},
        'contract': {'models': {'fixture_only': True}},
        'scoped_reference_closure':scoped_reference_closure(trial, {'current':ArtifactStore(roots)}),
        'artifact_hashes': {}, 'store_roots': {'media': str(roots['media'])},
        'state': 'ABSTAIN', 'frozen': True, 'replay': True, 'receipt_sha256': 'b'*64})
    op = TrainingProductionCaseOperator(plan=plan, package=package, store=ArtifactStore(roots),
        request_authority=RequestFixture(old_store.roots['origin']), trial_authority=authority,
        family_id='regression', stage='engineering', output_root=roots['evidence'],
        evidence_store='evidence', matched_context=matched, python=sys.executable)
    call = OperatorCall('baseline-fixture', 'case-1', 'case-1', op.spec.operator_id, '1', 0,
        str(roots['evidence']/'calls/first'), parameters=op.parameters(), input_artifact_refs=(op.source,))
    return op, call, authority


def test_real_media_loading_preserves_failure_and_never_qualifies(tmp_path):
    op, call, authority = setup_case(tmp_path)
    receipt = op.execute(call)
    assert op.verify_effect(call, receipt).status is EvidenceStatus.VERIFIED
    case = receipt.stage_result['production_case']
    assert not any(case['checks'].values()) and not case['candidate_invocation']['actual_execution']
    assert receipt.stage_result['metrics']['production_qualification_passed'] == 0
    assert case['usage']['model_calls'] == case['usage']['planner_calls'] == 0
    load = json.loads(op.store.resolve(ArtifactRef.from_dict(case['artifacts']['independent_load'])).read_text())
    assert load['decoded_media']['isolated_python'] and len(load['decoded_media']['decoded_media']) == 6
    assert load['loaded_dataset_records'] == 0 and not load['eligible_for_delivery']
    quality = json.loads(op.store.resolve(ArtifactRef.from_dict(case['artifacts']['quality_checks'])).read_text())
    assert quality['model_review'] == authority.value['trial']['model_review']
    assert not quality['baseline_qualified'] and quality['composition_exact']
    costs = json.loads(op.store.resolve(ArtifactRef.from_dict(case['artifacts']['cost_ledger'])).read_text())
    assert costs['historical_union_usage']['observed_wall_seconds'] == 7.
    assert not costs['historical_attempts_recharged']
    assert not (tmp_path/'EXECUTED').exists()


def test_completed_recovery_never_decodes_or_assembles_again(tmp_path, monkeypatch):
    op, call, _ = setup_case(tmp_path)
    first = op.execute(call)
    def forbidden(*a, **kw):
        raise AssertionError('completed media must not be reprocessed')
    monkeypatch.setattr(op, '_assemble', forbidden)
    monkeypatch.setattr(op, '_load_media', forbidden)
    second = op.execute(call)
    assert second.reused and second.effect_hash == first.effect_hash
    assert second.stage_result == first.stage_result


@pytest.mark.parametrize('kind', ['request_source', 'request_candidate', 'media_source', 'input', 'owner'])
def test_wrong_source_receipt_and_call_identity_refuse(tmp_path, kind):
    op, call, authority = setup_case(tmp_path)
    if kind.startswith('request_'):
        field = 'source' if kind == 'request_source' else 'candidate'
        authority.value['trial'][field]['sha256'] = 'f'*64
    elif kind == 'media_source':
        authority.value['trial']['media'][0]['artifact']['sha256'] = 'f'*64
    elif kind == 'input':
        call = replace(call, input_artifact_refs=())
    else:
        op.execute(call)
        call = replace(call, call_id='different', work_dir=str(op.root/'calls/second'))
    with pytest.raises((ValueError, OSError)):
        op.execute(call)


@pytest.mark.parametrize('kind', ['media', 'native', 'quality', 'manifest', 'extra', 'usage'])
def test_copied_media_native_failure_and_accounting_tampering_refuse_recovery(tmp_path, kind):
    op, call, _ = setup_case(tmp_path); op.execute(call)
    work = Path(call.work_dir)
    if kind == 'media': target = next((work/'relocated-media').glob('*.png'))
    elif kind == 'native': target = next((work/'native-audit').rglob('old.error.json'))
    elif kind == 'quality': target = work/'quality_checks.json'
    elif kind == 'manifest': target = work/'relocated-media/manifest.json'
    elif kind == 'extra': target = work/'unrecorded.txt'
    else:
        target = work/'production-case.json'; v = json.loads(target.read_text()); v['usage']['model_calls'] = 1
        target.write_text(json.dumps(v))
        assert op.recover(call).status is RecoveryStatus.AMBIGUOUS
        return
    target.write_bytes(b'changed')
    assert op.recover(call).status is RecoveryStatus.AMBIGUOUS


def test_output_escape_and_symlink_refuse_before_capture(tmp_path):
    op, call, _ = setup_case(tmp_path)
    with pytest.raises(ValueError): op.execute(replace(call, work_dir=str(tmp_path/'escape')))
    op.root.mkdir(); (op.root/'linked').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError): op.execute(replace(call, work_dir=str(op.root/'linked/out')))


def test_finite_audit_bound_rejects_before_copy(tmp_path):
    origin = tmp_path/'origin'; origin.mkdir(); (origin/'config.json').write_text('{}')
    out = tmp_path/'copy'
    with pytest.raises(ValueError, match='finite output budget'):
        capture_execution_audit((origin,), out, max_output_bytes=1)
    assert not out.exists()


def test_historical_compiler_can_only_verify_existing_files(tmp_path):
    a = tmp_path/'a'; b = tmp_path/'b'; a.write_text('{}'); b.write_text('{}')
    _verify_copy(a, b); _verify_json(a, {})
    b.unlink()
    with pytest.raises(OSError): _verify_copy(a, b)
    assert not b.exists()
    with pytest.raises(ValueError): _verify_json(a, {'changed': True})
    _read_only_event('open', (str(a), 'r', 0))
    _read_only_event('open', ('/dev/null', None, 2))
    for event, args in [('open', (str(a), 'w', 1)), ('os.mkdir', (str(b),)),
                        ('os.remove', (str(a),)), ('subprocess.Popen', ()), ('socket.__new__', ())]:
        with pytest.raises(RuntimeError): _read_only_event(event, args)
    assert a.read_text() == '{}'


def test_trial_authority_uses_original_training_freeze_without_invented_commit_file(tmp_path):
    cfg=tmp_path/'historical';cfg.mkdir();campaign=cfg/'campaigns/original';campaign.mkdir(parents=True)
    for name, value in [('goal.json', {'campaign_id':'original'}),
                        ('frozen.json', {'source_commit':'a'*40})]:
        (cfg/name).write_text(json.dumps(value))
    (cfg/'env.sh').write_text('export CUDA_VISIBLE_DEVICES=\n')
    for name in ('frozen_contract.json','state.json','events.jsonl'):
        (campaign/name).write_text('{}')
    reader=FrozenTrainingTrialAuthority(config=cfg,call_id='original-call',python=sys.executable)
    assert reader.identity()['source_commit']=='a'*40
    assert not (cfg/'source-commit').exists()
    (campaign/'events.jsonl').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='authority changed'): reader.identity()


def test_same_historical_store_path_with_different_hashes_remains_two_exact_parts(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    left=write_ref(a,'dataset','same.json',{'from':'earlier'})
    right=write_ref(b,'dataset','same.json',{'from':'current','prior':left})
    raw=json.dumps(right)
    scopes={'earlier':ArtifactStore({'dataset':a}),'current':ArtifactStore({'dataset':b})}
    closure=scoped_reference_closure(right,scopes)
    assert len(closure)==2 and {r['sha256'] for r in closure}=={left['sha256'],right['sha256']}
    assert json.dumps(right)==raw and (a/'same.json').read_text()!= (b/'same.json').read_text()
    with pytest.raises(ValueError,match='no exact declared scope'):
        scoped_reference_closure(right,{'current':scopes['current']})


def test_legacy_observation_stores_require_the_declared_config_and_its_frozen_roots(tmp_path):
    cfg=tmp_path/'old';cfg.mkdir();root=tmp_path/'old-evidence';root.mkdir()
    ref=write_ref(root,'agent_evidence','observation.json',{'historical':True})
    config=cfg/'config.json';config.write_text('{}')
    digest=file_sha256(config)
    frozen=cfg/'frozen.json';frozen.write_text(json.dumps({'config_sha256':digest,'roots':{'agent_evidence':str(root)}}))
    contract={'production':{'preceding_runs':[{'config_dir':str(cfg),'config_sha256':digest,'evidence':ref}]}}
    scopes,hashes=declared_predecessor_scopes(contract)
    assert len(scoped_reference_closure(contract,scopes))==1 and str(frozen) in hashes
    config.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='config identity changed'): declared_predecessor_scopes(contract)
