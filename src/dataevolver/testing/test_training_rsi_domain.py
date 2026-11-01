"""Synthetic contract checks; these are not visual or RSI benefit evidence."""
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from dataevolver.dataset.contracts import payload_sha256, file_sha256
from dataevolver.harness.adapters.training_rsi import TrainingRSIDomain, validate_regions, observation_media
from dataevolver.harness.state_store import atomic_write_json


def fixture_domain(tmp_path):
    campaign = tmp_path / 'campaign'
    trial = {'scope': 'baseline', 'source': {'sha256': 'a' * 64}, 'candidate': {'sha256': 'b' * 64},
             'media': [], 'qualified': False, 'model_usage': {'fixture': True}}
    path = campaign / 'initial/trial.json'; atomic_write_json(path, trial)
    task = {'slot_id': 'fixture-source', 'selected_subject_ids': ['s1', 's2']}
    config = {'production_contract': {'production': {'count': 1, 'tasks': [task]},
              'planning': {'goal': {'original_request': 'fixture original constraint'}},
              'dataset_store': 'data', 'bundle_store': 'bundle'},
              'roots': {'data': str(tmp_path / 'seed'), 'bundle': str(tmp_path / 'seed-bundle')},
              'scopes': {k: {'dataset': str(tmp_path / k / 'data'), 'bundle': str(tmp_path / k / 'bundle')}
                         for k in ('baseline', 'candidate-000', 'candidate-001')},
              'evidence_root': str(tmp_path / 'evidence'),
              'seed': {'family_id': 'fixture-source-a0', 'work_dir': str(tmp_path),
                       'dataset_root': str(tmp_path / 'seed'), 'file_hashes': {}}}
    domain = TrainingRSIDomain(config, campaign)
    state = SimpleNamespace(data={'last_stage_result': {'evidence_refs': [str(path)]}}, experiment_index=0)
    return domain, state, trial


def observe(domain, state, trial, *, stale=False):
    path = domain.campaign / 'observations/fixture.json'
    atomic_write_json(path, {'source': trial['source'],
                            'candidate': {'sha256': 'c' * 64} if stale else trial['candidate'], 'output': {'fixture': True}})
    state.data['rsi_observations'] = [{'call_id': 'probe-fixture', 'verification': {'status': 'VERIFIED'},
                                     'stage_result': {'evidence_refs': [str(path)]}}]


def proposal():
    return {'layer': 'R', 'patch': {'kind': 'effect_support', 'parameters_json': json.dumps({'regions': [
        {'subject_id': 's1', 'box_xyxy_normalized': [.1, .2, .3, .4], 'reason': 'fixture observation only'}]})},
        'evidence_ids': ['probe-fixture'], 'experience_id': '', 'hypothesis': 'fixture hypothesis'}


def test_runtime_region_proposal_requires_actual_current_candidate_probe(tmp_path):
    domain, state, trial = fixture_domain(tmp_path)
    with pytest.raises(ValueError, match='registered probe'):
        domain.compile_intervention(proposal(), state)
    observe(domain, state, trial, stale=True)
    with pytest.raises(ValueError, match='registered probe'):
        domain.compile_intervention(proposal(), state)
    observe(domain, state, trial)
    compiled = domain.compile_intervention(proposal(), state)
    assert compiled.candidate_delta['proposal_actor'] == 'runtime_agent'
    assert compiled.parameters['scope'] == 'candidate-000'
    assert compiled.candidate_delta['source'] == trial['source']
    assert compiled.candidate_delta['evidence'][0]['sha256']
    assert domain.config()['production_contract']['planning']['goal']['original_request'] == 'fixture original constraint'


@pytest.mark.parametrize('box', [[-.1, 0, .3, .4], [.1, .2, .1, .4], [0, 0, float('nan'), 1], [0, 0, 300, 400]])
def test_runtime_patch_rejects_wrong_units_or_nonfinite_regions(box):
    with pytest.raises(ValueError):
        validate_regions({'regions': [{'subject_id': 's1', 'box_xyxy_normalized': box, 'reason': 'fixture'}]}, ['s1'])


def test_patch_cannot_change_preserved_subjects_models_or_evaluation(tmp_path):
    with pytest.raises(ValueError):
        validate_regions({'regions': [{'subject_id': 'preserved', 'box_xyxy_normalized': [0, 0, 1, 1], 'reason': 'fixture'}]}, ['s1'])
    with pytest.raises(ValueError):
        validate_regions({'regions': [], 'quality_threshold': 0}, ['s1'])
    domain, state, trial = fixture_domain(tmp_path); observe(domain, state, trial)
    p = proposal(); p['layer'] = 'M'
    with pytest.raises(ValueError, match='method M'):
        domain.compile_intervention(p, state)


def test_trial_namespaces_are_disjoint_and_unregistered_scope_rejected(tmp_path):
    domain, state, _ = fixture_domain(tmp_path)
    assert domain.scope('baseline').roots['data'] != domain.scope('candidate-000').roots['data']
    assert domain.scope('candidate-000').roots['data'] != domain.scope('candidate-001').roots['data']
    with pytest.raises(ValueError, match='not frozen'):
        domain.scope('../../seed')
    context = domain.context(state)
    assert 'model_usage' not in context['current_trial']
    assert context['compatible_experience_ids'] == []


def test_probe_cannot_carry_an_edit_or_shell_operation(tmp_path):
    domain, state, _ = fixture_domain(tmp_path)
    with pytest.raises(ValueError):
        domain.compile_observation({'observation': {'tool': 'pixel_comparison', 'parameters_json': '{"shell":"edit"}'}}, state)
    with pytest.raises(ValueError):
        domain.compile_observation({'observation': {'tool': 'shell', 'parameters_json': '{}'}}, state)


def test_probe_obeys_registered_worker_image_capacity_without_changing_images():
    media = [{'label': str(i), 'artifact': {'sha256': str(i)}} for i in range(14)]
    assert observation_media({'media': media}, 6) == media[:6]
    assert observation_media({'media': media}, 4) == media[:4]
    with pytest.raises(ValueError): observation_media({'media': media}, 2)


def test_hardware_resume_reuses_exact_completed_probe_not_a_new_forward(tmp_path):
    domain,state,trial=fixture_domain(tmp_path);observe(domain,state,trial)
    item=state.data.pop('rsi_observations')[0]
    path=item['stage_result']['evidence_refs'][0]
    domain.value['prior_observations']=[{'item':item,'path':path,'sha256':file_sha256(Path(path))}]
    assert domain.context(state)['available_evidence_ids']==['current_trial','probe-fixture']
    assert domain.compile_intervention(proposal(),state).candidate_delta['evidence'][0]['path']==path
    atomic_write_json(Path(path),{'tampered':True})
    with pytest.raises(ValueError,match='inherited observation changed'):domain.context(state)


def test_rollback_exposes_actual_rejected_media_to_next_runtime_probe(tmp_path):
    domain,state,trial=fixture_domain(tmp_path)
    rejected={**trial,'scope':'candidate-000','candidate':{'sha256':'d'*64}}
    path=domain.campaign/'experiments/e000/candidate/trial.json';atomic_write_json(path,rejected)
    state.data.update(rsi_history=[{'decision':{'decision':'ROLLBACK'}}],
        experiment_candidate_stage_result={'evidence_refs':[str(path)]})
    context=domain.context(state)
    assert context['current_trial']['candidate']==rejected['candidate']
    assert context['current_trial_role']=='rejected_experiment'
    _,call=domain.compile_observation({'observation':{'tool':'pixel_comparison','parameters_json':'{}'}},state)
    assert call['trial_path']==str(path) and call['trial_sha256']==payload_sha256(rejected)
