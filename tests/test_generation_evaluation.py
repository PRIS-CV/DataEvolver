import pytest
from dataevolver.dataset.contracts import file_sha256
from dataevolver.dataset.generation_evaluation import (
    EvaluationJournal, inference_costs, soft_tifa, spatial_answers, spatial_score)


def test_five_rollouts_four_correct_and_order():
    answers = ['A']*10
    rollouts = [spatial_answers('\n'.join('A: reason' for _ in range(10))) for _ in range(5)]
    rollouts[0][2] = 'B'
    assert spatial_score(rollouts, answers)['Position'] == 1
    rollouts[1][2] = 'B'
    assert spatial_score(rollouts, answers)['Position'] == 0
    with pytest.raises(ValueError): spatial_score(rollouts[:4], answers)
    with pytest.raises(ValueError): spatial_answers('A: one answer only')


def test_am_gm_share_probabilities_and_zero_is_not_clamped():
    score = soft_tifa([[.25, 1.], [0., 1.]])
    assert score['soft_tifa_gm'] == pytest.approx(25.)
    assert score['soft_tifa_am'] == pytest.approx(56.25)
    with pytest.raises(ValueError): soft_tifa([['yes']])
    with pytest.raises(ValueError): soft_tifa([[float('nan')]])


def test_manifest_binding_resume_and_missing_denominator(tmp_path):
    a, b = tmp_path/'a.png', tmp_path/'b.png'
    a.write_bytes(b'first'); b.write_bytes(b'second')
    manifest = {'items': [{'id': 'a', 'image_sha256': file_sha256(a)},
                          {'id': 'b', 'image_sha256': file_sha256(b)}]}
    journal = EvaluationJournal(tmp_path/'results', manifest)
    args = dict(image_path=a, image_sha256=file_sha256(a), result={'score': 1}, evidence={'status': 'VERIFIED'})
    journal.record('a', **args); journal.record('a', **args)
    with pytest.raises(ValueError, match='incomplete'): journal.complete()
    with pytest.raises(ValueError, match='association'):
        journal.record('b', **args)
    journal.record('b', image_path=b, image_sha256=file_sha256(b), result={'score': 0}, evidence={'status': 'VERIFIED'})
    assert len(journal.complete()) == 2


def test_budget_has_model_seed_and_atom_multipliers():
    costs = inference_costs(model_count=5, spatial_prompts=120, atom_lengths=[7]*80,
        generation_seeds=1, training_jobs=4, checkpoints=3, development_prompts=16)
    assert costs['final_images'] == 1000
    assert costs['checkpoint_images'] == 192
    assert costs['spatial_valid_calls'] == 3000
    assert costs['geneval2_atom_calls'] == 2800
