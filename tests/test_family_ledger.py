from dataevolver.dataset.family_ledger import FamilyLedger
import pytest


def row(name, pixel, source, split='train'):
    return dict(id=name, pixel_sha256=pixel*64, dhash=('0' if pixel=='a' else 'f')*16,
                source_dataset='source', source_family=source, split=split)


def test_incremental_smaller_member_and_restart_keep_identity(tmp_path):
    ledger = FamilyLedger(tmp_path/'ledger.json', namespace='example', hamming_distance=0)
    first = ledger.assign([row('z', 'b', 'same')])['rows'][0]
    other = FamilyLedger(tmp_path/'ledger.json', namespace='example', hamming_distance=0)
    second = other.assign([row('a', 'a', 'same', 'eval')])['rows'][0]
    assert second['family_id'] == first['family_id']
    assert second['split'] == 'train'
    assert other.assign([row('z', 'b', 'same')])['rows'][0]['family_id'] == first['family_id']


def test_bridge_invalidates_both_old_releases(tmp_path):
    ledger = FamilyLedger(tmp_path/'ledger.json', namespace='example', hamming_distance=0)
    ledger.assign([row('a', 'a', 'one'), row('b', 'b', 'two', 'eval')])
    assert ledger.register_release('training', ['a'])['clean']
    assert ledger.register_release('evaluation', ['b'])['clean']
    result = ledger.assign([row('bridge', 'a', 'two')])
    assert result['rows'][0]['family_status'] == 'split_conflict'
    assert not ledger.release_status('training')['clean']
    assert not ledger.release_status('evaluation')['clean']


def test_content_and_release_membership_are_immutable(tmp_path):
    ledger = FamilyLedger(tmp_path/'ledger.json', namespace='example', hamming_distance=0)
    ledger.assign([row('a', 'a', 'one'), row('b', 'b', 'two')])
    ledger.register_release('release', ['a'])
    with pytest.raises(ValueError, match='membership'):
        ledger.register_release('release', ['b'])
    with pytest.raises(ValueError, match='content'):
        ledger.assign([row('a', 'b', 'one')])


def test_later_same_split_merger_blocks_unique_family_training_consumption(tmp_path):
    ledger=FamilyLedger(tmp_path/'ledger.json',namespace='example',hamming_distance=0)
    ledger.assign([row('a','a','one'),row('b','b','two')])
    ledger.register_release('training',['a','b'])
    ledger.require_release('training',['a','b'],unique_families=True)
    ledger.assign([row('bridge','a','two')])
    with pytest.raises(ValueError,match='duplicate canonical'):
        ledger.require_release('training',['a','b'],unique_families=True)
