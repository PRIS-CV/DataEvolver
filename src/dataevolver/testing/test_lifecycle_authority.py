"""Historical construction must neither mutate nor pretend to resolve revocation."""
import copy
import json
from pathlib import Path

import pytest

from dataevolver.harness.operator_lifecycle import OperatorVersionStore, EMPTY_HEAD, SYNTHETIC_SCOPE
from dataevolver.harness.operator_lifecycle_authority import HistoricalCatalogPrefix, FrozenOperatorLifecycleAuthority


def catalog(tmp_path):
    store=OperatorVersionStore(tmp_path/'catalog')
    value={'operator_id':'data.fixture','version':'1','scope':SYNTHETIC_SCOPE}
    with store.lease():
        adopted=store.append(action='ADOPTED',snapshot=value,owner={'call':'adopt'},expected_head=store.head())
        revoked=store.append(action='REVOKED',snapshot=value,owner={'call':'revoke'},expected_head=store.head(),reason='fixture')
    return store,adopted,revoked


def test_historical_prefix_is_a_real_readonly_state_not_current_eligibility(tmp_path):
    store,adopted,revoked=catalog(tmp_path)
    files={p.name:p.read_bytes() for p in store.root.glob('*.json')}
    empty=HistoricalCatalogPrefix(store,EMPTY_HEAD)
    previous=HistoricalCatalogPrefix(store,adopted['sha256'])
    assert empty.head()==EMPTY_HEAD and empty.events()==[]
    assert previous.head()==adopted['sha256'] and previous.events()==[adopted]
    assert store.head()==revoked['sha256']
    with pytest.raises(ValueError,match='revoked'):
        store.active(adopted,scope=SYNTHETIC_SCOPE,head=store.head())
    with pytest.raises(ValueError,match='cannot mutate'):previous.lease()
    with pytest.raises(ValueError,match='cannot mutate'):previous.append(action='ADOPTED')
    assert {p.name:p.read_bytes() for p in store.root.glob('*.json')}==files


def test_historical_prefix_refuses_invented_or_changed_history(tmp_path):
    store,adopted,_=catalog(tmp_path)
    with pytest.raises(ValueError,match='actual prefix'):HistoricalCatalogPrefix(store,'f'*64)
    view=HistoricalCatalogPrefix(store,adopted['sha256'])
    path=store.root/'00000001.json';value=json.loads(path.read_text());value['snapshot']['version']='changed'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):view.events()


@pytest.mark.parametrize('kind',['candidate_module','missing_snapshot','candidate_environment'])
def test_reader_rejects_unbound_or_candidate_owned_bootstrap_before_subprocess(tmp_path,kind):
    cfg=tmp_path/'host';source=cfg/'source';source.mkdir(parents=True)
    (cfg/'source-commit').write_text('a'*40)
    env=cfg/'env.sh';env.write_text('exit 0\n')
    module=source
    if kind=='candidate_module':
        module=cfg/'candidate/source';module.mkdir(parents=True)
    elif kind=='missing_snapshot':
        (cfg/'source-commit').unlink()
    else:
        env=cfg/'candidate/env.sh';env.parent.mkdir();env.write_text('exit 0\n');(env.parent/'source-commit').write_text('a'*40)
    with pytest.raises(ValueError,match='explicit host'):
        FrozenOperatorLifecycleAuthority(config=cfg,module_root=module,env_path=env,python='/usr/bin/python3')
