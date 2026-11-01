"""Real Gateway/transport projection with a captured wire, no model inference."""
import copy
import json

import pytest

from dataevolver.adapters.model_contracts import ModelPolicy, ModelProfile, ModelRequest
from dataevolver.adapters.model_gateway import LocalEndpointProvider, ModelError, ModelGateway
from dataevolver.harness.runtime_rsi import ACTION_SCHEMA, SYSTEM_PROMPT


def endpoint(tmp_path,monkeypatch,output):
    monkeypatch.setenv('FIXTURE_ENDPOINT','http://127.0.0.1:1')
    profile=ModelProfile('fixture','local_endpoint','fixture-qwen','fixture-1',endpoint_env='FIXTURE_ENDPOINT')
    provider=LocalEndpointProvider(profile,'http://127.0.0.1:1/v1',profile.model,None)
    wire=[]
    def send(suffix,*,payload=None,timeout=10):
        wire.append(copy.deepcopy(payload))
        return {'model':profile.model,'choices':[{'message':{'content':json.dumps(output)},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':1,'completion_tokens':1}}
    provider._send=send
    gateway=ModelGateway(profiles=(profile,),policy=ModelPolicy({'agent.runtime_rsi':'fixture'},1,
        allowed_local_origins=('http://127.0.0.1:1',)),receipt_root=tmp_path/'gateway',providers={'fixture':lambda:provider})
    return gateway,wire


def terminal():
    return {'action':'ABSTAIN','layer':'none','hypothesis':'Insufficient development validation.',
        'evidence_ids':[],'observation':{'tool':'','parameters_json':'{}'},
        'patch':{'kind':'','parameters_json':'{}'},'experience_id':'','reason':'Preserve existing budget.'}


@pytest.mark.parametrize('has_system',[False,True])
def test_endpoint_preserves_one_leading_system_and_exact_request(tmp_path,monkeypatch,has_system):
    gateway,wire=endpoint(tmp_path,monkeypatch,terminal())
    messages=({'role':'system','content':SYSTEM_PROMPT},) if has_system else ()
    messages+=({'role':'user','content':'Frozen context: 训练反馈\n{"loss": 0.08}'},)
    request=ModelRequest('fixture-request','agent.runtime_rsi',messages,ACTION_SCHEMA)
    before=copy.deepcopy(request.to_dict())
    receipt=gateway.invoke(request)
    assert receipt.output==terminal() and not receipt.reused
    assert [m['role'] for m in wire[0]['messages']]==['system','user']
    assert wire[0]['messages'][1]==messages[-1]
    assert 'Return one JSON object satisfying this schema:' in wire[0]['messages'][0]['content']
    if has_system:assert SYSTEM_PROMPT in wire[0]['messages'][0]['content']
    assert request.to_dict()==before
    assert gateway.recover(request).without_reuse_hash()==receipt.without_reuse_hash()
    assert len(wire)==1 and gateway.audit_usage()['model_calls']==1


def test_full_terminal_schema_still_rejects_invalid_output(tmp_path,monkeypatch):
    output=terminal();output['observation']['tool']='forbidden-tool'
    gateway,wire=endpoint(tmp_path,monkeypatch,output)
    request=ModelRequest('fixture-invalid','agent.runtime_rsi',
        ({'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':'Frozen context'}),ACTION_SCHEMA)
    with pytest.raises(ModelError):gateway.invoke(request)
    assert not list(gateway.root.glob('*.receipt.json'))
    assert gateway.audit_usage()['model_calls']==1
    assert json.loads((gateway.root/'fixture-invalid.intent.json').read_text())['state']=='INVALID_OUTPUT'
