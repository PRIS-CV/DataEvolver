"""Structured catalog I/O; grouping and selection reuse the image operators."""
from .contracts import payload_sha256


def load(ctx):
    value = ctx.read('source')
    if not isinstance(value, dict) or not isinstance(value.get('rows'), list) or not value['rows']:
        raise ValueError('catalog requires nonempty rows')
    ctx.budget.reserve_rows(ctx.node.node_id, len(value['rows']))
    return {'catalog': ctx.write_json('catalog.json', value)}


def export(ctx):
    value = ctx.read('catalog')
    rows = value['rows']
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('catalog export contains duplicate row identities')
    report = {'rows': len(rows), 'families': len({r['family_id'] for r in rows}),
              'catalog_sha256': payload_sha256(value),
              'semantic_quality_verified': False, 'training_benefit_verified': False}
    return {'catalog': ctx.write_json('catalog.json', value), 'report': ctx.write_json('report.json', report)}


def register(registry):
    from dataevolver.harness.operator_sdk import declare_step
    catalog = {'type': 'object', 'required': ['rows', 'sources'],
               'properties': {'rows': {'type': 'array', 'minItems': 1}, 'sources': {'type': 'array'}}}
    closed = {'type': 'object', 'properties': {}, 'additionalProperties': False}
    registry.register_step(declare_step('catalog.load', load, inputs={'source': catalog},
                            outputs={'catalog': catalog}, parameters=closed))
    registry.register_step(declare_step('catalog.export', export, inputs={'catalog': catalog},
                            outputs={'catalog': catalog, 'report': {'type': 'object'}}, parameters=closed))
