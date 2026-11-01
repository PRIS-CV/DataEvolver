"""Runtime evidence bounds removal coverage and CPU surface harmonization.

Coverage observations never grant pixels. A fresh, separately typed attribution
may authorize effects; reconstruction uncertainty and the numerical solver do not.
The solver changes only the already compiled conditioning/composition support.
"""
from copy import deepcopy
import hashlib

from .contracts import canonical_json, payload_sha256, ArtifactRef
from .surface_preflight import TYPED_SURFACE_SCHEMA, validate_typed_observation, UnsupportedReconstruction
from .surface_effects import SurfaceSupport, _polygon

OPERATOR_ID = 'data.removal_coverage_surface_composition'
OPERATOR_VERSION = '1'
OBSERVATION_VERSION = 'dataevolver.removal_coverage_observation.v1'
OBSERVATION_SCHEMA = deepcopy(TYPED_SURFACE_SCHEMA)
OBSERVATION_SCHEMA['properties']['schema_version']['enum'] = [OBSERVATION_VERSION]
OBSERVATION_SCHEMA['properties']['subjects']['items']['properties']['boundary_assessment'] = {
    'type': 'string', 'enum': ['observed', 'none_observed', 'uncertain']}
OBSERVATION_SCHEMA['properties']['subjects']['items']['required'].append('boundary_assessment')
OBSERVATION_SCHEMA['properties']['removal_boundaries'] = {
    'type': 'array', 'maxItems': 32, 'items': {
        'type': 'object', 'additionalProperties': False,
        'required': ['subject_id', 'polygon', 'reason'],
        'properties': {'subject_id': {'type': 'string', 'minLength': 1},
                       'polygon': deepcopy(TYPED_SURFACE_SCHEMA['properties']['effects']['items']['properties']['polygon']),
                       'reason': {'type': 'string', 'minLength': 1, 'maxLength': 2000}}}}
OBSERVATION_SCHEMA['required'].append('removal_boundaries')


def validate_coverage_observation(value, selected, *, structural_only=False):
    from jsonschema import validate
    validate(value, OBSERVATION_SCHEMA)
    canonical_json(value)
    base = deepcopy(value)
    base['schema_version'] = 'dataevolver.surface_observation.v2'
    base.pop('removal_boundaries')
    for row in base['subjects']:
        row.pop('boundary_assessment')
    validate_typed_observation(base, selected, structural_only=structural_only)
    subjects = {x['subject_id']: x for x in value['subjects']}
    for row in value['removal_boundaries']:
        _polygon(row['polygon'])
        if row['subject_id'] not in subjects or (not structural_only
                and subjects[row['subject_id']]['boundary_assessment'] != 'observed'):
            raise ValueError('removal boundary needs a selected subject with observed coverage evidence')
    for sid, row in subjects.items():
        if row['boundary_assessment'] == 'observed' and not any(x['subject_id'] == sid for x in value['removal_boundaries']):
            raise ValueError('observed boundary assessment has no boundary evidence')
    return value


def coverage_surface_support(observation, selected, kept, *, size, load_mask,
                             existing_support, bindings=None, edge_margin=0,
                             padding_fraction=None, model_size=None):
    import numpy as np
    from PIL import Image, ImageDraw
    from .context_reconstruction import context_crop_box
    ids = [x['subject_id'] for x in selected]
    validate_coverage_observation(observation, ids)
    if existing_support is None or existing_support.size != size:
        raise ValueError('coverage requires the actual existing support')
    if type(edge_margin) is not int or not 0 <= edge_margin <= 8:
        raise ValueError('frozen edge margin changed')
    values = np.asarray(existing_support.convert('L'))
    if not np.isin(values, [0, 255]).all():
        raise ValueError('existing authority must be binary')
    old = values > 0
    def polygon(points):
        im = Image.new('L', size)
        ImageDraw.Draw(im).polygon([(round(x*(size[0]-1)), round(y*(size[1]-1))) for x, y in points], fill=255)
        return np.asarray(im) > 0
    def masks(rows):
        each = {}
        for row in rows:
            image = load_mask(row)
            if image.size != size:
                raise ValueError('original instance dimensions changed')
            each[row['subject_id']] = np.asarray(image.convert('L')) > 127
        total = np.zeros_like(old)
        for mask in each.values(): total |= mask
        return total, each
    silhouettes, by_id = masks(selected)
    visible, _ = masks(kept)
    for row in observation['preservation_regions']: visible |= polygon(row['polygon'])
    uncertain = np.zeros_like(old)
    reconstruction = np.zeros_like(old)
    for row in observation['uncertain_regions']:
        if row['kind'] == 'effect_attribution': uncertain |= polygon(row['polygon'])
        else: reconstruction |= polygon(row['polygon'])
    surfaces = {x['surface_id']: polygon(x['polygon']) for x in observation['surfaces']}
    receiving = np.zeros_like(old)
    surface_count = np.zeros(old.shape, dtype=np.int32)
    for mask in surfaces.values():
        receiving |= mask; surface_count += mask
    conflicts = []
    def conflict(category, mask, subjects):
        if mask.any(): conflicts.append({'category': category, 'pixels': int(mask.sum()), 'subject_ids': list(subjects)})
    conflict('selected_silhouette_protected_visible', silhouettes & visible, ids)
    conflict('selected_silhouette_outside_prior_authority', silhouettes & ~old, ids)
    effects = np.zeros_like(old)
    per_subject = {sid: np.zeros_like(old) for sid in ids}
    for row in observation['effects']:
        mask = polygon(row['polygon'])
        conflict('effect_crosses_receiving_surface', mask & ~surfaces[row['surface_id']], [row['subject_id']])
        conflict('effect_uncertain_attribution', mask & uncertain & ~silhouettes, [row['subject_id']])
        conflict('effect_protected_visible', mask & visible, [row['subject_id']])
        conflict('effect_expands_into_reconstruction_uncertainty', mask & reconstruction & ~old, [row['subject_id']])
        effects |= mask
        per_subject[row['subject_id']] |= mask
    # Only explicit fresh attributable effects add authority. Boundary polygons
    # are coverage witnesses and must never implicitly expand the support.
    prior_effect = old & receiving & ~silhouettes & ~visible & ~uncertain
    support = silhouettes | prior_effect | effects
    protected = visible | (uncertain & ~silhouettes) | (reconstruction & ~old)
    conflict('executable_protected_overlap', support & protected, ids)
    conflict('unresolved_receiving_surface_coverage', support & (surface_count != 1), ids)
    coverage = []
    for row in observation['subjects']:
        sid = row['subject_id']; boundary = np.zeros_like(old)
        for witness in observation['removal_boundaries']:
            if witness['subject_id'] == sid: boundary |= polygon(witness['polygon'])
        unknown = np.zeros_like(old)
        for region in observation['uncertain_regions']:
            if region['kind'] == 'effect_attribution' and sid in region['subject_ids']:
                unknown |= polygon(region['polygon']) & ~silhouettes
        missing = boundary & ~support
        complete = not missing.any() and not unknown.any() and row['boundary_assessment'] != 'uncertain' and row['effect_assessment'] != 'uncertain'
        coverage.append({'subject_id': sid, 'silhouette_pixels': int(by_id[sid].sum()),
                         'uncovered_boundary_pixels': int(missing.sum()),
                         'uncertain_effect_pixels': int(unknown.sum()),
                         'observed_effect_pixels_outside_execution': int((per_subject[sid] & ~support).sum()),
                         'coverage_supported': complete})
        conflict('uncovered_selected_boundary', missing, [sid])
        if not complete and not missing.any():
            conflicts.append({'category': 'unresolved_removal_coverage', 'pixels': int(unknown.sum()), 'subject_ids': [sid]})
    if not silhouettes.any(): conflicts.append({'category': 'empty_selected_authority', 'pixels': 0, 'subject_ids': ids})
    receipt = {'schema_version': 'dataevolver.removal_coverage.v1', 'operator_id': OPERATOR_ID,
        'operator_version': OPERATOR_VERSION, 'bindings': bindings or {},
        'observation_sha256': payload_sha256(observation), 'source_size': list(size),
        'existing_support_pixel_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
        'coverage': coverage, 'conflicts': conflicts, 'editable_pixels': int(support.sum()),
        'protected_pixels': int(protected.sum()), 'fresh_attributed_effect_expansion_pixels': int((effects & ~old).sum()),
        'coverage_added_authority_pixels': 0, 'context_added_editable_pixels': 0,
        'uncertainty_added_editable_pixels': 0, 'conditioning_and_composition_identical': True,
        'unresolved_subject_ids': [x['subject_id'] for x in coverage if not x['coverage_supported']],
        'coverage_supported': all(x['coverage_supported'] for x in coverage),
        'complete_goal_established': False, 'quality_verified': False,
        'image_model_calls': 0, 'supported': not conflicts}
    image = Image.fromarray(support.astype('uint8')*255)
    if padding_fraction is not None and not conflicts:
        try:
            box = context_crop_box(image, padding_fraction)
            dimensions = list(model_size or size)
            if len(dimensions) != 2 or dimensions[0] != dimensions[1] or any(type(x) is not int or x <= 0 for x in dimensions):
                raise ValueError('positive square model dimensions required')
            side = box[2]-box[0]
            receipt['crop'] = {'source_crop_xyxy': list(box), 'model_size': dimensions,
                'padding_fraction': padding_fraction, 'source_origin_xy': list(box[:2]),
                'source_to_model_scale': [dimensions[0]/side]*2, 'canvas_expanded': False}
        except ValueError as exc:
            conflicts.append({'category': 'unsupported_context_crop', 'reason': str(exc), 'pixels': 0, 'subject_ids': ids})
            receipt['supported'] = False
    receipt['execution_status'] = 'supported' if receipt['supported'] else 'unsupported'
    receipt['receipt_sha256'] = payload_sha256(receipt)
    if conflicts: raise UnsupportedReconstruction(receipt)
    return SurfaceSupport(image, Image.fromarray(protected.astype('uint8')*255), receipt)


def harmonize_arrays(source, generated, support, protected, surface_labels=None):
    """Dirichlet harmonic color correction; retain generated interior gradients."""
    import numpy as np
    from scipy import sparse, ndimage
    from scipy.sparse.linalg import cg
    s, g = np.asarray(source), np.asarray(generated)
    m, p = np.asarray(support), np.asarray(protected)
    if s.dtype != np.uint8 or g.dtype != np.uint8 or s.shape != g.shape or s.ndim != 3 or s.shape[2] != 3:
        raise ValueError('harmonization requires same-frame RGB bytes')
    if m.shape != s.shape[:2] or p.shape != m.shape or not np.isin(m, [0, 255]).all() or not np.isin(p, [0, 255]).all():
        raise ValueError('harmonization requires same-frame binary support and protection')
    m, p = m > 0, p > 0
    if not m.any() or (m & p).any(): raise ValueError('empty or protected harmonization support')
    labels = np.ones(m.shape, dtype=np.int32) if surface_labels is None else np.asarray(surface_labels)
    if labels.shape != m.shape or not np.issubdtype(labels.dtype, np.integer) or np.any(labels[m] <= 0):
        raise ValueError('harmonization requires unambiguous receiving surfaces')
    ys, xs = np.nonzero(m); n = len(ys)
    index = np.full(m.shape, -1, dtype=np.int64); index[m] = np.arange(n)
    degree = np.zeros(n); rhs = np.zeros((n, 3)); rows = []; cols = []; data = []
    anchors = np.zeros_like(m)
    delta = s.astype(float)-g.astype(float)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        yy, xx = ys+dy, xs+dx
        valid = (yy >= 0) & (yy < m.shape[0]) & (xx >= 0) & (xx < m.shape[1])
        here = np.flatnonzero(valid); ny, nx = yy[valid], xx[valid]
        same_surface = labels[ys[here], xs[here]] == labels[ny, nx]
        here, ny, nx = here[same_surface], ny[same_surface], nx[same_surface]
        degree[here] += 1
        interior = m[ny, nx]
        rows.extend(here[interior]); cols.extend(index[ny[interior], nx[interior]]); data.extend([-1.]*int(interior.sum()))
        edge = here[~interior]
        rhs[edge] += delta[ny[~interior], nx[~interior]]
        anchors[ys[edge], xs[edge]] = True
    components = 0
    for surface in np.unique(labels[m]):
        parts, count = ndimage.label(m & (labels == surface)); components += count
        anchored = set(np.unique(parts[anchors]))
        if set(range(1, count+1)) - anchored:
            raise ValueError('surface component has no preserved Dirichlet boundary')
    if np.any(degree == 0): raise ValueError('surface component has no preserved Dirichlet boundary')
    rows.extend(np.arange(n)); cols.extend(np.arange(n)); data.extend(degree)
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    preconditioner = sparse.diags(1./degree)
    correction = np.zeros((n, 3)); residuals = []
    for channel in range(3):
        result, info = cg(matrix, rhs[:, channel], M=preconditioner, rtol=1e-7, atol=1e-6, maxiter=4096)
        residual = float(np.max(np.abs(matrix @ result-rhs[:, channel])))
        if info != 0 or not np.isfinite(result).all() or residual > .05:
            raise ValueError('surface harmonization did not converge')
        correction[:, channel] = result; residuals.append(residual)
    output = s.copy()
    output[m] = np.clip(np.rint(g[m].astype(float)+correction), 0, 255).astype('uint8')
    if not np.array_equal(output[~m], s[~m]) or not np.array_equal(output[p], s[p]):
        raise ValueError('harmonization escaped authority')
    return output, {'solver': 'dirichlet_harmonic_correction', 'solver_version': '1',
        'maximum_residual_by_channel': residuals, 'components': components,
        'changed_pixels': int(np.any(output != s, axis=-1).sum()),
        'unauthorized_changed_pixels': 0, 'protected_changed_pixels': 0,
        'support_pixels': n, 'quality_verified': False}


def harmonize_surface_edit(store, store_name, family, source, generated, contract_ref, projection_ref):
    import json
    import numpy as np
    from PIL import Image
    from .supervision import write_image, write_artifact
    contract = json.loads(store.resolve(contract_ref).read_text())
    if contract['source_image'] != source.to_dict() or contract['proposal_actor'] != 'runtime_agent':
        raise ValueError('surface harmonization source binding changed')
    refs = {key: ArtifactRef.from_dict(contract[key]) for key in ('support', 'conditioning_region', 'composition_alpha', 'protected')}
    if refs['support'] != refs['conditioning_region'] or refs['support'] != refs['composition_alpha']:
        raise ValueError('conditioning and composition authority differ')
    diagnostics = contract['diagnostics']
    if (not diagnostics.get('coverage_supported') or not diagnostics.get('supported')
            or diagnostics['bindings']['source'] != source.to_dict()
            or diagnostics['bindings']['candidate'] != contract['rejected_candidate']
            or diagnostics['observation_sha256'] != payload_sha256(contract['observation'])):
        raise ValueError('harmonization requires registered complete executable coverage')
    store.resolve(ArtifactRef.from_dict(contract['rejected_candidate']))
    read = lambda ref, mode: np.asarray(Image.open(store.resolve(ref)).convert(mode))
    projection = json.loads(store.resolve(projection_ref).read_text())
    transform = json.loads(store.resolve(ArtifactRef.from_dict(projection['condition_transform'])).read_text())
    if (projection['source'] != source.to_dict() or projection['projected_model_output'] != generated.to_dict()
            or projection['authorized_support'] != refs['support'].to_dict()
            or any(projection[key] != transform[key] for key in transform)):
        raise ValueError('harmonization projection binding changed')
    box = transform['source_crop_xyxy']; dimensions = transform['source_size']
    if (len(box) != 4 or any(type(x) is not int for x in box)
            or not 0 <= box[0] < box[2] <= dimensions[0]
            or not 0 <= box[1] < box[3] <= dimensions[1]
            or box[2]-box[0] != box[3]-box[1]):
        raise ValueError('harmonization inverse transform changed')
    original = Image.open(store.resolve(source)).convert('RGB')
    model = Image.open(store.resolve(ArtifactRef.from_dict(projection['raw_model_output']))).convert('RGB')
    if list(original.size) != dimensions or list(model.size) != transform['model_size']:
        raise ValueError('harmonization transform dimensions changed')
    expected = original.copy()
    expected.paste(model.resize((box[2]-box[0], box[3]-box[1]), Image.Resampling.LANCZOS), box[:2])
    if not np.array_equal(np.asarray(expected), read(generated, 'RGB')):
        raise ValueError('generated edit does not match the recorded inverse projection')
    condition = Image.open(store.resolve(ArtifactRef.from_dict(transform['source_condition']))).convert('RGB')
    expected_condition = condition.crop(box).resize(tuple(transform['model_size']), Image.Resampling.LANCZOS)
    if not np.array_equal(np.asarray(expected_condition), read(ArtifactRef.from_dict(transform['model_condition']), 'RGB')):
        raise ValueError('actual model condition differs from its recorded forward transform')
    from PIL import ImageDraw
    labels = np.zeros((original.height, original.width), dtype=np.int32)
    overlaps = np.zeros(labels.shape, dtype=bool)
    for index, surface in enumerate(contract['observation']['surfaces'], 1):
        mask = Image.new('L', original.size)
        ImageDraw.Draw(mask).polygon([(round(x*(original.width-1)), round(y*(original.height-1)))
            for x, y in surface['polygon']], fill=255)
        region = np.asarray(mask) > 0; overlaps |= region & (labels > 0); labels[region] = index
    support = read(refs['support'], 'L')
    if np.any((support > 0) & ((labels == 0) | overlaps)):
        raise ValueError('surface harmonization authority has ambiguous receiving surfaces')
    output, proof = harmonize_arrays(read(source, 'RGB'), read(generated, 'RGB'), support, read(refs['protected'], 'L'), labels)
    ref = write_image(store, store_name, family+'/surface-composition/harmonized-edit.png', Image.fromarray(output))
    proof.update(schema_version='dataevolver.surface_harmonization.v1', operator_id=OPERATOR_ID,
        operator_version=OPERATOR_VERSION, source=source.to_dict(), generated=generated.to_dict(),
        surface_contract=contract_ref.to_dict(), support=refs['support'].to_dict(), protected=refs['protected'].to_dict(),
        projection=projection_ref.to_dict(), inverse_projection_verified=True,
        forward_condition_transform_verified=True, surface_boundaries_preserved=True,
        harmonized_edit=ref.to_dict(), proposal_actor='runtime_agent', source_dimensions_preserved=True)
    evidence = write_artifact(store, store_name, family+'/surface-composition/audit.json', canonical_json(proof).encode(), 'application/json')
    return ref, evidence
