"""Build a small public trace bundle from explicitly supplied experiment copies.

No model calls or experiment mutations. Host paths and raw model transcripts are
not published. Exported excerpts retain source basenames and SHA256 provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diff(a, b, prefix=''):
    rows = []
    for key in sorted(a.keys() | b.keys()):
        old, new = a.get(key), b.get(key)
        field = f'{prefix}/{key}'
        if isinstance(old, dict) and isinstance(new, dict):
            rows.extend(diff(old, new, field))
        elif old != new:
            rows.append({'field': field, 'before': old, 'after': new})
    return rows


def build(evidence_root, archive_root):
    target = Path(__file__).parent
    (target/'images').mkdir(exist_ok=True)
    (target/'evidence').mkdir(exist_ok=True)
    provenance = []

    def read(path):
        provenance.append({'file': path.name, 'sha256': sha(path)})
        return json.loads(path.read_text())

    def image(path, name):
        destination = target/'images'/name
        shutil.copyfile(path, destination)
        provenance.append({'file': name, 'sha256': sha(path), 'transform': 'byte-identical copy'})
        return 'images/'+name

    case_id = 'arisuniv_0130_obj1595__obj1645__outdoor7'
    root = evidence_root/'grounding'
    status = read(root/'gate_status.json')
    action = read(root/'agent_round01.json')
    states = [read(root/'states'/f'round{i:02d}.json') for i in range(2)]
    reviews = [read(root/'reviews'/f'{case_id}_r{i:02d}_agg.json') for i in range(2)]
    metadata = [read(root/f'round{i:02d}_renders'/'metadata.json') for i in range(2)]
    assert states[0]['camera'] == states[1]['camera']
    assert action['sample_id'] == case_id and status['accepted_round_idx'] == 1
    rounds = []
    for i in range(2):
        im = archive_root/'selected108_representative_images_20260519'/'vlm_round1_accept'/case_id/'rounds'/f'round{i:02d}_{case_id}.png'
        rounds.append({'round': i, 'image': image(im, f'grounding-r{i}.png'),
                       'issues': reviews[i]['issue_tags'], 'hybrid_score': reviews[i]['hybrid_score'],
                       'gate_passed': status['rounds'][i]['passed'],
                       'max_ground_gap': reviews[i]['cv_metrics']['geometry_metrics']['max_ground_gap'],
                       'control_state': states[i], 'camera': metadata[i]['camera']})
    grounding = {'id': 'grounding', 'source_run': 'universal_contract_selected108_vlm_scorecalib_t075_r5_20260519',
                 'source_case': case_id, 'kind': 'historical_scene_loop', 'date': '2026-05-19',
                 'rounds': rounds, 'parameter_diff': diff(*states),
                 'recorded_actions': action['actions'],
                 'execution_evidence': {k: v['post_placement_adjustments'] for k,v in metadata[1]['objects'].items()},
                 'shadow_execution': metadata[1]['lighting']['shadow_adjustments'],
                 'pairwise_review': reviews[1]['pairwise_vs_prev'],
                 'realized_camera_diff': diff(metadata[0]['camera'], metadata[1]['camera']),
                 'decision': status['status'], 'threshold': status['accepted_threshold'],
                 'limits': ['Selected historical example, not a benchmark or a new Harness run.',
                            'Hybrid scores mix VLM, image and geometry signals; first and later review contexts differ.',
                            'Ground snapping has execution metadata. Shadow lights_touched is empty; no isolated shadow-improvement claim.',
                            'Requested camera controls stay fixed but the realized camera position changes; this is not a fixed-camera ablation.',
                            'Images are preserved local archive copies; intermediate remote renders were cleaned. Metadata was read from the server.'],
                 'provenance': list(provenance)}
    (target/'evidence'/'grounding.json').write_text(json.dumps(grounding, ensure_ascii=False, indent=2)+'\n')
    provenance.clear()
    source_run = 't2i_constraint_loop_hard_glass_antireg_20260622_142826'
    glass_root = archive_root/source_run
    loop = read(glass_root/'loop_state.json')
    glass_rounds = []
    for r in loop['history']:
        i=r['round_idx']
        assert i == len(glass_rounds)
        review=r['review']
        # Keep only final structured constraint outcomes, not raw/thinking text.
        glass_rounds.append({'round': i,
            'image': image(glass_root/f'round_{i:02d}'/'samples'/'images'/f'hardglass{i:02d}_0001.png', f'count-r{i}.png'),
            'prompt': r['prompt'], 'rewrite_reason': r['rewrite_reason'],
            'failed_constraints': r['failed_constraints'], 'uncertain_constraints': r['uncertain_constraints'],
            'passed_constraints': r['passed_constraints'],
            'changes_from_previous': r['constraint_improvement_from_prev'],
            'constraint_status': [{k: c.get(k) for k in ['constraint_id','description','status']} for c in review['constraint_checklist']]})
    glass = {'id': 'count', 'source_run': source_run, 'kind': 'historical_t2i_prompt_repair',
             'date': '2026-06-22', 'rounds': glass_rounds, 'decision': loop['status'],
             'success_criterion': loop['success_criterion'], 'success_evidence': loop['success_evidence'],
             'limits': ['Round 2 to 3 repairs bottle count; it does not pass the full request.',
                        'Occlusion still fails. Earlier regressions remain visible.',
                        'These are regenerated images, not edits preserving identical pixels or object identity.',
                        'Counts are archived VLM judgments, not an independent human benchmark. The final ruler also has visible markings.',
                        'No headline overall-score improvement is inferred from these records.'],
             'provenance': list(provenance)}
    (target/'evidence'/'count.json').write_text(json.dumps(glass, ensure_ascii=False, indent=2)+'\n')
    manifest = {'schema_version': 'dataevolver.public_trace.v1', 'curated_on': '2026-09-22',
                'selection': 'Two purposively selected historical repairs with original images and structured records; not a random sample or current Harness quality result.',
                'screening': {'multi_round_gate_records_inspected': 246,
                              'note': 'Many remote intermediate PNGs were cleaned; score-only candidates were excluded. Local archive copies preserved the selected scene pair.'},
                'cases': [grounding, glass]}
    (target/'traces.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    # A classic data script keeps the page usable from file:// without fetch/CORS.
    (target/'traces-data.js').write_text('window.TRACE_SHOWCASE = '+json.dumps(manifest, ensure_ascii=False).replace('<','\\u003c')+';\n')
    print(json.dumps({'cases':len(manifest['cases']), 'images':6, 'output':str(target)}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--evidence-root', required=True, type=Path)
    parser.add_argument('--archive-root', required=True, type=Path)
    args=parser.parse_args()
    build(args.evidence_root, args.archive_root)
