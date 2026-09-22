// Export one explicitly selected historical comparison, never run a model.
import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const archive = process.argv[2];
if (!archive) throw new Error('Usage: node tooling/export-model-case.mjs <reviewed-local-archive>');
const target = fileURLToPath(new URL('../frontend/traces/', import.meta.url));
const id = 'prod_00475_layout_00334_v1';
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const provenanceBytes = await readFile(resolve(archive, 'SOURCE_PROVENANCE.json'));
const provenance = JSON.parse(provenanceBytes);
const manifestBytes = await readFile(resolve(archive, 'metadata/qwen_manifest.jsonl'));
const rows = manifestBytes.toString().trim().split('\n').map(JSON.parse);
const baselineBytes = await readFile(resolve(archive, `flux_baseline/${id}/flux_depth_g10__spatial/record.json`));
const baseline = JSON.parse(baselineBytes);
const reportBytes = await readFile(resolve(archive, 'REPORT.md'));
const retryHash = 'c1bdf176f6f53fbca3d61ea631a4149119afcb53704ad4dfad1ea3c141b778d2';
if (!reportBytes.toString().includes(retryHash)) throw new Error('Retry image hash is absent from archived report');
const frames = [
  { id: 'reference', file: 'target.jpg', role: 'blender_reference', model: 'Blender', extension: 'jpg' },
  { id: 'depth', file: 'depth.png', role: 'depth_control', model: 'raw8 depth', extension: 'png' },
  { id: 'qwen-free', file: 'no_depth__spatial.png', role: 'qwen_no_depth', model: 'Qwen-Image-2512', arm: 'no_depth__spatial', extension: 'png' },
  { id: 'qwen-depth', file: 'raw8_s1p2__spatial.png', role: 'qwen_depth', model: 'Qwen-Image-2512', arm: 'raw8_s1p2__spatial', extension: 'png' },
  { id: 'flux-before', file: 'flux_baseline_spatial_seed0.png', role: 'flux_baseline', model: 'FLUX.1-Depth-dev', arm: 'flux_depth_g10__spatial', extension: 'png' },
  { id: 'flux-after', file: 'flux_retry_simple_selection_seed0.png', role: 'flux_retry', model: 'FLUX.1-Depth-dev', arm: 'flux_depth_g10__simple_selection_seed0', extension: 'png' },
];
await mkdir(resolve(target, 'images'), { recursive: true });
const exported = [];
for (const frame of frames) {
  const local = `cases/${id}/${frame.file}`;
  const bytes = await readFile(resolve(archive, local));
  const expected = frame.id === 'flux-before' ? baseline.output_sha256 : frame.id === 'flux-after' ? retryHash : provenance.files.find(f => f.local_path === local)?.local_sha256;
  if (!expected || sha(bytes) !== expected) throw new Error(`Image integrity mismatch: ${frame.file}`);
  const record = rows.find(r => r.layout_id === id && r.arm === frame.arm);
  if (frame.id.startsWith('qwen-') && record?.output_sha256 !== expected) throw new Error('Qwen manifest mismatch');
  const image = `images/models-medium-${frame.id}.${frame.extension}`;
  await copyFile(resolve(archive, local), resolve(target, image));
  exported.push({ id: frame.id, role: frame.role, model: frame.model, arm: frame.arm || null, image,
    sha256: expected, transform: 'byte-identical copy',
    record_hash: record?.record_hash || (frame.id === 'flux-before' ? baseline.record_hash : null),
    evidence_level: frame.id === 'flux-after' ? 'archived_report_image_hash' : 'archived_provenance_or_generation_record',
  });
}
const data = {
  schema_version: 'dataevolver.public_model_comparison.v1', curated_on: '2026-09-22',
  case_id: id, date: '2026-08-18', split: 'tuning',
  subjects: ['traffic barrier', 'electric kettle', 'shopping cart'],
  kind: 'historical_configuration_comparison_not_continuous_agent_trace',
  images: exported,
  findings: {
    qwen: 'Photographic material and background, but object positions and sizes differ from the Blender layout.',
    flux: 'Baseline and retry better follow the layout; kettle shape and handle change in the retry. This is a visual observation, not an isolated causal or final-success claim.',
    change: 'FLUX prompt variant changes from spatial to simple; the source report labels both selection seed0. Raw retry request was not available in this local evidence package, so identical seed or other parameters are not independently asserted.',
    experiment_status: 'blocked', vlm_used: false,
    qwen_compatibility: 'Qwen-Image-2512 with Qwen-Image-Blockwise-ControlNet-Depth is experimental; it is not the official ControlNet training base.',
  },
  limits: [
    'Curated tuning case, not a benchmark, independent test or new training result.',
    'Qwen arms are a configuration comparison; Qwen to FLUX changes the generator and is not one continuous repair trace.',
    'FLUX before/after is a historical baseline/retry comparison, not a newly executed autonomous Harness run.',
    'No per-case numeric score is published without the corresponding original evaluator output.',
    'The three-case experiment remained blocked according to the archived report; visual improvement is not full acceptance.',
    'Original images are unedited; crops, scoring and model inference were not rerun.',
  ],
  provenance: [
    { file: 'SOURCE_PROVENANCE.json', sha256: sha(provenanceBytes) },
    { file: 'qwen_manifest.jsonl', sha256: sha(manifestBytes) },
    { file: 'flux_baseline_record.json', sha256: sha(baselineBytes) },
    { file: 'REPORT.md', sha256: sha(reportBytes), note: 'Local archived report; raw retry evaluator outputs are not bundled.' },
  ],
};
await writeFile(resolve(target, 'model-cases.json'), JSON.stringify(data, null, 2) + '\n');
await writeFile(resolve(target, 'model-cases-data.js'), `export default ${JSON.stringify(data, null, 2).replaceAll('<', '\\u003c')};\n`);
console.log(`Exported ${data.images.length} byte-identical images with public provenance.`);
