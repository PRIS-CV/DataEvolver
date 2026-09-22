"""Offline evidence-integrity checks; no model calls or network access."""
import hashlib
import json
from pathlib import Path
import unittest

from build_showcase import diff


ROOT = Path(__file__).parent


class TraceEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / 'traces.json').read_text())
        cls.cases = {case['id']: case for case in cls.manifest['cases']}

    def test_bundle_and_embedded_copy_agree(self):
        payload = (ROOT / 'traces-data.js').read_text()
        embedded = json.loads(payload.removeprefix('window.TRACE_SHOWCASE = ').strip().removesuffix(';'))
        self.assertEqual(embedded, self.manifest)
        self.assertEqual(set(self.cases), {'grounding', 'count'})
        for case_id, case in self.cases.items():
            self.assertEqual(case, json.loads((ROOT / 'evidence' / f'{case_id}.json').read_text()))

    def test_complete_original_frames(self):
        for case_id, expected_count in [('grounding', 2), ('count', 4)]:
            case = self.cases[case_id]
            self.assertEqual([r['round'] for r in case['rounds']], list(range(expected_count)))
            hashes = {x['file']: x['sha256'] for x in case['provenance'] if x.get('transform') == 'byte-identical copy'}
            for r in case['rounds']:
                path = ROOT / r['image']
                self.assertTrue(path.is_file())
                self.assertTrue(path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), hashes[path.name])

    def test_grounding_evidence_and_boundaries(self):
        case = self.cases['grounding']
        before, after = case['rounds']
        self.assertFalse(before['gate_passed'])
        self.assertTrue(after['gate_passed'])
        self.assertLess(before['hybrid_score'], case['threshold'])
        self.assertGreaterEqual(after['hybrid_score'], case['threshold'])
        self.assertEqual(case['parameter_diff'], diff(before['control_state'], after['control_state']))
        self.assertEqual(before['control_state']['camera'], after['control_state']['camera'])
        self.assertNotEqual(before['camera']['location'], after['camera']['location'])
        self.assertTrue(case['realized_camera_diff'])
        self.assertEqual(case['shadow_execution']['lights_touched'], [])
        contact = case['execution_evidence']['b'][0]
        self.assertAlmostEqual(contact['bottom_z_after'] - contact['ground_z'], 0.005, places=5)
        self.assertGreater(contact['bottom_z_before'], contact['bottom_z_after'])

    def test_count_trace_keeps_regression_and_partial_failure(self):
        rounds = self.cases['count']['rounds']
        self.assertEqual([len(r['passed_constraints']) for r in rounds], [8, 8, 7, 8])
        self.assertNotIn('c_count_primary', rounds[0]['failed_constraints'])
        self.assertIn('c_count_primary', rounds[2]['failed_constraints'])
        self.assertNotIn('c_count_primary', rounds[3]['failed_constraints'])
        self.assertIn('c_controlled_occlusion', rounds[3]['failed_constraints'])
        self.assertEqual(self.cases['count']['success_criterion'], 'failure_then_constraint_improvement')
        for r in rounds:
            self.assertTrue(r['prompt'].strip())
            self.assertEqual(set(r['passed_constraints']) & set(r['failed_constraints']), set())

    def test_no_private_paths_or_raw_model_transcripts(self):
        text = json.dumps(self.manifest)
        for forbidden in ['/aaaidata/', '/home/', '/Users/', 'raw_response', '<think>', 'api_key']:
            self.assertNotIn(forbidden, text)


if __name__ == '__main__':
    unittest.main()
