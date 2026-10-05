"""Use deterministic fake inference to check orchestration, never model accuracy."""
import importlib.util
import json
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from compare_batch import sequential_batch, read_records, compare_records


class BatchComparisonTests(unittest.TestCase):
    def test_decode_parallel_inference_serial_and_equal_scores(self):
        caller = threading.get_ident()
        decode_threads, inference_threads = set(), []
        def decode(path):
            decode_threads.add(threading.get_ident())
            time.sleep(0.01)
            if Path(path).stem == 'bad':
                raise ValueError('broken photo')
            return Path(path).stem
        def evaluate(path, image, aesthetic_weight, return_features):
            self.assertTrue(return_features)
            inference_threads.append(threading.get_ident())
            score = float(image)
            return dict(aesthetic_score=score, technical_score=score+1,
                        overall_score=score+0.4, feature_vector=[1.0]*1280,
                        aesthetic_weight=aesthetic_weight, technical_weight=1-aesthetic_weight)
        fake = types.SimpleNamespace(IMAGE_EXTENSIONS={'.jpg'}, LAST_ERROR=None,
                                     load_image=decode, evaluate_photo=evaluate)
        with patch.dict('sys.modules', {'ai_inference': fake}):
            spec = importlib.util.spec_from_file_location('isolated_batch',
                    Path(__file__).resolve().parents[1] / 'batch_pipeline.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp)/f'{n}.jpg' for n in range(8)]
            seq, par = Path(tmp)/'seq.jsonl', Path(tmp)/'par.jsonl'
            sequential_batch(paths, seq, .6, fake)
            decode_threads.clear()
            progress = []
            module.analyze_batch(paths, par, on_progress=lambda *args: progress.append(args))
            self.assertGreater(len(decode_threads), 1)
            self.assertNotIn(caller, decode_threads)
            self.assertEqual(set(inference_threads), {caller})
            self.assertEqual([p[0] for p in progress], list(range(1, 9)))
            self.assertTrue(compare_records(read_records(seq), read_records(par))['all_scores_match'])
            summary = module.analyze_batch([Path(tmp)/'bad.jpg', paths[0]], Path(tmp)/'bad.jsonl')
            self.assertEqual((summary['success'], summary['failed']), (1, 1))
            with self.assertRaises(FileExistsError):
                module.analyze_batch(paths, par)

    def test_full_benchmark_report(self):
        import compare_batch
        import contextlib
        import io
        fake = types.SimpleNamespace(
            IMAGE_EXTENSIONS={'.jpg'}, UNSUPPORTED_PHOTO_EXTENSIONS={'.heic'},
            LAST_ERROR=None, DEVICE=types.SimpleNamespace(type='cpu'),
            torch=types.SimpleNamespace(__version__='test'), model_version=lambda:'test-only',
            load_image=lambda path: object(),
            evaluate_photo=lambda path, **kwargs: dict(
                aesthetic_score=50, technical_score=60, overall_score=54,
                aesthetic_weight=.6, technical_weight=.4, feature_vector=[1.0]*1280))
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules', {'ai_inference':fake}):
            spec = importlib.util.spec_from_file_location('isolated_batch_report',
                    Path(__file__).resolve().parents[1] / 'batch_pipeline.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            photos = Path(tmp)/'photos'
            photos.mkdir()
            (photos/'one.jpg').touch()
            output = Path(tmp)/'report'
            with patch.dict('sys.modules', {'batch_pipeline':module}), patch('sys.argv',
                    ['compare_batch.py',str(photos),'--output',str(output),'--repeats','2']), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(compare_batch.main(), 0)
            report = json.loads((output/'comparison.json').read_text(encoding='utf-8'))
            self.assertTrue(report['all_scores_match'])
            self.assertEqual(len(report['seconds']['parallel']), 2)
            self.assertGreater(report['speedup'], 0)
            self.assertTrue((output/'summary.md').exists())

    def test_mismatches_failures_and_empty_not_pass(self):
        a = dict(ok=True, aesthetic_score=50, technical_score=60, overall_score=54)
        self.assertFalse(compare_records({}, {})['all_scores_match'])
        self.assertFalse(compare_records({'a':a}, {'a':dict(a, overall_score=55)})['all_scores_match'])
        self.assertFalse(compare_records({'a':{'ok':False}}, {'a':{'ok':False}})['all_scores_match'])
        with self.assertRaises(ValueError):
            compare_records({'a':a}, {})
        with self.assertRaises(ValueError):
            compare_records({'a':a}, {'a':dict(a, overall_score=float('nan'))})


if __name__ == '__main__':
    unittest.main()
