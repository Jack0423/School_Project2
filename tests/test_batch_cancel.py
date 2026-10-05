"""
批次分析的取消（batch_pipeline.analyze_batch 的 should_stop）。

前台的「取消分析」只能等手上這張推論完再停（不強制中斷 GPU／RAW 解碼），
所以要守住：停下來之後不再推論任何一張、已完成的都寫進結果檔、回傳 cancelled。
模型與解碼都換成假的，和 test_batch_comparison.py 的做法相同。
"""
import importlib.util
import json
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_pipeline(evaluated):
    def decode(path):
        time.sleep(0.01)
        return Path(path).stem

    def evaluate(path, image, aesthetic_weight, return_features):
        evaluated.append(path)
        return dict(aesthetic_score=50.0, technical_score=60.0, overall_score=54.0,
                    feature_vector=[1.0] * 1280,
                    aesthetic_weight=aesthetic_weight, technical_weight=1 - aesthetic_weight)

    fake = types.SimpleNamespace(IMAGE_EXTENSIONS={'.jpg'}, UNSUPPORTED_PHOTO_EXTENSIONS=set(),
                                 LAST_ERROR=None, load_image=decode, evaluate_photo=evaluate)
    with patch.dict('sys.modules', {'ai_inference': fake}):
        spec = importlib.util.spec_from_file_location(
            'isolated_batch_cancel', Path(__file__).resolve().parents[1] / 'batch_pipeline.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class TestBatchCancel(unittest.TestCase):

    def test_stops_after_current_photo_and_keeps_finished_results(self):
        evaluated, progress = [], []
        pipeline = load_pipeline(evaluated)
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f'{n}.jpg' for n in range(10)]
            output = Path(tmp) / 'out.jsonl'
            summary = pipeline.analyze_batch(
                paths, output,
                on_progress=lambda done, total, record: progress.append(done),
                should_stop=lambda: len(progress) >= 3)
            lines = output.read_text(encoding='utf-8').splitlines()

        self.assertTrue(summary['cancelled'])
        self.assertEqual(summary['success'], 3)
        self.assertEqual(len(evaluated), 3, '取消後不可再推論任何一張')
        self.assertEqual(len(lines), 3, '已完成的每一張都要寫進結果檔')
        self.assertTrue(all(json.loads(line)['ok'] for line in lines))

    def test_not_cancelled_when_never_asked(self):
        evaluated = []
        pipeline = load_pipeline(evaluated)
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f'{n}.jpg' for n in range(5)]
            summary = pipeline.analyze_batch(paths, Path(tmp) / 'out.jsonl', should_stop=lambda: False)
        self.assertFalse(summary['cancelled'])
        self.assertEqual(summary['success'], 5)


if __name__ == '__main__':
    unittest.main()
