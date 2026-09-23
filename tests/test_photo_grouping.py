"""
分組三種模式的測試：只看內容、看相機與時間、使用者自己指定一組。

用合成的批次結果（自己寫 JSONL 與 metadata），不需要權重與資料集，
因此在任何一台機器上都能跑。
"""
import json
import tempfile
import unittest
from pathlib import Path

import burst_metadata
import photo_grouping
from burst_metadata import build_burst_check, camera_identity, parse_capture_time
from photo_grouping import group_photos
from pick_best import rank_photos


def record(path, score, feature, weight=0.6, ok=True):
    return {
        "path": path,
        "ok": ok,
        "aesthetic_score": score,
        "technical_score": score,
        "overall_score": score,
        "aesthetic_weight": weight,
        "technical_weight": round(1 - weight, 2),
        "feature_vector": feature,
    }


def feature(bias):
    """產生 1280 維特徵。bias 越接近，餘弦相似度越高。"""
    return [1.0] * 1279 + [bias]


def write_jsonl(directory, records, name='batch.jsonl'):
    path = Path(directory) / name
    with open(path, 'w', encoding='utf-8') as file:
        for item in records:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')
    return str(path)


def write_metadata(directory, items, name='meta.json'):
    path = Path(directory) / name
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(items, file, ensure_ascii=False)
    return str(path)


class TestCameraIdentity(unittest.TestCase):
    """
    相機識別的三個層級。

    釘住的缺陷：原本只讀 SerialNumber，讀不到就直接放棄配對。
    實測 Sony ILCE-7M4 的 JPG 沒有這個欄位、ARW 只有 InternalSerialNumber，
    整個資料夾因此永遠得到 0 組連拍，而且不會有任何錯誤訊息。
    """

    SONY_RAW = {'Make': 'SONY', 'Model': 'ILCE-7M4',
                'InternalSerialNumber': '6eff00008a09'}
    SONY_JPG = {'Make': 'SONY', 'Model': 'ILCE-7M4'}
    CANON = {'Make': 'Canon', 'Model': 'EOS R6', 'SerialNumber': '123456'}

    def test_serial_number_is_preferred(self):
        self.assertEqual(camera_identity(self.CANON), ('serial', '123456'))

    def test_internal_serial_is_accepted_as_serial(self):
        """Sony 把序號放在 MakerNotes 的另一個欄位，準確度相同。"""
        self.assertEqual(camera_identity(self.SONY_RAW), ('serial', '6eff00008a09'))
        self.assertEqual(camera_identity(self.SONY_RAW, 'serial'),
                         ('serial', '6eff00008a09'))

    def test_model_fallback_only_in_model_mode(self):
        self.assertEqual(camera_identity(self.SONY_JPG),
                         ('model', 'SONY', 'ILCE-7M4'))
        self.assertIsNone(camera_identity(self.SONY_JPG, 'serial'))

    def test_serial_and_model_identified_photos_do_not_match(self):
        """同一台相機的 RAW 與 JPG，識別層級不同就不該被當成同一台。"""
        self.assertNotEqual(camera_identity(self.SONY_RAW),
                            camera_identity(self.SONY_JPG))

    def test_ignore_mode_matches_everything(self):
        self.assertEqual(camera_identity(self.SONY_JPG, 'ignore'),
                         camera_identity(self.CANON, 'ignore'))

    def test_unknown_camera_is_none(self):
        self.assertIsNone(camera_identity({'Make': 'SONY'}))

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            camera_identity(self.CANON, 'whatever')


class TestCaptureTime(unittest.TestCase):
    """
    小數秒的解析。

    釘住的缺陷：ExifTool 依相機寫出的小數秒位數不固定（Sony 實測兩位），
    而 Python 3.10 的 fromisoformat 只接受 3 位或 6 位。原本的寫法在 3.10 上
    會解析失敗並退回只精確到秒的欄位，同一秒內的連拍因此分不出先後——
    開發機是 3.10、批次管線開發者的 Mac 是 3.12，同一份程式碼會給出不同結果。
    """

    def test_two_digit_fraction(self):
        parsed = parse_capture_time(
            {'SubSecDateTimeOriginal': '2026:05:13 21:55:35.32+08:00'})
        self.assertIsNotNone(parsed, '兩位小數秒解析失敗')
        self.assertEqual(parsed.microsecond, 320000)
        self.assertIsNotNone(parsed.tzinfo)

    def test_six_digit_fraction_and_no_fraction(self):
        self.assertEqual(parse_capture_time(
            {'SubSecDateTimeOriginal': '2026:05:13 21:55:35.123456+08:00'}
        ).microsecond, 123456)
        self.assertEqual(parse_capture_time(
            {'DateTimeOriginal': '2026:05:13 21:55:35'}).microsecond, 0)

    def test_falls_back_to_datetime_original(self):
        self.assertIsNotNone(parse_capture_time(
            {'SubSecDateTimeOriginal': '壞掉的值',
             'DateTimeOriginal': '2026:05:13 21:55:35'}))

    def test_unparsable_returns_none(self):
        self.assertIsNone(parse_capture_time({'DateTimeOriginal': 'x'}))
        self.assertIsNone(parse_capture_time({}))


class TestBurstCheck(unittest.TestCase):
    """連拍判定：時間、相機、以及 --camera-match 的三種嚴格程度。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.a = str(Path(self.tmp.name) / 'a.jpg')
        self.b = str(Path(self.tmp.name) / 'b.jpg')
        self.far = str(Path(self.tmp.name) / 'far.jpg')

    def _check(self, camera_match='model', seconds=2.0):
        items = [
            {'SourceFile': self.a, 'Make': 'SONY', 'Model': 'ILCE-7M4',
             'SubSecDateTimeOriginal': '2026:05:13 21:55:35.32+08:00'},
            {'SourceFile': self.b, 'Make': 'SONY', 'Model': 'ILCE-7M4',
             'SubSecDateTimeOriginal': '2026:05:13 21:55:35.54+08:00'},
            {'SourceFile': self.far, 'Make': 'SONY', 'Model': 'ILCE-7M4',
             'SubSecDateTimeOriginal': '2026:05:13 22:10:00.00+08:00'},
        ]
        path = write_metadata(self.tmp.name, items)
        return build_burst_check(path, max_seconds=seconds,
                                 camera_match=camera_match)

    def test_model_mode_pairs_photos_without_serial(self):
        can_pair, stats = self._check('model')
        self.assertTrue(can_pair(self.a, self.b))
        self.assertEqual(stats['model'], 3)

    def test_serial_mode_rejects_photos_without_serial(self):
        can_pair, stats = self._check('serial')
        self.assertFalse(can_pair(self.a, self.b))
        self.assertEqual(stats['無法識別'], 3)

    def test_time_gap_still_applies(self):
        can_pair, _ = self._check('model')
        self.assertFalse(can_pair(self.a, self.far))

    def test_subsecond_precision_is_used(self):
        """
        a 與 b 在同一秒（相差 0.22 秒）。門檻設 0.1 秒時必須分開——
        小數秒若解析失敗，兩張會被視為同一秒而錯誤地配成一組。
        """
        can_pair, _ = self._check('model', seconds=0.1)
        self.assertFalse(can_pair(self.a, self.b))
        can_pair, _ = self._check('model', seconds=0.3)
        self.assertTrue(can_pair(self.a, self.b))

    def test_ignore_mode_still_requires_capture_time(self):
        items = [
            {'SourceFile': self.a, 'SubSecDateTimeOriginal':
                '2026:05:13 21:55:35.32+08:00'},
            {'SourceFile': self.b},
        ]
        path = write_metadata(self.tmp.name, items, 'meta2.json')
        can_pair, stats = build_burst_check(path, camera_match='ignore')
        self.assertFalse(can_pair(self.a, self.b))
        self.assertEqual(stats['無拍攝時間'], 1)


class TestDefaults(unittest.TestCase):
    """
    預設值來自 325 張照片的人工標註（見兩個常數的說明）。
    這個測試不是怕有人改動，而是確保改動時會看到「這些數字是量出來的」。
    """

    def test_similarity_threshold(self):
        self.assertAlmostEqual(photo_grouping.DEFAULT_THRESHOLD, 0.85, places=2)

    def test_burst_time_window(self):
        self.assertAlmostEqual(burst_metadata.DEFAULT_MAX_SECONDS, 5.0, places=1)


class TestGroupPhotos(unittest.TestCase):
    """只看內容的分組，不需要任何 EXIF。"""

    def test_groups_by_similarity_and_picks_highest_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_jsonl(tmp, [
                record('a.jpg', 50.0, feature(1.0)),
                record('b.jpg', 60.0, feature(1.02)),     # 與 a 幾乎相同
                record('c.jpg', 70.0, feature(-400.0)),   # 明顯不同
            ])
            groups = group_photos(path, threshold=0.9)
        sizes = sorted(g['count'] for g in groups)
        self.assertEqual(sizes, [1, 2])
        pair = next(g for g in groups if g['count'] == 2)
        self.assertEqual(Path(pair['best_path']).name, 'b.jpg')

    def test_mixed_weights_are_rejected(self):
        """不同權重的綜合分不能一起排名，否則挑出來的「最佳」沒有意義。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = write_jsonl(tmp, [
                record('a.jpg', 50.0, feature(1.0), weight=0.6),
                record('b.jpg', 60.0, feature(1.0), weight=0.8),
            ])
            with self.assertRaises(ValueError):
                group_photos(path)


class TestPickBest(unittest.TestCase):
    """
    使用者自己指定一組連拍，只做排名與挑選。
    這個模式不依賴 EXIF，也不依賴相似度門檻——照片再不像也不會被排除。
    """

    def test_ranks_by_overall_score_and_marks_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_jsonl(tmp, [
                record('a.jpg', 50.0, feature(1.0)),
                record('b.jpg', 72.5, feature(1.01)),
                record('c.jpg', 61.0, feature(1.02)),
            ])
            ranked = rank_photos(path)
        self.assertEqual([Path(p['path']).name for p in ranked],
                         ['b.jpg', 'c.jpg', 'a.jpg'])
        self.assertTrue(ranked[0]['is_best'])
        self.assertEqual(sum(p['is_best'] for p in ranked), 1)
        self.assertAlmostEqual(ranked[0]['similarity_to_best'], 1.0, places=6)

    def test_dissimilar_photo_is_kept_but_flagged(self):
        """相似度只用來提醒，不可以把照片排除掉。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = write_jsonl(tmp, [
                record('a.jpg', 80.0, feature(1.0)),
                record('odd.jpg', 40.0, feature(-400.0)),
            ])
            ranked = rank_photos(path)
        self.assertEqual(len(ranked), 2)
        odd = next(p for p in ranked if Path(p['path']).name == 'odd.jpg')
        self.assertLess(odd['similarity_to_best'], 0.9)

    def test_failed_records_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_jsonl(tmp, [
                record('a.jpg', 50.0, feature(1.0)),
                {'path': 'broken.arw', 'ok': False, 'stage': 'decode',
                 'error': 'x', 'aesthetic_weight': 0.6, 'technical_weight': 0.4},
            ])
            ranked = rank_photos(path)
        self.assertEqual([Path(p['path']).name for p in ranked], ['a.jpg'])


if __name__ == '__main__':
    unittest.main()
