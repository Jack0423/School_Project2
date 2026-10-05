"""
相似照片分組的底層（F06）：特徵與拍攝時間存進資料庫，前台用 main_v2.group_folder 直接分組。

守的是：
  * 舊的 photos.db 沒有新欄位，CREATE TABLE IF NOT EXISTS 不會補；沒補的話一寫入就是 no such column。
  * 特徵和 model_version 綁在一起：重新分析沒有特徵時要清掉舊的，不能留著別的模型算的。
  * 前台分組與命令列工具（photo_grouping.py）對同一批照片要分出一樣的組。
  * 連拍：時間、相機都要對得上；讀不到拍攝時間的照片要列出來，不能默默變成單張。
  * 「開始批次分析」要把沒有特徵、沒讀過拍攝時間的照片補跑，否則舊資料永遠不能分組。
  * ExifTool 讀拍攝時間：中文檔名、沒有 EXIF 的圖、ExifTool 壞掉三種情況。
"""
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import burst_metadata
from tests._util import requires_weights

requires_exiftool = unittest.skipUnless(shutil.which('exiftool'), '未安裝 ExifTool')


def _pyqt6_available():
    try:
        import PyQt6.QtWidgets  # noqa: F401
        return True
    except ImportError:
        return False


requires_pyqt6 = unittest.skipUnless(_pyqt6_available(), '未安裝 PyQt6')

RNG = np.random.default_rng(20261005)
BASE = RNG.normal(size=1280)
OTHER = RNG.normal(size=1280)          # 和 BASE 的餘弦相似度約 0


def near(base, scale=0.05):
    """和 base 的相似度約 0.999"""
    return (base + RNG.normal(scale=scale, size=1280)).tolist()


def write_exif_jpg(path, model='ILCE-7M3', when='2026:05:13 21:55:35', subsec='32'):
    exif = Image.Exif()
    exif[0x010F] = 'SONY'
    exif[0x0110] = model
    exif[0x8769] = {0x9003: when, 0x9291: subsec, 0x9011: '+08:00'}
    Image.new('RGB', (8, 8), 'red').save(path, exif=exif)


class TestReadMetadata(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    @requires_exiftool
    def test_reads_capture_time_with_chinese_name(self):
        photo = self.dir / '連拍一.jpg'
        write_exif_jpg(photo)
        Image.new('RGB', (8, 8)).save(self.dir / 'screenshot.png')
        paths = [str(photo), str(self.dir / 'screenshot.png'), str(self.dir / 'missing.jpg')]

        found = burst_metadata.read_metadata(paths)

        self.assertEqual(set(found), set(paths[:2]), '不存在的檔案不列；存在的都要列')
        self.assertEqual(found[paths[1]], {}, '沒有 EXIF 的圖記成空的（讀過了，就是沒有）')
        item = found[paths[0]]
        self.assertEqual(str(burst_metadata.parse_capture_time(item)), '2026-05-13 21:55:35.320000+08:00')
        self.assertEqual(burst_metadata.camera_identity(item), ('model', 'SONY', 'ILCE-7M3'))

    def test_without_exiftool_returns_none(self):
        with mock.patch('burst_metadata.shutil.which', return_value=None):
            self.assertIsNone(burst_metadata.read_metadata([str(self.dir / 'a.jpg')]))

    def test_broken_exiftool_is_an_error_not_empty(self):
        """ExifTool 整個壞掉時不能回傳「每張都沒有 EXIF」，那會被存起來、之後再也不讀。"""
        photo = self.dir / 'a.jpg'
        write_exif_jpg(photo)
        broken = subprocess.CompletedProcess([], returncode=1, stdout=b'', stderr=b'Can\'t locate')
        with mock.patch('burst_metadata.shutil.which', return_value='exiftool'), \
                mock.patch('burst_metadata.subprocess.run', return_value=broken):
            with self.assertRaises(RuntimeError):
                burst_metadata.read_metadata([str(photo)])

    def test_burst_check_accepts_metadata_already_read(self):
        """前台從資料庫讀出來的 dict 和命令列的 ExifTool 匯出檔，判斷結果要相同。"""
        a, b, far = (str(self.dir / n) for n in ('a.jpg', 'b.jpg', 'far.jpg'))
        items = {a: {'Make': 'SONY', 'Model': 'X', 'SubSecDateTimeOriginal': '2026:05:13 21:55:35.32+08:00'},
                 b: {'Make': 'SONY', 'Model': 'X', 'SubSecDateTimeOriginal': '2026:05:13 21:55:37.10+08:00'},
                 far: {'Make': 'SONY', 'Model': 'X', 'SubSecDateTimeOriginal': '2026:05:13 22:10:00.00+08:00'}}
        exported = self.dir / 'meta.json'
        exported.write_text(json.dumps([dict(item, SourceFile=path) for path, item in items.items()]),
                            encoding='utf-8')
        from_dict, stats_dict = burst_metadata.build_burst_check(items)
        from_file, stats_file = burst_metadata.build_burst_check(str(exported))
        for x, y in ((a, b), (a, far), (b, far)):
            self.assertEqual(from_dict(x, y), from_file(x, y))
        self.assertEqual(stats_dict, stats_file)
        self.assertTrue(from_dict(a, b))
        self.assertFalse(from_dict(a, far))


@requires_weights
@requires_pyqt6
class TestFolderGrouping(unittest.TestCase):

    def setUp(self):
        import main_v2
        self.m = main_v2
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name) / 'photos'
        self.folder.mkdir()
        patcher = mock.patch.object(main_v2, 'DB_PATH', str(Path(self.tmp.name) / 'photos.db'))
        patcher.start()
        self.addCleanup(patcher.stop)
        main_v2.init_db()

    def _add(self, name, aes=None, tech=60.0, feature=None):
        """新增一張照片；給了 aes 就當成已分析（feature=None 表示這次分析沒有特徵）。"""
        path = self.folder / name
        path.write_bytes(b'fake')
        self.m.insert_photo(name, str(path))
        if aes is not None:
            result = {'aesthetic_score': aes, 'technical_score': tech,
                      'suggestion': '', 'technical_issues': []}
            if feature is not None:
                result['feature_vector'] = feature
            self.m.update_analysis(str(path), result, 0.6)
        return os.path.abspath(path)

    def _names(self, paths):
        return [Path(p).name for p in paths]

    # ── 資料庫欄位 ──────────────────────────────────────
    def test_old_database_gets_new_columns(self):
        db = Path(self.tmp.name) / 'old.db'
        con = sqlite3.connect(db)
        con.execute("""CREATE TABLE photos_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT, file_name TEXT NOT NULL, file_path TEXT NOT NULL UNIQUE,
            aesthetic_score REAL, technical_score REAL, overall_score REAL,
            aesthetic_weight REAL DEFAULT 0.6, technical_weight REAL DEFAULT 0.4, model_version TEXT DEFAULT '',
            status TEXT DEFAULT '', suggestion TEXT DEFAULT '', action TEXT DEFAULT '',
            technical_issues TEXT DEFAULT '', is_best INTEGER DEFAULT 0, analyzed INTEGER DEFAULT 0)""")
        con.execute("INSERT INTO photos_v2 (file_name, file_path, analyzed) VALUES ('a.ARW', 'C:/a.ARW', 1)")
        con.commit()
        con.close()

        with mock.patch.object(self.m, 'DB_PATH', str(db)):
            self.m.init_db()
            self.m.init_db()      # 第二次不能再加一次
        con = sqlite3.connect(db)
        columns = [row[1] for row in con.execute('PRAGMA table_info(photos_v2)')]
        rows = con.execute('SELECT file_name, analyzed, feature, capture_meta FROM photos_v2').fetchall()
        con.close()
        self.assertEqual(columns[-2:], ['feature', 'capture_meta'])
        self.assertEqual(rows, [('a.ARW', 1, None, None)], '原本的資料要保留')

    def test_feature_round_trip_and_cleared_when_missing(self):
        vector = near(BASE)
        path = self._add('a.ARW', 60.0, feature=vector)
        con = sqlite3.connect(self.m.DB_PATH)
        blob = con.execute('SELECT feature FROM photos_v2 WHERE file_path=?', (path,)).fetchone()[0]
        self.assertEqual(len(blob), 1280 * 4)
        np.testing.assert_array_equal(np.frombuffer(blob, dtype='<f4'), np.asarray(vector, dtype=np.float32))

        self.m.update_analysis(path, {'aesthetic_score': 50.0, 'technical_score': 50.0}, 0.6)
        blob = con.execute('SELECT feature FROM photos_v2 WHERE file_path=?', (path,)).fetchone()[0]
        con.close()
        self.assertIsNone(blob, '這次分析沒有特徵時要清掉，不能留著上一次的')

    def test_bad_feature_is_not_stored(self):
        self.assertIsNone(self.m.feature_blob([1.0] * 10))
        self.assertIsNone(self.m.feature_blob([float('nan')] * 1280))

    # ── 相似分組 ────────────────────────────────────────
    def _similar_folder(self):
        paths = {
            'a.ARW': self._add('a.ARW', 70.0, feature=near(BASE)),
            'b.ARW': self._add('b.ARW', 80.0, feature=near(BASE)),     # a、b、c 一組，b 最高分
            'c.ARW': self._add('c.ARW', 50.0, feature=near(BASE)),
            'd.ARW': self._add('d.ARW', 90.0, feature=near(OTHER)),    # e 和 d 一組，d 最高分（整體也最高）
            'e.ARW': self._add('e.ARW', 40.0, feature=near(OTHER)),
            'f.ARW': self._add('f.ARW', 60.0, feature=RNG.normal(size=1280).tolist()),   # 單張
            'g.ARW': self._add('g.ARW'),                                 # 還沒分析
            'h.ARW': self._add('h.ARW', 60.0, feature=near(BASE)),       # 舊模型版本
            'i.ARW': self._add('i.ARW', 60.0),                           # 分組功能之前分析的，沒有特徵
        }
        con = sqlite3.connect(self.m.DB_PATH)
        con.execute("UPDATE photos_v2 SET model_version='舊版' WHERE file_path=?", (paths['h.ARW'],))
        con.commit()
        con.close()
        self.m.recompute_scores(0.6, str(self.folder))
        return paths

    def test_similar_groups(self):
        self._similar_folder()
        result = self.m.group_folder(str(self.folder))

        self.assertEqual([g['group_id'] for g in result['groups']], [1, 2], '只列 2 張以上的組，從 1 連號')
        first, second = result['groups']
        self.assertEqual(Path(first['best_path']).name, 'd.ARW', '組依組內最高分排序')
        self.assertEqual([p['file_name'] for p in first['photos']], ['d.ARW', 'e.ARW'])
        self.assertEqual([p['file_name'] for p in second['photos']], ['b.ARW', 'a.ARW', 'c.ARW'])
        self.assertEqual([p['is_best'] for p in second['photos']], [True, False, False])
        self.assertEqual(self._names(result['singles']), ['f.ARW'])
        self.assertEqual(self._names(result['not_ready']), ['g.ARW', 'h.ARW', 'i.ARW'])
        self.assertEqual(self._names(self.m.non_best_paths(result)), ['e.ARW', 'a.ARW', 'c.ARW'])
        self.assertEqual(result['no_capture_time'], [], '相似模式不看拍攝時間')

    def test_same_groups_as_command_line(self):
        """前台與 photo_grouping.py 對同一批照片要分出一樣的組。"""
        import photo_grouping
        self._similar_folder()
        result = self.m.group_folder(str(self.folder))

        con = sqlite3.connect(self.m.DB_PATH)
        rows = con.execute('SELECT file_path, overall_score, feature, model_version, aesthetic_weight '
                           'FROM photos_v2 WHERE feature IS NOT NULL').fetchall()
        con.close()
        jsonl = Path(self.tmp.name) / 'batch.jsonl'
        with open(jsonl, 'w', encoding='utf-8') as f:
            for path, score, blob, version, weight in rows:
                if version != self.m.MODEL_VERSION:
                    continue
                f.write(json.dumps({'path': path, 'ok': True, 'overall_score': score,
                                    'aesthetic_weight': weight, 'technical_weight': round(1 - weight, 2),
                                    'feature_vector': np.frombuffer(blob, dtype='<f4').tolist()}) + '\n')
        cli = [g for g in photo_grouping.group_photos(str(jsonl)) if g['count'] > 1]

        self.assertEqual([[p['path'] for p in g['photos']] for g in cli],
                         [[p['path'] for p in g['photos']] for g in result['groups']])
        self.assertEqual([g['best_path'] for g in cli], [g['best_path'] for g in result['groups']])

    def test_weight_changes_best_photo(self):
        """組內建議保留的是目前權重下綜合分最高的。"""
        a = self._add('a.ARW', 80.0, tech=40.0, feature=near(BASE))    # 美感 0.6 → 64，0.9 → 76
        b = self._add('b.ARW', 60.0, tech=75.0, feature=near(BASE))    # 美感 0.6 → 66，0.9 → 61.5
        self.m.recompute_scores(0.6, str(self.folder))
        self.assertEqual(self.m.group_folder(str(self.folder))['groups'][0]['best_path'], b)
        self.m.recompute_scores(0.9, str(self.folder))
        self.assertEqual(self.m.group_folder(str(self.folder))['groups'][0]['best_path'], a)

    def test_other_folders_and_deleted_files_are_ignored(self):
        self._add('a.ARW', 70.0, feature=near(BASE))
        gone = self._add('b.ARW', 60.0, feature=near(BASE))
        os.remove(gone)
        other = Path(self.tmp.name) / 'other'
        other.mkdir()
        (other / 'c.ARW').write_bytes(b'fake')
        self.m.insert_photo('c.ARW', str(other / 'c.ARW'))
        self.m.update_analysis(str(other / 'c.ARW'), {'aesthetic_score': 70.0, 'technical_score': 60.0,
                                                      'feature_vector': near(BASE)}, 0.6)
        result = self.m.group_folder(str(self.folder))
        self.assertEqual(result['groups'], [])
        self.assertEqual(self._names(result['singles']), ['a.ARW'])

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            self.m.group_folder(str(self.folder), mode='scene')

    # ── 連拍分組 ────────────────────────────────────────
    def test_burst_needs_same_camera_and_time(self):
        def meta(seconds, model='ILCE-7M3'):
            return {'Make': 'SONY', 'Model': model,
                    'SubSecDateTimeOriginal': f'2026:05:13 21:55:{seconds:05.2f}+08:00'}

        paths = {name: self._add(name, score, feature=near(BASE))
                 for name, score in (('a.ARW', 70.0), ('b.ARW', 80.0), ('late.ARW', 90.0),
                                     ('other_cam.ARW', 85.0), ('no_exif.ARW', 75.0), ('unread.ARW', 65.0))}
        self.m.recompute_scores(0.6, str(self.folder))
        self.m.store_capture_metadata({
            paths['a.ARW']: meta(30.10), paths['b.ARW']: meta(32.50),
            paths['late.ARW']: meta(50.00),                          # 20 秒後
            paths['other_cam.ARW']: meta(31.00, model='ILCE-7M4'),   # 同一時間、另一台相機
            paths['no_exif.ARW']: {},                                # 讀過了，沒有 EXIF
        })                                                           # unread.ARW 還沒讀過

        similar = self.m.group_folder(str(self.folder))
        self.assertEqual(similar['groups'][0]['count'], 6, '只看內容的話全部都是同一組')

        burst = self.m.group_folder(str(self.folder), mode='burst')
        self.assertEqual([[p['file_name'] for p in g['photos']] for g in burst['groups']], [['b.ARW', 'a.ARW']])
        self.assertEqual(self._names(burst['singles']),
                         ['late.ARW', 'no_exif.ARW', 'other_cam.ARW', 'unread.ARW'])
        self.assertEqual(self._names(burst['no_capture_time']), ['no_exif.ARW', 'unread.ARW'])
        self.assertFalse(burst['exiftool_missing'] and shutil.which('exiftool'))

        with mock.patch.object(self.m.shutil, 'which', return_value=None):
            self.assertTrue(self.m.group_folder(str(self.folder), mode='burst')['exiftool_missing'],
                            '有照片沒讀過拍攝時間、又沒有 ExifTool，要提示安裝')
        self.m.store_capture_metadata({paths['unread.ARW']: {}})
        with mock.patch.object(self.m.shutil, 'which', return_value=None):
            self.assertFalse(self.m.group_folder(str(self.folder), mode='burst')['exiftool_missing'],
                             '都讀過了就不需要 ExifTool')

    # ── 「開始批次分析」要補跑的照片 ────────────────────
    def test_analysis_candidates(self):
        done = self._add('done.ARW', 60.0, feature=near(BASE))
        no_meta = self._add('no_meta.ARW', 60.0, feature=near(BASE))
        no_feature = self._add('no_feature.ARW', 60.0)
        new = self._add('new.ARW')
        old = self._add('old.ARW', 60.0, feature=near(BASE))
        self.m.store_capture_metadata({done: {}, no_feature: {}, old: {}})
        con = sqlite3.connect(self.m.DB_PATH)
        con.execute("UPDATE photos_v2 SET model_version='舊版' WHERE file_path=?", (old,))
        con.commit()
        con.close()
        order = [done, no_meta, no_feature, new, old]

        with mock.patch.object(self.m.shutil, 'which', return_value='exiftool'):
            self.assertEqual(self._names(self.m.analysis_candidates(order)),
                             ['no_meta.ARW', 'no_feature.ARW', 'new.ARW', 'old.ARW'])
        with mock.patch.object(self.m.shutil, 'which', return_value=None):
            self.assertEqual(self._names(self.m.analysis_candidates(order)),
                             ['no_feature.ARW', 'new.ARW', 'old.ARW'],
                             '沒有 ExifTool 時重跑也讀不到拍攝時間，不要每次都重跑')
        self.assertEqual(self._names(self.m.paths_missing_capture_metadata(order)),
                         ['no_meta.ARW', 'new.ARW'])


if __name__ == '__main__':
    unittest.main()
