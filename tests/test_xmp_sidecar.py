"""
XMP 星等寫入（raw_processor.py、rebuild_xmp_sidecars.py）的測試。

這組模組會在使用者的照片資料夾裡寫檔，出錯的代價是弄壞照片或 Lightroom 的調色，
而且通常不會有任何錯誤訊息。每一項都對應 2026-10-01 核對時實測過的情況：

  * 原始照片一個位元組都不能動。
  * 已有 sidecar 時只改 Rating，Lightroom 的調色與色標要保留。
  * RAW+JPG 同時拍攝時檔名相同，JPG 不可蓋掉 RAW 的星等
    （實測 DSC04606.ARW 的 4 星被同名 JPG 蓋成 2 星）。
  * 沒有 ExifTool 時不可自己改 sidecar（實測會寫出無法解析的 XML，卻回報成功）。

raw_processor 會 import ai_inference（連帶載入模型），所以整組掛 @requires_weights，
並在測試內才 import。照片一律用假的位元組，不需要真的解碼。
"""
import os
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import numpy as np

from tests._util import make_image, requires_weights

XMP_RATING = '{http://ns.adobe.com/xap/1.0/}Rating'
XMP_LABEL = '{http://ns.adobe.com/xap/1.0/}Label'
CRS_EXPOSURE = '{http://ns.adobe.com/camera-raw-settings/1.0/}Exposure2012'

# Lightroom 寫出的 sidecar 的樣子：Rating 是屬性，調色參數在 crs 命名空間
LIGHTROOM_SIDECAR = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
   xmp:Rating="1"
   xmp:Label="Red"
   crs:Exposure2012="+0.65"/>
 </rdf:RDF>
</x:xmpmeta>"""

# ExifTool 改寫過之後的樣子：Rating 變成元素，命名空間宣告在另一個 Description 上。
# 舊版的文字替換備援就是在這種檔案上寫出未宣告的前綴。
EXIFTOOL_SIDECAR = """<x:xmpmeta xmlns:x='adobe:ns:meta/'>
<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
 <rdf:Description rdf:about=''
  xmlns:crs='http://ns.adobe.com/camera-raw-settings/1.0/'>
  <crs:Exposure2012>+0.65</crs:Exposure2012>
 </rdf:Description>
 <rdf:Description rdf:about=''
  xmlns:xmp='http://ns.adobe.com/xap/1.0/'>
  <xmp:Rating>4</xmp:Rating>
 </rdf:Description>
</rdf:RDF>
</x:xmpmeta>"""

FAKE_RAW = b'FAKE-RAW-BYTES\x00\x01\x02' * 64

requires_exiftool = unittest.skipUnless(shutil.which('exiftool'), '未安裝 ExifTool')


def xmp_fields(path):
    """解析 .xmp，回傳 {完整名稱: 值}。屬性與元素兩種寫法都收；XML 壞掉時直接拋錯。"""
    fields = {}
    for el in ET.parse(path).getroot().iter():
        fields.update(el.attrib)
        if len(el) == 0 and el.text and el.text.strip():
            fields[el.tag] = el.text.strip()
    return fields


@requires_weights
class TestSafeUpdateXmp(unittest.TestCase):

    def setUp(self):
        from raw_processor import RawProcessor
        self.rp = RawProcessor
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _photo(self, name, content=FAKE_RAW):
        path = self.dir / name
        path.write_bytes(content)
        return path

    def test_new_sidecar_leaves_photo_untouched(self):
        photo = self._photo('DSC0001.ARW')
        ok, rating = self.rp.safe_update_xmp(str(photo), 70.0)

        self.assertTrue(ok)
        self.assertEqual(rating, 5)
        self.assertEqual(photo.read_bytes(), FAKE_RAW, '原始照片被改到了')
        self.assertEqual(xmp_fields(self.dir / 'DSC0001.xmp')[XMP_RATING], '5')

    @requires_exiftool
    def test_existing_sidecar_keeps_lightroom_edits(self):
        photo = self._photo('DSC0002.ARW')
        sidecar = self.dir / 'DSC0002.xmp'
        sidecar.write_text(LIGHTROOM_SIDECAR, encoding='utf-8')

        ok, rating = self.rp.safe_update_xmp(str(photo), 55.0)

        self.assertTrue(ok)
        fields = xmp_fields(sidecar)
        self.assertEqual(fields[XMP_RATING], str(rating))
        self.assertEqual(fields[XMP_LABEL], 'Red', 'Lightroom 的色標不見了')
        self.assertEqual(fields[CRS_EXPOSURE], '+0.65', 'Lightroom 的調色不見了')
        self.assertEqual(photo.read_bytes(), FAKE_RAW, '原始照片被改到了')

    def test_without_exiftool_existing_sidecar_is_not_touched(self):
        for name, content in (('A', EXIFTOOL_SIDECAR), ('B', LIGHTROOM_SIDECAR)):
            with self.subTest(sidecar=name):
                photo = self._photo(f'{name}.ARW')
                sidecar = self.dir / f'{name}.xmp'
                sidecar.write_text(content, encoding='utf-8')
                before = sidecar.read_bytes()

                with mock.patch('raw_processor.shutil.which', return_value=None):
                    ok, _ = self.rp.safe_update_xmp(str(photo), 70.0)

                self.assertFalse(ok, '沒有 ExifTool 卻回報成功')
                self.assertEqual(sidecar.read_bytes(), before, '沒有 ExifTool 時不該改 sidecar')

    def test_exiftool_failure_is_reported_and_sidecar_kept(self):
        photo = self._photo('DSC0003.ARW')
        sidecar = self.dir / 'DSC0003.xmp'
        sidecar.write_text(EXIFTOOL_SIDECAR, encoding='utf-8')
        before = sidecar.read_bytes()
        failed = subprocess.CompletedProcess([], returncode=1, stdout='', stderr='boom')

        with mock.patch('raw_processor.shutil.which', return_value='exiftool'), \
                mock.patch('raw_processor.subprocess.run', return_value=failed):
            ok, _ = self.rp.safe_update_xmp(str(photo), 70.0)

        self.assertFalse(ok, 'ExifTool 失敗卻回報成功')
        self.assertEqual(sidecar.read_bytes(), before)

    def test_new_sidecar_works_without_exiftool(self):
        photo = self._photo('DSC0004.ARW')
        with mock.patch('raw_processor.shutil.which', return_value=None):
            ok, _ = self.rp.safe_update_xmp(str(photo), 40.0)
        self.assertTrue(ok, '新建 sidecar 不需要 ExifTool')
        self.assertEqual(xmp_fields(self.dir / 'DSC0004.xmp')[XMP_RATING], '1')

    def test_non_raw_gets_no_sidecar(self):
        for name in ('IMG_0001.jpg', 'IMG_0002.JPEG', 'scan.png', 'DSC0005.dng'):
            with self.subTest(photo=name):
                photo = self._photo(name)
                ok, _ = self.rp.safe_update_xmp(str(photo), 70.0)
                self.assertFalse(ok)
                self.assertFalse((self.dir / (Path(name).stem + '.xmp')).exists(),
                                 f'{name} 不該有 sidecar：Lightroom 不讀')

    def test_rating_boundaries(self):
        # 2026-10-04 依 935 張實拍重訂（見 raw_processor.RATING_THRESHOLDS）
        cases = [(69, 5), (68.99, 4), (64, 4), (63.99, 3), (54, 3),
                 (53.99, 2), (47, 2), (46.99, 1), (0, 1)]
        for score, stars in cases:
            with self.subTest(score=score):
                self.assertEqual(self.rp.map_score_to_rating(score), stars)



@requires_weights
class TestUpdateRatings(unittest.TestCase):
    """
    批次寫星等（RawProcessor.update_ratings）。

    逐張修改已有的 .xmp 時，每張都要啟動一次 ExifTool（約 0.4 秒，325 張將近 2 分鐘）。
    批次版把同一個星等的照片交給同一次 ExifTool；結果必須和逐張一樣，而且不能因此弄壞任何檔案。
    """

    def setUp(self):
        from raw_processor import RawProcessor
        self.rp = RawProcessor
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _photo(self, name, sidecar=True):
        photo = self.dir / name
        photo.write_bytes(FAKE_RAW)
        if sidecar:
            photo.with_suffix('.xmp').write_text(LIGHTROOM_SIDECAR, encoding='utf-8')
        return str(photo)

    @requires_exiftool
    def test_one_exiftool_call_per_rating_and_edits_kept(self):
        items = [(self._photo('A1.ARW'), 70.0), (self._photo('A2.ARW'), 72.0),
                 (self._photo('中文檔名.ARW'), 75.0),                  # 5 星
                 (self._photo('B1.ARW'), 60.0), (self._photo('B2.ARW'), 58.0),   # 3 星
                 (self._photo('NEW.ARW', sidecar=False), 50.0)]                # 新建，2 星
        real_run = subprocess.run
        calls = []

        def counting_run(cmd, *args, **kwargs):
            calls.append(cmd)
            return real_run(cmd, *args, **kwargs)

        with mock.patch('raw_processor.subprocess.run', side_effect=counting_run):
            results = self.rp.update_ratings(items)

        self.assertEqual(results, [(True, 5), (True, 5), (True, 5), (True, 3), (True, 3), (True, 2)])
        self.assertEqual(len(calls), 2, '兩個星等各一次，不可每張啟動一次 ExifTool')
        for (path, _), (_, stars) in zip(items, results):
            fields = xmp_fields(Path(path).with_suffix('.xmp'))
            self.assertEqual(fields[XMP_RATING], str(stars), path)
            if not path.endswith('NEW.ARW'):
                self.assertEqual(fields[CRS_EXPOSURE], '+0.65', 'Lightroom 的調色不見了')
                self.assertEqual(fields[XMP_LABEL], 'Red', 'Lightroom 的色標不見了')
            self.assertEqual(Path(path).read_bytes(), FAKE_RAW, '原始照片被改到了')

    def test_failed_group_falls_back_to_one_by_one(self):
        items = [(self._photo('A.ARW'), 70.0), (self._photo('B.ARW'), 72.0)]
        before = [Path(p).with_suffix('.xmp').read_bytes() for p, _ in items]
        failed = subprocess.CompletedProcess([], returncode=1, stdout=b'', stderr=b'boom')
        with mock.patch('raw_processor.shutil.which', return_value='exiftool'), \
                mock.patch('raw_processor.subprocess.run', return_value=failed) as run, \
                mock.patch('builtins.print'):
            errors = []
            results = self.rp.update_ratings(items, errors=errors)
        self.assertEqual(results, [(False, 5), (False, 5)])
        self.assertEqual(run.call_count, 3, '整組一次，失敗後逐張各一次')
        self.assertEqual([Path(p).with_suffix('.xmp').read_bytes() for p, _ in items], before)
        self.assertEqual([reason for _, reason in errors], ['ExifTool 錯誤：boom'] * 2,
                         '失敗原因要回傳給畫面顯示（雙擊啟動時沒有主控台）')

    def test_without_exiftool_only_new_sidecars_are_written(self):
        items = [(self._photo('OLD.ARW'), 70.0), (self._photo('NEW.ARW', sidecar=False), 70.0),
                 (self._photo('IMG.jpg', sidecar=False), 70.0)]
        before = Path(items[0][0]).with_suffix('.xmp').read_bytes()
        with mock.patch('raw_processor.shutil.which', return_value=None), mock.patch('builtins.print'):
            errors = []
            results = self.rp.update_ratings(items, errors=errors)
        self.assertEqual(results, [(False, 5), (True, 5), (False, 5)])
        self.assertEqual(Path(items[0][0]).with_suffix('.xmp').read_bytes(), before)
        self.assertFalse((self.dir / 'IMG.xmp').exists(), 'JPG 不寫 sidecar')
        self.assertEqual([(Path(p).name, 'ExifTool' in r) for p, r in errors],
                         [('IMG.jpg', False), ('OLD.ARW', True)])


@requires_weights
class TestDecodeMatchesInference(unittest.TestCase):
    """顯示與寫星等用的影像必須和評分時完全相同（包含 JPG 依 EXIF 轉正）。"""

    def test_rotated_jpg_decodes_like_load_image(self):
        import ai_inference
        from PIL import Image
        from raw_processor import RawProcessor

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rotated.jpg'
            img = Image.fromarray(make_image(64, 48, kind='photo_like'))
            exif = img.getexif()
            exif[0x0112] = 6          # 顯示時要順時針轉 90°
            img.save(path, exif=exif)

            ours = RawProcessor.decode_image_rgb(str(path))
            theirs = ai_inference.load_image(str(path))

        self.assertEqual(ours.shape, (64, 48, 3), 'JPG 沒有依 EXIF 轉正')
        np.testing.assert_array_equal(ours, theirs)


@requires_weights
class TestBatchRefresh(unittest.TestCase):
    """rebuild_xmp_sidecars.batch_refresh_xmp：解碼與評分都換成假的，只測寫檔行為。"""

    SCORES = {'DSC0001.ARW': 60.0, 'DSC0001.JPG': 40.0, 'IMG_0002.jpg': 70.0,
              'DSC0003.NEF': 66.0}

    def _fake_evaluate(self, path, image=None, **kwargs):
        return {'overall_score': self.SCORES[os.path.basename(path)]}

    def test_raw_jpg_pair_keeps_raw_rating_and_photos_untouched(self):
        import ai_inference
        from raw_processor import RawProcessor
        from rebuild_xmp_sidecars import batch_refresh_xmp

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for name in self.SCORES:
                (base / name).write_bytes(FAKE_RAW + name.encode())

            with mock.patch.object(ai_inference, 'evaluate_photo', self._fake_evaluate), \
                    mock.patch.object(RawProcessor, 'decode_image_rgb',
                                      staticmethod(lambda p: np.zeros((8, 8, 3), np.uint8))), \
                    mock.patch('builtins.print'):
                batch_refresh_xmp(str(base))

            # 60 分是 3 星；同名 JPG 的 40 分（1 星）不可蓋掉它
            self.assertEqual(xmp_fields(base / 'DSC0001.xmp')[XMP_RATING], '3')
            self.assertEqual(xmp_fields(base / 'DSC0003.xmp')[XMP_RATING], '4')
            self.assertFalse((base / 'IMG_0002.xmp').exists(), 'JPG 不該有 sidecar')
            for name in self.SCORES:
                self.assertEqual((base / name).read_bytes(), FAKE_RAW + name.encode(),
                                 f'{name} 被改到了')


if __name__ == '__main__':
    unittest.main()
