"""Orientation and fallback tests don't initialize the inference models."""
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image, ImageOps
from grouping_thumbnails import load_thumbnail


class ThumbnailTests(unittest.TestCase):
    def setUp(self):
        self.ai = types.SimpleNamespace(RAW_EXTENSIONS={'.arw'},
            load_image=lambda path: np.zeros((60, 90, 3), np.uint8))
        self.modules = patch.dict('sys.modules', {'ai_inference': self.ai})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def raw_module(self, flip=0, orientation=None, fail=False, bitmap=False):
        image = Image.new('RGB', (90, 60), 'red')
        data = BytesIO()
        exif = Image.Exif()
        if orientation:
            exif[274] = orientation
        image.save(data, format='JPEG', exif=exif)
        class Raw:
            sizes = types.SimpleNamespace(flip=flip)
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def extract_thumb(self):
                if fail: raise ValueError('no embedded thumbnail')
                return types.SimpleNamespace(format=2 if bitmap else 1,
                    data=np.array(image) if bitmap else data.getvalue())
        return types.SimpleNamespace(imread=lambda path: Raw(),
                                     ThumbFormat=types.SimpleNamespace(JPEG=1))

    def test_jpg_uses_standard_loader_and_is_small(self):
        with patch.object(self.ai, 'load_image', wraps=self.ai.load_image) as loader:
            result = load_thumbnail('test.jpg', (40, 40))
        loader.assert_called_once_with('test.jpg')
        self.assertEqual(result.size, (40, 27))

    def test_portrait_jpg_preserves_loader_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/'portrait.jpg')
            image = Image.new('RGB', (90, 60))
            exif = Image.Exif(); exif[274] = 6
            image.save(path, exif=exif)
            def standard_loader(p):
                with Image.open(p) as im:
                    return np.array(ImageOps.exif_transpose(im).convert('RGB'))
            self.ai.load_image = standard_loader
            result = load_thumbnail(path)
        self.assertEqual(result.size, (60, 90))

    def test_raw_embedded_jpeg_without_full_decode(self):
        with patch.dict('sys.modules', {'rawpy': self.raw_module()}), patch.object(
                self.ai, 'load_image', side_effect=AssertionError('full decode')):
            result = load_thumbnail('test.ARW')
        self.assertEqual(result.size, (90, 60))

    def test_raw_portrait_libraw_flip(self):
        for flip in (5, 6):
            with self.subTest(flip=flip), patch.dict('sys.modules', {'rawpy': self.raw_module(flip)}):
                self.assertEqual(load_thumbnail('test.arw').size, (60, 90))

    def test_embedded_exif_not_rotated_twice(self):
        with patch.dict('sys.modules', {'rawpy': self.raw_module(6, orientation=6)}):
            self.assertEqual(load_thumbnail('test.arw').size, (60, 90))

    def test_raw_bitmap_and_fallback(self):
        with patch.dict('sys.modules', {'rawpy': self.raw_module(5, bitmap=True)}):
            self.assertEqual(load_thumbnail('test.arw').size, (60, 90))
        with patch.dict('sys.modules', {'rawpy': self.raw_module(fail=True)}), patch.object(
                self.ai, 'load_image', wraps=self.ai.load_image) as loader:
            self.assertEqual(load_thumbnail('test.arw').size, (90, 60))
        loader.assert_called_once_with('test.arw')

    def test_heic_uses_oriented_standard_loader(self):
        self.ai.load_image = lambda path: np.zeros((90, 60, 3), np.uint8)
        self.assertEqual(load_thumbnail('test.heic').size, (60, 90))
