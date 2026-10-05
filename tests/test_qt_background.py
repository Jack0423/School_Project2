import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest


class QtBackgroundTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('PyQt6'), 'PyQt6 required')
    def test_real_qt_in_isolated_process(self):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('qt_background_check.py'))],
                                capture_output=True, text=True, timeout=30,
                                env={**os.environ, 'QT_QPA_PLATFORM':'offscreen'})
        self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)


if __name__ == '__main__':
    unittest.main()
