"""Run Qt grouping tests in isolation from native model libraries."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest

class GroupingUiTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('PyQt6'), 'PyQt6 required')
    def test_grouping_ui_in_isolated_process(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name('qt_grouping_check.py')), '-v'],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=30, env={**os.environ, 'QT_QPA_PLATFORM':'offscreen'})
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
