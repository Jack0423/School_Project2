"""
一鍵啟動（start.py）：缺套件時裝對版本、問過才裝、權重或套件不齊時不開主程式。

守的是：
  * PyTorch 在 requirements.txt 是 CUDA 版（+cu128），一般 PyPI 沒有；
    有 NVIDIA 顯卡要從 PyTorch 的索引裝，沒有（包括所有 Mac）要裝同版本的 CPU 版，裝錯就整個失敗。
  * 版本只寫在 requirements.txt 一個地方。
  * 沒有回答（例如被別的程式呼叫）不能當成同意安裝。
"""
import sys
import unittest
from unittest import mock

import start
from tests._util import requires_weights


class TestInstallPlan(unittest.TestCase):

    def setUp(self):
        self.specs = start.requirement_specs()

    def test_versions_come_from_requirements(self):
        self.assertEqual(self.specs['torch'], 'torch==2.11.0+cu128')
        self.assertEqual(self.specs['opencv-python'], 'opencv-python==4.13.0.92')
        self.assertEqual(set(start.RUNTIME_PACKAGES) - set(self.specs), set(),
                         '主程式需要的套件都要在 requirements.txt 裡')

    def test_nvidia_installs_cuda_build_from_pytorch_index(self):
        torch_cmd, others = start.install_commands(['torch', 'torchvision', 'rawpy'], self.specs, nvidia=True)
        self.assertEqual(torch_cmd[1:], ['-m', 'pip', 'install', 'torch==2.11.0+cu128',
                                         'torchvision==0.26.0+cu128', '--index-url', start.CUDA_INDEX])
        self.assertEqual(others[-1], 'rawpy==0.27.1')
        self.assertNotIn('--index-url', others, '其他套件從一般 PyPI 裝')

    def test_without_nvidia_installs_same_version_cpu_build(self):
        (torch_cmd,) = start.install_commands(['torch'], self.specs, nvidia=False)
        self.assertEqual(torch_cmd[1:], ['-m', 'pip', 'install', 'torch==2.11.0'])

    def test_installs_into_the_running_python(self):
        (cmd,) = start.install_commands(['numpy'], self.specs, nvidia=False)
        self.assertEqual(cmd[0], sys.executable, '要裝進執行 start.py 的這個 Python，不是 PATH 上另一個')

    def test_no_answer_is_not_consent(self):
        with mock.patch('builtins.input', side_effect=EOFError):
            self.assertFalse(start.ask('要自動安裝嗎？'))
        with mock.patch('builtins.input', return_value=' Y '):
            self.assertTrue(start.ask('要自動安裝嗎？'))

    def test_declined_install_does_not_run_pip_or_open_program(self):
        with mock.patch.object(start, 'missing_packages', return_value=['rawpy']), \
                mock.patch('builtins.input', return_value='n'), \
                mock.patch.object(start.subprocess, 'run') as run, \
                mock.patch.object(start.subprocess, 'call') as call, \
                mock.patch('builtins.print'):
            self.assertEqual(start.main([]), 1)
        run.assert_not_called()
        call.assert_not_called()

    def test_missing_weights_stop_before_opening(self):
        with mock.patch.object(start, 'missing_packages', return_value=[]), \
                mock.patch.object(start, 'ROOT', start.ROOT / 'no-such-folder'), \
                mock.patch.object(start.subprocess, 'call') as call, \
                mock.patch('builtins.print'):
            self.assertEqual(start.main([]), 1)
        call.assert_not_called()

    @requires_weights
    def test_check_passes_on_a_complete_setup(self):
        with mock.patch('builtins.print'), mock.patch('builtins.input', side_effect=EOFError):
            self.assertEqual(start.main(['--check']), 0)


if __name__ == '__main__':
    unittest.main()
