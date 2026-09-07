"""
啟動階段保護機制的測試（對應 A1 與 B20）。

這兩項都必須用子行程測試：
  A1  是「模組載入時」就該失敗，同一個行程內 ai_inference 已經匯入過，
      無法在原地重現「找不到權重」的情境。
  B20 測的是腳本的離開代碼與副作用，本來就該以腳本方式執行。
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests._util import PROJECT_ROOT, required_weight_files, weights_available


def _run(code_or_script, cwd, args=()):
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    if isinstance(code_or_script, Path):
        cmd = [sys.executable, str(code_or_script), *args]
    else:
        cmd = [sys.executable, '-c', code_or_script, *args]
    return subprocess.run(cmd, cwd=str(cwd), env=env,
                          capture_output=True, text=True, encoding='utf-8')


class TestMissingWeightsFailsLoudly(unittest.TestCase):
    """
    A1 的回歸測試，也是整份稽核裡最嚴重的一項。

    原本找不到權重時只印一行警告就繼續，模型維持隨機初始化狀態，
    照樣回傳一組看起來完全正常、還附帶專業建議的分數。
    實測曾產生：美感 -12.48、技術 13.69，全部是隨機權重的噪音。
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        sandbox = Path(self.tmp.name)
        # 只複製程式碼，刻意不複製任何 .pth
        shutil.copy(PROJECT_ROOT / 'ai_inference.py', sandbox)
        shutil.copytree(PROJECT_ROOT / 'common', sandbox / 'common')
        self.sandbox = sandbox

    def tearDown(self):
        self.tmp.cleanup()

    def test_import_raises_file_not_found(self):
        result = _run('import ai_inference', self.sandbox)
        self.assertNotEqual(result.returncode, 0,
                            '缺少權重檔時模組竟然成功載入了')
        self.assertIn('FileNotFoundError', result.stderr)

    def test_error_message_names_the_missing_file_and_location(self):
        """
        訊息必須指出「缺哪個檔」與「在哪裡找」。
        刻意不寫死檔名——美感權重會隨著模型改版更換
        （例如從 nima_best.pth 換成 nima_aes_dist.pth），
        寫死檔名會讓這個測試在每次換模型時假性失敗。
        """
        result = _run('import ai_inference', self.sandbox)
        self.assertRegex(result.stderr, r'\S+\.pth',
                         '錯誤訊息應指出缺少哪一個 .pth 權重檔')
        self.assertIn(str(self.sandbox), result.stderr,
                      '錯誤訊息應指出程式在哪個目錄找權重')

    def test_no_score_is_ever_produced_without_weights(self):
        """關鍵斷言：寧可整個失敗，也不可以產生任何分數。"""
        code = (
            'import sys\n'
            'try:\n'
            '    import ai_inference as ai\n'
            '    r = ai.evaluate_photo("whatever.jpg")\n'
            '    print("PRODUCED_SCORE", r)\n'
            'except FileNotFoundError:\n'
            '    print("REFUSED")\n'
        )
        result = _run(code, self.sandbox)
        self.assertIn('REFUSED', result.stdout)
        self.assertNotIn('PRODUCED_SCORE', result.stdout)


class TestWeightPathIsIndependentOfCwd(unittest.TestCase):
    """
    A1 的另一半：權重路徑以檔案位置為基準，不是當前工作目錄。
    原本用相對路徑，從其他目錄啟動就找不到權重（然後靜默給假分數）。
    """

    def test_loads_from_a_different_working_directory(self):
        if not weights_available():
            self.skipTest('缺少現役權重：' + ' / '.join(required_weight_files()))
        with tempfile.TemporaryDirectory() as elsewhere:
            code = (
                'import sys\n'
                f'sys.path.insert(0, {str(PROJECT_ROOT)!r})\n'
                'import ai_inference as ai\n'
                'print("BASE_DIR", ai.BASE_DIR)\n'
            )
            result = _run(code, elsewhere)
        self.assertEqual(result.returncode, 0, result.stderr[-800:])
        self.assertIn(str(PROJECT_ROOT), result.stdout)


class TestSplitDataRerunGuard(unittest.TestCase):
    """
    B20：split_data.py 讀取 data/train.csv 又覆寫同一個檔案，
    重複執行會每次少掉 20%，且驗證集被覆蓋、永久消失，全程沒有錯誤訊息。
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sandbox = Path(self.tmp.name)
        shutil.copy(PROJECT_ROOT / 'split_data.py', self.sandbox)
        (self.sandbox / 'data').mkdir()
        rows = ['index,image,label'] + [f'{i},{1000 + i},{i % 2}' for i in range(100)]
        (self.sandbox / 'data' / 'train.csv').write_text('\n'.join(rows), encoding='utf-8')

    def tearDown(self):
        self.tmp.cleanup()

    def _train_rows(self):
        text = (self.sandbox / 'data' / 'train.csv').read_text(encoding='utf-8')
        return len([l for l in text.splitlines() if l.strip()]) - 1

    def test_first_run_splits_normally(self):
        result = _run(self.sandbox / 'split_data.py', self.sandbox)
        self.assertEqual(result.returncode, 0, result.stderr[-500:])
        self.assertEqual(self._train_rows(), 80)

    def test_second_run_is_blocked(self):
        _run(self.sandbox / 'split_data.py', self.sandbox)
        before = self._train_rows()

        result = _run(self.sandbox / 'split_data.py', self.sandbox)

        self.assertEqual(result.returncode, 1, '重複執行未被擋下')
        self.assertEqual(self._train_rows(), before,
                         '重複執行被擋下了，但資料仍被改動')

    def test_blocked_message_states_the_consequence(self):
        _run(self.sandbox / 'split_data.py', self.sandbox)
        result = _run(self.sandbox / 'split_data.py', self.sandbox)
        self.assertIn('--force', result.stdout,
                      '應告訴使用者如何在確認後強制執行')
        self.assertIn('64', result.stdout,
                      '應具體說明再跑一次會剩下幾筆（80 -> 64）')

    def test_force_flag_still_works(self):
        _run(self.sandbox / 'split_data.py', self.sandbox)
        result = _run(self.sandbox / 'split_data.py', self.sandbox, args=('--force',))
        self.assertEqual(result.returncode, 0, result.stderr[-500:])
        self.assertEqual(self._train_rows(), 64)


if __name__ == '__main__':
    unittest.main()
