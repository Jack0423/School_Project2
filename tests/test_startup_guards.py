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


class TestSplitDataNeverWritesItsSource(unittest.TestCase):
    """
    B20：split_data.py 原本讀 data/train.csv 又「寫回同一個檔」，
    重複執行會每次再砍掉 20%、驗證集被覆蓋，全程沒有任何錯誤訊息
    （3920 -> 3136 -> 2508 ...）。

    2026-09-14 把來源與輸出分離（改成與 split_koniq.py 相同的結構）之後，
    這裡守的不變量比原本更強：

        舊：重複執行要被擋下      —— 只擋誤觸，設計本身仍然危險
        新：來源檔永遠不被寫入    —— 資料流失在結構上就不可能發生

    關鍵的那一項是 test_force_rerun_is_idempotent：舊設計下加了 --force
    重跑會從 80 筆變成 64 筆，新設計下必須每次都切出完全一樣的結果。
    """

    ROWS = 100

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sandbox = Path(self.tmp.name)
        shutil.copy(PROJECT_ROOT / 'split_data.py', self.sandbox)
        (self.sandbox / 'data').mkdir()
        rows = ['index,image,label'] + [f'{i},{1000 + i},{i % 2}' for i in range(self.ROWS)]
        self.source = self.sandbox / 'data' / 'train_full.csv'
        self.source.write_text('\n'.join(rows), encoding='utf-8')
        self.source_bytes = self.source.read_bytes()

    def tearDown(self):
        self.tmp.cleanup()

    def _rows(self, name):
        text = (self.sandbox / 'data' / name).read_text(encoding='utf-8')
        return len([l for l in text.splitlines() if l.strip()]) - 1

    def _out_bytes(self):
        return ((self.sandbox / 'data' / 'train.csv').read_bytes(),
                (self.sandbox / 'data' / 'val.csv').read_bytes())

    def test_first_run_splits_normally(self):
        result = _run(self.sandbox / 'split_data.py', self.sandbox)
        self.assertEqual(result.returncode, 0, result.stderr[-500:])
        self.assertEqual(self._rows('train.csv'), 80)
        self.assertEqual(self._rows('val.csv'), 20)

    def test_source_file_is_never_modified(self):
        """最重要的一項：來源檔跑完之後必須一個位元組都沒變。"""
        _run(self.sandbox / 'split_data.py', self.sandbox)
        _run(self.sandbox / 'split_data.py', self.sandbox, args=('--force',))
        self.assertEqual(self.source.read_bytes(), self.source_bytes,
                         '來源檔被寫入了——這正是原本會吃掉 20% 資料的那個 bug')

    def test_rerun_is_blocked_by_default(self):
        _run(self.sandbox / 'split_data.py', self.sandbox)
        before = (self._rows('train.csv'), self._rows('val.csv'))

        result = _run(self.sandbox / 'split_data.py', self.sandbox)

        self.assertEqual(result.returncode, 1, '重複執行未被擋下')
        self.assertEqual((self._rows('train.csv'), self._rows('val.csv')), before,
                         '重複執行被擋下了，但輸出仍被改動')
        self.assertIn('--force', result.stdout,
                      '應告訴使用者如何在確認後強制執行')

    def test_force_rerun_is_idempotent(self):
        """
        舊設計下這裡會是 80 -> 64（每次再砍 20%）。
        來源與輸出分離之後，同一個來源加同一個亂數種子必須切出完全相同的結果。
        """
        _run(self.sandbox / 'split_data.py', self.sandbox)
        first = self._out_bytes()

        result = _run(self.sandbox / 'split_data.py', self.sandbox, args=('--force',))

        self.assertEqual(result.returncode, 0, result.stderr[-500:])
        self.assertEqual(self._rows('train.csv'), 80, '重跑後資料變少了')
        self.assertEqual(self._out_bytes(), first,
                         '同一個來源重跑應產生完全相同的切分')

    def test_refuses_when_an_output_is_the_source_itself(self):
        """就算使用者自己把輸出指回來源，也必須擋下來。"""
        result = _run(self.sandbox / 'split_data.py', self.sandbox,
                      args=('--train-out', 'data/train_full.csv', '--force'))
        self.assertEqual(result.returncode, 1, '輸出指回來源竟然被允許')
        self.assertEqual(self.source.read_bytes(), self.source_bytes,
                         '被擋下了，但來源檔仍被改動')

    def test_missing_source_explains_how_to_build_it(self):
        self.source.unlink()
        result = _run(self.sandbox / 'split_data.py', self.sandbox)
        self.assertEqual(result.returncode, 1)
        self.assertIn('train_full.csv', result.stdout, '應指出缺少哪個來源檔')


if __name__ == '__main__':
    unittest.main()
