"""
主控台輸出編碼的回歸測試。

守的是這個缺陷：Windows 繁體中文環境下，當程式的輸出被重導向到檔案或管線時
（`python train_nima.py > log.txt`，或被別的程式以 subprocess 呼叫），
Python 不會走 UTF-8，而是退回地區設定的 cp950。此時 `print()` 裡只要有一個
emoji 就會拋 UnicodeEncodeError：

    UnicodeEncodeError: 'cp950' codec can't encode character U+26D4

實測 `python train_nima.py --save nima_aes_dist.pth > log.txt` 會炸在
「權重檔已存在」那句保護訊息上——**一句純裝飾的字元把保護機制本身弄成了崩潰**，
使用者看到的是 traceback 而不是「請加 --force」。

在互動主控台直接執行時不會發生（Windows console 走 UTF-8），
所以這個問題很容易漏掉，也因此值得用測試守住。

專案的慣例是一律用 [ OK ] / [WARN] / [FAIL] / [INFO] 純文字標籤，
中文本身在 cp950 沒問題，只有 emoji 與少數數學符號不行。
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests._util import PROJECT_ROOT

# arw_viewer_gui.py 是同組成員負責的 PyQt6 前台，不在本次修正範圍內。
# 它的 emoji 絕大多數是放進 Qt 標籤（不經過主控台編碼，完全沒問題），
# 只有兩處 print() 會受影響，已列入交接說明請前台負責人處理。
EXEMPT = {'arw_viewer_gui.py'}

SKIP_DIRS = {'_archive', '__pycache__', '.venv', 'venv', '.git'}


def _source_files():
    for path in sorted(PROJECT_ROOT.rglob('*.py')):
        if SKIP_DIRS & set(path.relative_to(PROJECT_ROOT).parts):
            continue
        if path.name in EXEMPT:
            continue
        yield path


class TestSourcesAreCp950Encodable(unittest.TestCase):
    """
    靜態檢查：原始碼裡不該出現 cp950 編不出來的字元。

    刻意檢查整個檔案而不是只檢查 print() 的參數——判斷一個字串最後會不會被
    印出來需要做資料流分析，而且實際踩過的兩個案例都不是字面上的 print()：
      * ai_inference.py 的 U+2265 是塞進 technical_issues 清單，由 score.py 印出。
      * docstring 會經由 argparse 的 epilog 與 help() 印出。
    整個檔案一律要求可編碼最單純，也不會冤枉任何合法寫法——
    中文、全形標點、製表框線字元在 cp950 都是合法的。
    """

    def test_no_unencodable_characters(self):
        offenders = {}
        for path in _source_files():
            bad = {}
            for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                for ch in line:
                    try:
                        ch.encode('cp950')
                    except UnicodeEncodeError:
                        bad.setdefault(ch, []).append(lineno)
            if bad:
                offenders[path.relative_to(PROJECT_ROOT).as_posix()] = bad

        if offenders:
            report = []
            for name, bad in offenders.items():
                for ch, lines in bad.items():
                    report.append(f'  {name}:{lines[0]}  U+{ord(ch):04X} {ch!r}'
                                  f'（共 {len(lines)} 處）')
            self.fail(
                '以下字元在 cp950 編不出來，輸出被重導向時會拋 UnicodeEncodeError。\n'
                '請改用 [ OK ] / [WARN] / [FAIL] / [INFO] 純文字標籤：\n'
                + '\n'.join(report)
            )


class TestGuardMessagesSurviveCp950(unittest.TestCase):
    """
    動態檢查：實際用 cp950 跑一次保護訊息的路徑。

    靜態檢查已經涵蓋得更廣，但這一項證明的是「真的跑起來不會炸」，
    而不只是「字元集合看起來沒問題」。
    """

    def _run_under_cp950(self, script, cwd, args=()):
        env = dict(os.environ, PYTHONIOENCODING='cp950')
        return subprocess.run([sys.executable, str(script), *args], cwd=str(cwd),
                              env=env, capture_output=True, text=True,
                              encoding='cp950', errors='replace')

    def test_split_data_missing_source_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = Path(tmp)
            (sandbox / 'data').mkdir()
            script = PROJECT_ROOT / 'split_data.py'
            result = self._run_under_cp950(script, sandbox)

        self.assertNotIn('UnicodeEncodeError', result.stderr,
                         '保護訊息在 cp950 下崩潰了')
        self.assertEqual(result.returncode, 1)
        self.assertIn('[FAIL]', result.stdout)

    def test_train_scripts_help_text(self):
        """--help 會把 docstring 與參數說明全部印出來，是最長的一段輸出。"""
        for name in ('train_nima.py', 'train_tech.py', 'score.py',
                     'build_ava_labels.py', 'reset_db_analysis.py',
                     'compare_aesthetic_models.py', 'split_data.py',
                     'benchmark_gpu.py', 'model_report.py'):
            with self.subTest(script=name):
                result = self._run_under_cp950(PROJECT_ROOT / name, PROJECT_ROOT,
                                               args=('--help',))
                self.assertNotIn('UnicodeEncodeError', result.stderr,
                                 f'{name} --help 在 cp950 下崩潰了')
                self.assertEqual(result.returncode, 0, result.stderr[-400:])


if __name__ == '__main__':
    unittest.main()
