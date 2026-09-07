"""
清除資料庫中以「舊模型尺度」算出的分析結果，讓系統把那些照片重新分析一次。

為什麼需要這支腳本
------------------
本學期美感模型由二元分類器換成評分分佈模型，兩者的輸出本質不同：

    舊模型  二元分類機率 x 100，沒有上下界（實測 -32.8 ~ 154.9），
            必須裁切才能壓進 0~100，因此大量照片正好卡在 0 或 100。
    新模型  10 級評分分佈取期望值（1~10 分）再換算，
            實際輸出範圍約 21 ~ 74，永遠不可能出現 100 分。

同一張照片在兩個模型下的美感分實測相差 8 ~ 60 分。
「優秀」門檻也同步由 85 改為 62。

於是資料庫裡上學期存下的分數，與現在新分析出來的分數不是同一把尺，
但它們被放在同一個清單裡排序、比大小：

  1. ★最佳照片會選錯。舊資料的 100 分永遠贏過新模型的上限 74 分，
     任何沒有重新分析的舊照片都會霸佔第一名。
  2. 「優秀」標記混著兩套門檻（85 與 62）。
  3. 依綜合分產生的 .xmp 評等同樣停留在舊尺度。

這個問題不會拋出任何錯誤，只會安靜地把最佳照片選錯——正是展示時最容易被看出來的功能。

這支腳本做什麼
--------------
把分析結果相關的欄位清空、並把「是否已分析」旗標歸零，
讓系統依照既有的「未分析」流程重新跑一次。
照片的檔名、路徑與 metadata 完全不動——那些與模型無關，重掃很浪費時間。

安全設計
--------
  * 預設為試算（dry run），只印出「打算做什麼」，不寫入任何東西。
    真正要執行必須明確加上 --apply。
  * --apply 會先把整個資料庫檔複製一份備份，才開始修改。
  * 欄位不是猜死的：先讀 PRAGMA table_info 取得實際 schema，
    再以關鍵字比對，並把「比對到的」與「沒比對到的」全部印出來供人工核對。
    比對不到任何分數欄位時直接中止，不會憑空亂改。
  * 可用 --columns 明確指定要清空的欄位，完全略過自動比對。

用法
----
    python reset_db_analysis.py photos.db                  # 試算，不寫入
    python reset_db_analysis.py photos.db --apply          # 實際執行（會先備份）
    python reset_db_analysis.py photos.db --table photos
    python reset_db_analysis.py photos.db --columns aesthetic_score,technical_score --apply

長期建議
--------
本腳本是一次性的補救。要根治應在資料表加一個 model_version 欄位，
記錄每筆分數是哪一版模型算的，日後換模型時只需重跑版本不符的照片，
而不是像現在這樣只能整批清空。
"""
import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# 要清空的分析欄位——以關鍵字比對實際 schema，不假設確切命名。
# 中英文都列，因為前台的欄位命名慣例不一定與模型端相同。
SCORE_KEYWORDS = (
    'aesthetic', 'aes_', 'aes-', '美感',
    'technical', 'tech_', 'tech-', '技術',
    'overall', 'total', 'final', 'combined', '綜合',
    'status', '狀態',
    'suggest', 'advice', 'recommend', '建議',
    'judge', 'judgement', 'judgment', 'verdict', '判斷',
    'issue', '問題',
)

# 布林旗標欄位：設為 0 而不是 NULL。
#
# 「是否已分析」設 0，系統才會把照片視為待分析而重跑。
# 「是否為最佳照片」也歸在這裡而非上面的分數欄位——把布林欄位設成 NULL 很危險：
# Python 端 `if row['is_best']` 判斷 None 沒問題，但 SQL 的
# `WHERE is_best = 0` 不會匹配到 NULL，兩種讀法會給出不同答案。
FLAG_KEYWORDS = ('analyzed', 'analysed', 'has_analyzed', '已分析', '完成分析',
                 'best', '最佳')

# 這些一定不能動——與模型無關，清掉只會逼系統重掃硬碟。
PROTECTED_KEYWORDS = ('file_name', 'filename', 'file_path', 'filepath',
                      'path', 'metadata', 'meta', '檔名', '路徑')

# 主鍵等短名稱要求完全相符。'id' 若當成子字串比對，
# 任何含有 id 兩個字母的欄位（例如 width、video_id）都會被誤判為受保護。
PROTECTED_EXACT = ('id', 'rowid', 'pk')


def _matches(column, keywords):
    low = column.lower()
    return any(k.lower() in low for k in keywords)


def pick_columns(columns):
    """從實際 schema 挑出要清空的分數欄位與要歸零的旗標欄位。"""
    protected = [c for c in columns
                 if c.lower() in PROTECTED_EXACT or _matches(c, PROTECTED_KEYWORDS)]
    flags = [c for c in columns
             if _matches(c, FLAG_KEYWORDS) and c not in protected]
    scores = [c for c in columns
              if _matches(c, SCORE_KEYWORDS) and c not in protected and c not in flags]
    untouched = [c for c in columns
                 if c not in protected and c not in flags and c not in scores]
    return scores, flags, protected, untouched


def main():
    p = argparse.ArgumentParser(
        description='清除舊模型尺度的分析結果，讓系統重新分析',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('db', help='SQLite 資料庫檔路徑')
    p.add_argument('--table', default='photos', help='資料表名稱（預設 photos）')
    p.add_argument('--columns', help='明確指定要清空的欄位，以逗號分隔（略過自動比對）')
    p.add_argument('--apply', action='store_true',
                   help='真正執行寫入。未指定時只試算，不改動任何資料')
    args = p.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f'[FAIL] 找不到資料庫檔：{db_path}')
        return 1

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        if args.table not in tables:
            print(f'[FAIL] 資料表 {args.table!r} 不存在。這個資料庫裡有：{tables}')
            print('       請用 --table 指定正確的資料表名稱。')
            return 1

        columns = [r['name'] for r in con.execute(f'PRAGMA table_info({args.table})')]
        total = con.execute(f'SELECT COUNT(*) FROM {args.table}').fetchone()[0]

        if args.columns:
            scores = [c.strip() for c in args.columns.split(',') if c.strip()]
            unknown = [c for c in scores if c not in columns]
            if unknown:
                print(f'[FAIL] 這些欄位不存在於 {args.table}：{unknown}')
                print(f'       實際欄位：{columns}')
                return 1
            _, flags, protected, _ = pick_columns(columns)
            untouched = [c for c in columns if c not in scores and c not in flags]
        else:
            scores, flags, protected, untouched = pick_columns(columns)

        print(f'資料庫：{db_path}')
        print(f'資料表：{args.table}（{total} 筆）')
        print()
        print(f'  將清空為 NULL 的分析欄位 ({len(scores)}) : {scores or "（無）"}')
        print(f'  將設為 0 的旗標欄位 ({len(flags)})         : {flags or "（無）"}')
        print(f'  受保護、不會動的欄位 ({len(protected)})     : {protected or "（無）"}')
        print(f'  未歸類、不會動的欄位 ({len(untouched)})     : {untouched or "（無）"}')
        print()
        print('  ⚠ 請人工核對上面四行。自動比對是靠欄位名稱的關鍵字，')
        print('    命名習慣不同就可能漏抓或誤抓。有疑慮請改用 --columns 明確指定。')
        print()

        if not scores:
            print('[FAIL] 沒有比對到任何分析欄位，不做任何事。')
            print(f'       實際欄位：{columns}')
            print('       請用 --columns 明確指定要清空哪些欄位。')
            return 1

        if not flags:
            print('[WARN] 沒有比對到「是否已分析」旗標欄位。')
            print('       分數會被清空，但系統可能仍把這些照片視為已分析而不重跑。')
            print('       若確實有這個欄位，請確認命名後以 --columns 一併處理。')
            print()

        if not args.apply:
            print('[INFO] 這是試算，沒有任何資料被修改。')
            print('       確認上面的欄位清單無誤後，加上 --apply 再執行一次。')
            return 0

        backup = db_path.with_name(
            f'{db_path.stem}.backup-{datetime.now():%Y%m%d-%H%M%S}{db_path.suffix}')
        shutil.copy2(db_path, backup)
        print(f'[ OK ] 已備份原始資料庫：{backup.name}')

        sets = [f'{c} = NULL' for c in scores] + [f'{c} = 0' for c in flags]
        sql = f'UPDATE {args.table} SET ' + ', '.join(sets)
        cur = con.execute(sql)
        con.commit()
        print(f'[ OK ] 已清除 {cur.rowcount} 筆照片的分析結果。')
        print('       下次在系統中開啟這個資料夾時，這些照片會被視為「尚未分析」而重新評分。')
        return 0
    finally:
        con.close()


if __name__ == '__main__':
    sys.exit(main())
