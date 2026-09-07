"""
命令列評分工具 —— 把照片或整個資料夾丟進來，直接印出評分結果。

    python score.py 照片.jpg
    python score.py data/my_photos              # 整個資料夾
    python score.py a.jpg b.ARW --json out.json # 存成 JSON

存在的理由：ai_inference.evaluate_photo() 是給前台程式呼叫的函式介面，
要手動測一張照片得寫 python -c 一行式，很不方便。這支腳本只做「輸入輸出」，
所有評分邏輯仍然來自 ai_inference，不重複實作任何規則。

輸出一律用 [ OK ] / [FAIL] 純文字標籤，不使用 emoji ——
與 ai_inference.py、check_env.py 的既有慣例一致，
理由同樣是 Windows 繁中主控台預設 cp950 編碼會讓 emoji 拋 UnicodeEncodeError。
"""
import argparse
import json
import os
import sys

import ai_inference
from ai_inference import RAW_EXTENSIONS, evaluate_photo

# 能讀的副檔名 = RAW 全系列 + PIL 支援的常見格式
IMAGE_EXTENSIONS = RAW_EXTENSIONS | {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def collect_targets(paths):
    """把使用者給的路徑（檔案或資料夾）展開成實際要評分的檔案清單。"""
    targets = []
    for p in paths:
        if os.path.isdir(p):
            # 只掃這一層，不遞迴 —— 遞迴掃到 data/dataset 會是 7,000 張，
            # 使用者幾乎不會是想這樣，但等他發現時已經跑了很久。
            found = sorted(
                os.path.join(p, name)
                for name in os.listdir(p)
                if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
            )
            if not found:
                print(f"[WARN] 資料夾內沒有可讀的影像檔：{p}")
            targets.extend(found)
        elif os.path.isfile(p):
            targets.append(p)
        else:
            print(f"[FAIL] 找不到路徑：{p}")
    return targets


def print_result(path, r):
    """把單張結果印成人看得懂的區塊。"""
    print(f"\n{'=' * 60}")
    print(os.path.basename(path))
    print('=' * 60)
    print(f"  美感分  {r['aesthetic_score']:6.2f}   （權重 {r['aesthetic_weight']}）")
    print(f"  技術分  {r['technical_score']:6.2f}   （權重 {r['technical_weight']}）")
    print(f"  綜合分  {r['overall_score']:6.2f}")
    print(f"  狀態    {r['status']}")
    print(f"  建議    {r['suggestion']}")
    if r['technical_issues']:
        print("  影像量測附註：")
        for issue in r['technical_issues']:
            print(f"    - {issue}")


def main():
    p = argparse.ArgumentParser(
        description='對照片評分（美感 + 技術 + 綜合）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('paths', nargs='+', help='照片路徑或資料夾（可給多個）')
    p.add_argument('--aesthetic-weight', type=float, metavar='W',
                   help='綜合分中美感分的佔比，0~1。技術分自動補成 1-W。'
                        '不給則用預設 0.6 / 0.4')
    p.add_argument('--json', metavar='PATH', help='把完整結果另存為 JSON')
    p.add_argument('--sort', action='store_true',
                   help='全部評完後，依綜合分由高到低列出排名')
    args = p.parse_args()

    targets = collect_targets(args.paths)
    if not targets:
        print("\n[FAIL] 沒有任何可評分的檔案。")
        return 1

    print(f"\n共 {len(targets)} 張待評分。")

    results = {}
    failed = []
    for path in targets:
        r = evaluate_photo(path, aesthetic_weight=args.aesthetic_weight)
        if r is None:
            # evaluate_photo 失敗時回傳 None，原因放在模組層級的 LAST_ERROR。
            # 直接讀模組屬性而非 from ... import，因為那是會被改寫的全域變數。
            print(f"\n[FAIL] {os.path.basename(path)}：{ai_inference.LAST_ERROR}")
            failed.append(path)
            continue
        results[path] = r
        print_result(path, r)

    if args.sort and len(results) > 1:
        print(f"\n{'=' * 60}")
        print("依綜合分排名")
        print('=' * 60)
        ranked = sorted(results.items(), key=lambda kv: kv[1]['overall_score'], reverse=True)
        for i, (path, r) in enumerate(ranked, 1):
            print(f"  {i:2}. {r['overall_score']:6.2f}  "
                  f"(美 {r['aesthetic_score']:5.1f} / 技 {r['technical_score']:5.1f})  "
                  f"{os.path.basename(path)}")

    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[ OK ] 結果已寫入 {args.json}")

    print(f"\n完成：成功 {len(results)} 張、失敗 {len(failed)} 張。")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
