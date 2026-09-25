import argparse
from pathlib import Path

from batch_pipeline import analyze_batch, scan_folder


def show_progress(done, total, record):
    status = "成功" if record["ok"] else "失敗"
    print(
        f"[{done}/{total}] {status} "
        f"{Path(record['path']).name}"
    )

    if not record["ok"]:
        print(f"  原因：{record['error']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="照片資料夾")
    parser.add_argument("--weight", type=float, default=0.6)
    parser.add_argument("--output", default="batch_results.jsonl")
    args = parser.parse_args()

    paths, skipped = scan_folder(args.folder)

    if skipped:
        detail = "、".join(f"{ext} {n} 張" for ext, n in sorted(skipped.items()))
        print(f"[WARN] 略過不支援的照片格式：{detail}。"
              f"｜請先轉成 JPG（例如 iPhone 可在「設定 > 相機 > 格式」改存「最相容」）")

    if not paths:
        print("資料夾內沒有支援的照片。")
        return

    summary = analyze_batch(
        paths,
        output_path=args.output,
        aesthetic_weight=args.weight,
        on_progress=show_progress,
    )
    print(summary)


if __name__ == "__main__":
    main()
