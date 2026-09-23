import argparse
from pathlib import Path

from batch_pipeline import collect_photos, analyze_batch


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

    paths = collect_photos(args.folder)

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
