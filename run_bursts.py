import argparse
import json
from pathlib import Path

from burst_metadata import CAMERA_MATCH_CHOICES, build_burst_check
from photo_grouping import group_photos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("metadata")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--camera-match", choices=CAMERA_MATCH_CHOICES,
                        default="model",
                        help="相機識別的嚴格程度："
                             "serial 只認機身序號；"
                             "model 序號讀不到時退到廠牌＋型號（預設）；"
                             "ignore 不比對相機，只看拍攝時間")
    parser.add_argument("--output", default="burst_groups.json")
    args = parser.parse_args()

    can_pair, stats = build_burst_check(
        args.metadata,
        max_seconds=args.seconds,
        camera_match=args.camera_match,
    )
    print(f"相機識別（--camera-match {args.camera_match}）："
          f"以序號認定 {stats['serial']} 張、以型號認定 {stats['model']} 張、"
          f"不比對相機 {stats['ignored']} 張、無法識別 {stats['無法識別']} 張")
    if stats["無拍攝時間"]:
        print(f"[WARN] 有 {stats['無拍攝時間']} 張讀不到拍攝時間，這些照片不會被分進連拍組。")
    if stats["無法識別"]:
        print(f"[WARN] 有 {stats['無法識別']} 張無法識別相機，這些照片不會被分進連拍組。"
              f"｜可改用 --camera-match ignore 只依拍攝時間判斷")

    groups = group_photos(
        args.input,
        threshold=args.threshold,
        can_pair=can_pair,
    )

    bursts = [group for group in groups if group["count"] > 1]
    ungrouped = sum(group["count"] == 1 for group in groups)

    with open(args.output, "x", encoding="utf-8") as file:
        json.dump(
            {
                "mode": "burst_candidates",
                "threshold": args.threshold,
                "max_span_seconds": args.seconds,
                "ungrouped_count": ungrouped,
                "groups": bursts,
            },
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print(f"連拍候選組：{len(bursts)}")
    print(f"未分入連拍組的照片：{ungrouped}")

    for number, group in enumerate(bursts, start=1):
        print(f"\n第 {number} 組，共 {group['count']} 張")

        for photo in group["photos"]:
            mark = "建議保留" if photo["is_best"] else "待比較"
            print(
                f"  [{mark}] {Path(photo['path']).name}"
                f" | 分數 {photo['overall_score']:.2f}"
            )

    print(f"\n結果已保存：{args.output}")


if __name__ == "__main__":
    main()
