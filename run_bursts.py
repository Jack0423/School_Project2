import argparse
import json
from pathlib import Path

from burst_metadata import build_burst_check
from photo_grouping import group_photos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("metadata")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--output", default="burst_groups.json")
    args = parser.parse_args()

    can_pair = build_burst_check(
        args.metadata,
        max_seconds=args.seconds,
    )

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
