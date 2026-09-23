import argparse
import json
from pathlib import Path

import numpy as np


def load_photos(input_path):
    """
    讀取批次結果，回傳成功的那些紀錄。不同權重混在一起時直接中止。

    抽成函式是為了讓 pick_best.py 共用同一套檢查——
    「不同權重的綜合分數不能一起排名」這條規則只該有一份實作。
    """
    with open(input_path, encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    photos = [r for r in records if r.get("ok")]

    if not photos:
        return []

    # 不同權重的綜合分數不能直接拿來排名。
    weights = {
        (r["aesthetic_weight"], r["technical_weight"])
        for r in photos
    }
    if len(weights) != 1:
        raise ValueError("結果包含不同權重，請先統一重算綜合分數")

    return photos


def unit_features(photos):
    """取出 1280 維特徵並正規化。正規化後，向量內積就是餘弦相似度。"""
    features = np.asarray(
        [r["feature_vector"] for r in photos],
        dtype=np.float64,
    )

    if features.shape != (len(photos), 1280):
        raise ValueError("每張照片必須有 1280 維特徵")

    if not np.isfinite(features).all():
        raise ValueError("特徵包含無效數值")

    lengths = np.linalg.norm(features, axis=1, keepdims=True)

    if (lengths <= 0).any():
        raise ValueError("特徵不能是全零向量")

    return features / lengths


def group_photos(input_path, threshold=0.9, can_pair=None):
    if not 0 <= threshold <= 1:
        raise ValueError("相似度門檻必須介於 0 與 1")

    photos = load_photos(input_path)

    if not photos:
        return []

    features = unit_features(photos)

    # 先按分數排序，分數相同時依路徑決定順序。
    order = sorted(
        range(len(photos)),
        key=lambda i: (
            -photos[i]["overall_score"],
            photos[i]["path"],
        ),
    )

    groups = []

    for index in order:
        target_group = None

        for group in groups:
            # 連拍模式先檢查時間與相機，再計算相似度。
            if can_pair is not None:
                eligible = all(
                    can_pair(
                        photos[index]["path"],
                        photos[member]["path"],
                    )
                    for member in group
                )

                if not eligible:
                    continue

            similarities = features[group] @ features[index]

            # 必須和組內每一張都達到門檻，才能加入。
            if np.all(similarities >= threshold):
                target_group = group
                break

        if target_group is None:
            groups.append([index])
        else:
            target_group.append(index)

    results = []

    for group_id, members in enumerate(groups, start=1):
        # 照片依分數由高到低加入，因此第一張就是最高分。
        best_index = members[0]
        best_photo = photos[best_index]

        results.append({
            "group_id": group_id,
            "count": len(members),
            "best_path": best_photo["path"],
            "photos": [
                {
                    "path": photos[i]["path"],
                    "overall_score": photos[i]["overall_score"],
                    "is_best": i == best_index,
                    "similarity_to_best": float(np.clip(
                        features[i] @ features[best_index],
                        -1.0,
                        1.0,
                    )),
                }
                for i in members
            ],
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="批次分析的 JSONL 檔案")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--output", default="photo_groups.json")
    args = parser.parse_args()

    groups = group_photos(args.input, args.threshold)

    # 避免覆寫既有結果。
    with open(args.output, "x", encoding="utf-8") as file:
        json.dump(
            {
                "threshold": args.threshold,
                "groups": groups,
            },
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    similar_groups = [g for g in groups if g["count"] > 1]
    single_count = sum(g["count"] == 1 for g in groups)

    print(f"相似照片組：{len(similar_groups)}")
    print(f"未分到相似組的照片：{single_count}")

    for group in similar_groups:
        print(f"\n第 {group['group_id']} 組：{group['count']} 張")

        for photo in group["photos"]:
            mark = "建議保留" if photo["is_best"] else "待比較"
            print(
                f"  [{mark}] {Path(photo['path']).name}"
                f" | 分數 {photo['overall_score']:.2f}"
                f" | 與推薦照片相似度 "
                f"{photo['similarity_to_best']:.4f}"
            )

    print(f"\n結果已保存：{args.output}")


if __name__ == "__main__":
    main()
