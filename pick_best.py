"""
把整批照片當成「同一組連拍」，直接排名並挑出建議保留的那一張。

    python run_batch.py <一組連拍的資料夾> --output burst.jsonl
    python pick_best.py burst.jsonl

與 photo_grouping.py / run_bursts.py 的差別：本腳本不自己判斷哪些照片是一組。
分組由使用者決定——在相機或 Lightroom 裡挑好一組連拍、複製到同一個資料夾即可。

為什麼需要這個模式
------------------
自動分組有兩個前提：內容夠相似（預設門檻 0.85），以及（連拍模式）讀得到拍攝時間與相機資訊。
任何一項不成立就會分不出組——例如 Sony ILCE-7M4 的 JPG 完全沒有相機序號欄位，
相機識別因此改成三層判斷（見 burst_metadata.camera_identity）才不會整個資料夾 0 組。

使用者自己指定分組時，這些前提全部不需要，輸出也不會因為欄位缺漏而變成空的。
提案書「識別連拍中的微小差異並挑選最佳者」這項需求，本身就允許由使用者先框出一組。

相似度只拿來提醒
----------------
仍會計算每張與建議保留照片的相似度，但**不**用來排除任何照片，只在偏低時印出提醒，
協助發現「不小心把不同組的照片放進同一個資料夾」。
"""
import argparse
import json
from pathlib import Path

from photo_grouping import DEFAULT_THRESHOLD, load_photos, unit_features


def rank_photos(input_path):
    """回傳依綜合分由高到低排序的照片清單，每筆附上與最佳照片的相似度。"""
    photos = load_photos(input_path)

    if not photos:
        return []

    features = unit_features(photos)

    # 分數相同時以路徑決定順序，確保重跑結果一致。
    order = sorted(
        range(len(photos)),
        key=lambda i: (-photos[i]["overall_score"], photos[i]["path"]),
    )
    best_index = order[0]

    ranked = []
    for rank, index in enumerate(order, start=1):
        photo = photos[index]
        ranked.append({
            "rank": rank,
            "path": photo["path"],
            "aesthetic_score": photo["aesthetic_score"],
            "technical_score": photo["technical_score"],
            "overall_score": photo["overall_score"],
            "is_best": index == best_index,
            "similarity_to_best": float(
                min(1.0, max(-1.0, features[index] @ features[best_index]))
            ),
        })

    return ranked


def main():
    parser = argparse.ArgumentParser(
        description="把整批照片當成同一組連拍，挑出建議保留的那一張")
    parser.add_argument("input", help="run_batch.py 產生的 JSONL")
    # 與自動分組用同一個門檻：分組時「算同一組」的標準，也就是這裡「看起來不像同一組」的標準
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"相似度低於此值只會印出提醒，不會排除照片（預設 {DEFAULT_THRESHOLD}）")
    parser.add_argument("--output", default="best_pick.json")
    args = parser.parse_args()

    ranked = rank_photos(args.input)

    if not ranked:
        print("沒有成功分析的照片。")
        return

    # 避免覆寫既有結果。
    with open(args.output, "x", encoding="utf-8") as file:
        json.dump(
            {
                "mode": "manual_burst",
                "count": len(ranked),
                "best_path": ranked[0]["path"],
                "photos": ranked,
            },
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print(f"共 {len(ranked)} 張，依綜合分排名：\n")
    for photo in ranked:
        mark = "建議保留" if photo["is_best"] else "      "
        print(
            f"  {photo['rank']:>2}. [{mark}] {Path(photo['path']).name}"
            f" | 綜合 {photo['overall_score']:>6.2f}"
            f" | 美感 {photo['aesthetic_score']:>6.2f}"
            f" | 技術 {photo['technical_score']:>6.2f}"
            f" | 與最佳相似度 {photo['similarity_to_best']:.4f}"
        )

    odd = [p for p in ranked if p["similarity_to_best"] < args.threshold]
    if odd:
        print(f"\n[WARN] 有 {len(odd)} 張與建議保留的照片相似度低於 {args.threshold}，"
              f"請確認是否真的屬於同一組連拍：")
        for photo in odd:
            print(f"        {Path(photo['path']).name}"
                  f"（相似度 {photo['similarity_to_best']:.4f}）")

    print(f"\n結果已保存：{args.output}")


if __name__ == "__main__":
    main()
