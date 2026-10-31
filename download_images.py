# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=2.19.0",
#   "pillow>=10.0.0",
#   "tqdm>=4.66.0",
# ]
# ///
"""YANS-official/ogiri-bokete のお題画像を全てダウンロードする。

お題に画像があるのは type が image_to_text / image_text_to_text の行のみで、
image 列に画像本体 (PIL.Image, JPEG) がデータセットに埋め込まれている。
ファイル名は odai_id (お題ごとの一意なID) を使う。
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

DATASET_ID = "YANS-official/ogiri-bokete"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download odai images from YANS-official/ogiri-bokete")
    parser.add_argument("--out", type=Path, default=Path("images"), help="保存先ディレクトリ (default: ./images)")
    parser.add_argument("--overwrite", action="store_true", help="既存ファイルも上書き保存する")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("データセットを読み込み中...")
    ds = load_dataset(DATASET_ID, split="train")

    rows = [row for row in ds if row["image"] is not None]
    print(f"お題画像 {len(rows)} 件を検出しました。")

    saved = 0
    skipped = 0
    for row in tqdm(rows, desc="saving"):
        dest = args.out / f"{row['odai_id']}.jpg"
        if dest.exists() and not args.overwrite:
            skipped += 1
            continue
        image = row["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(dest, "JPEG")
        saved += 1

    print(f"\n完了: 保存 {saved} 件, スキップ {skipped} 件")
    print(f"保存先: {args.out.resolve()}")


if __name__ == "__main__":
    main()
