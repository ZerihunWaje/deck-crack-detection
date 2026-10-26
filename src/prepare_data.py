"""Create a stratified train/val/test split of the SDNET2018 bridge-deck subset.

Expected input layout (official SDNET2018):
    <raw>/D/CD/*.jpg   cracked deck
    <raw>/D/UD/*.jpg   uncracked deck

Output layout (Keras image_dataset_from_directory friendly):
    <out>/{train,val,test}/{cracked,uncracked}/*.jpg
"""
import argparse
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

CLASSES = {"CD": "cracked", "UD": "uncracked"}
IMG_EXT = {".jpg", ".jpeg", ".png"}


def collect(raw: Path, structure: str):
    paths, labels = [], []
    for code, name in CLASSES.items():
        folder = raw / structure / code
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)
        paths += files
        labels += [name] * len(files)
    return paths, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--structure", default="D", help="D=deck, W=wall, P=pavement")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    paths, labels = collect(raw, args.structure)

    # test split first, then val from the remainder (all stratified)
    x_tmp, x_test, y_tmp, y_test = train_test_split(
        paths, labels, test_size=args.test, stratify=labels, random_state=args.seed)
    val_rel = args.val / (1 - args.test)
    x_train, x_val, y_train, y_val = train_test_split(
        x_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=args.seed)

    if out.exists():
        shutil.rmtree(out)
    print(f"{'split':<6} {'cracked':>8} {'uncracked':>10} {'% cracked':>10}")
    for split, xs, ys in [("train", x_train, y_train), ("val", x_val, y_val), ("test", x_test, y_test)]:
        for p, y in zip(xs, ys):
            dst = out / split / y
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / p.name)
        c = ys.count("cracked")
        print(f"{split:<6} {c:>8} {len(ys) - c:>10} {100 * c / len(ys):>9.1f}%")


if __name__ == "__main__":
    main()
