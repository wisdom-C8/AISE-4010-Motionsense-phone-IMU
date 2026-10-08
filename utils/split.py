"""Split the MotionSense dataset by subject into train/valid/test folders.

Subjects are sorted and assigned in order: the first `train_ratio` of them go to train,
the next `valid_ratio` to valid, and the rest to test, so no person appears in two splits.

Input layout:  <dataset>/<motion>/sub_<n>.csv
Output layout: <dataset>_split/<split>/<motion>/sub_<n>.csv   (next to the input; raw data untouched)

Usage: python utils/split.py A_DeviceMotion_data 0.7 0.15 0.15 [--overwrite]
"""
import argparse
import math
import shutil
from pathlib import Path

SPLITS = ("train", "valid", "test")


def split_dataset(dataset_path, train_ratio, valid_ratio, test_ratio, overwrite=False):
    """Copy each subject's CSVs into one split. Returns {"train": [subjects], "valid": [...], "test": [...]}."""
    ratios = dict(zip(SPLITS, (train_ratio, valid_ratio, test_ratio)))
    if any(r < 0 for r in ratios.values()) or not math.isclose(sum(ratios.values()), 1, abs_tol=1e-6):
        raise ValueError(f"Ratios must be >= 0 and sum to 1, got {ratios}")

    dataset = Path(dataset_path)
    subjects = sorted({int(f.stem.split("_")[1]) for f in dataset.glob("*/sub_*.csv")})
    if not subjects:
        raise FileNotFoundError(f"No <motion>/sub_*.csv files found in {dataset}")

    n_train = round(len(subjects) * train_ratio)
    n_valid = round(len(subjects) * valid_ratio)
    assignment = {
        "train": subjects[:n_train],
        "valid": subjects[n_train:n_train + n_valid],
        "test": subjects[n_train + n_valid:],
    }
    if empty := [s for s in SPLITS if ratios[s] > 0 and not assignment[s]]:
        raise ValueError(f"{empty} would get no subjects with {len(subjects)} subjects and ratios {ratios}")

    out = dataset.with_name(dataset.name + "_split")
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} already exists; pass overwrite=True (--overwrite) to replace it")
        shutil.rmtree(out)  # don't mix in files from a previous split

    motions = sorted(d for d in dataset.iterdir() if d.is_dir())
    for split, split_subjects in assignment.items():
        n_files = 0
        for motion in motions:
            for s in split_subjects:
                src = motion / f"sub_{s}.csv"
                if src.exists():
                    dst = out / split / motion.name / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    n_files += 1
        print(f"{split:5}: {len(split_subjects):2} subjects {split_subjects}, {n_files} files")
    print(f"Written to {out}")
    return assignment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a MotionSense-style dataset by subject.")
    parser.add_argument("dataset_path")
    parser.add_argument("train_ratio", type=float)
    parser.add_argument("valid_ratio", type=float)
    parser.add_argument("test_ratio", type=float)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing <dataset>_split folder")
    args = parser.parse_args()
    split_dataset(args.dataset_path, args.train_ratio, args.valid_ratio, args.test_ratio, args.overwrite)
