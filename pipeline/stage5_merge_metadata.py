"""
Stage 5: Merge metadata and generate training CSVs.
Reads render metadata from pipeline/data/renders/*/metadata.json
Outputs:
  pipeline/data/image_metadata.json      — unified metadata
  pipeline/data/pairs_train.csv          — 80% training pairs
  pipeline/data/pairs_val.csv            — 10% validation pairs
  pipeline/data/pairs_test.csv           — 10% test pairs

CSV format (compatible with existing ARIS pipeline):
  source_image, target_image, azimuth_delta, elevation_delta, source_azimuth, target_azimuth
"""

import csv
import json
import os
import random


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RENDERS_DIR = os.path.join(DATA_DIR, "renders")
PROMPTS_PATH = os.path.join(DATA_DIR, "prompts.json")
METADATA_OUT = os.path.join(DATA_DIR, "image_metadata.json")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# Test = remaining 10%

RANDOM_SEED = 42


def load_object_metadata(obj_id: str):
    """Load per-object metadata from renders/{obj_id}/metadata.json."""
    meta_path = os.path.join(RENDERS_DIR, obj_id, "metadata.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r") as f:
        return json.load(f)


def generate_pairs(obj_meta: dict):
    """
    Generate (source, target) pairs for all azimuth pairs within the same elevation.
    Also generate cross-elevation pairs for same azimuth.

    Returns list of pair dicts with:
      source_image, target_image, azimuth_delta, elevation_delta,
      source_azimuth, target_azimuth, source_elevation, target_elevation, obj_id
    """
    obj_id = obj_meta["obj_id"]
    frames = obj_meta["frames"]

    # Index frames by (azimuth, elevation)
    frame_index = {}
    for f in frames:
        key = (f["azimuth"], f["elevation"])
        frame_index[key] = f

    pairs = []
    azimuths = obj_meta["azimuths"]
    elevations = obj_meta["elevations"]

    # For each elevation level: generate all azimuth pairs (source → target)
    for el in elevations:
        for src_az in azimuths:
            for tgt_az in azimuths:
                if src_az == tgt_az:
                    continue
                src_key = (src_az, el)
                tgt_key = (tgt_az, el)
                if src_key not in frame_index or tgt_key not in frame_index:
                    continue

                az_delta = (tgt_az - src_az + 360) % 360
                # Use signed delta in [-180, 180]
                if az_delta > 180:
                    az_delta -= 360

                src_frame = frame_index[src_key]
                tgt_frame = frame_index[tgt_key]

                pairs.append({
                    "obj_id": obj_id,
                    "source_image": src_frame["path"],
                    "target_image": tgt_frame["path"],
                    "azimuth_delta": az_delta,
                    "elevation_delta": 0,
                    "source_azimuth": src_az,
                    "target_azimuth": tgt_az,
                    "source_elevation": el,
                    "target_elevation": el,
                })

    # Cross-elevation pairs at same azimuth
    for az in azimuths:
        for src_el in elevations:
            for tgt_el in elevations:
                if src_el == tgt_el:
                    continue
                src_key = (az, src_el)
                tgt_key = (az, tgt_el)
                if src_key not in frame_index or tgt_key not in frame_index:
                    continue

                src_frame = frame_index[src_key]
                tgt_frame = frame_index[tgt_key]

                pairs.append({
                    "obj_id": obj_id,
                    "source_image": src_frame["path"],
                    "target_image": tgt_frame["path"],
                    "azimuth_delta": 0,
                    "elevation_delta": tgt_el - src_el,
                    "source_azimuth": az,
                    "target_azimuth": az,
                    "source_elevation": src_el,
                    "target_elevation": tgt_el,
                })

    return pairs


def split_pairs(all_pairs: list, seed: int = RANDOM_SEED):
    """Split pairs into train/val/test by object ID (object-level split)."""
    # Group by obj_id to ensure object-level split
    obj_ids = list(set(p["obj_id"] for p in all_pairs))
    random.seed(seed)
    random.shuffle(obj_ids)

    n = len(obj_ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_ids = set(obj_ids[:n_train])
    val_ids = set(obj_ids[n_train:n_train + n_val])
    test_ids = set(obj_ids[n_train + n_val:])

    train = [p for p in all_pairs if p["obj_id"] in train_ids]
    val = [p for p in all_pairs if p["obj_id"] in val_ids]
    test = [p for p in all_pairs if p["obj_id"] in test_ids]

    return train, val, test, train_ids, val_ids, test_ids


def write_csv(pairs: list, path: str):
    """Write pairs to CSV file."""
    if not pairs:
        print(f"  [WARN] No pairs to write: {path}")
        return

    fieldnames = [
        "source_image", "target_image",
        "azimuth_delta", "elevation_delta",
        "source_azimuth", "target_azimuth",
        "source_elevation", "target_elevation",
        "obj_id",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            writer.writerow({k: pair[k] for k in fieldnames})

    print(f"  Wrote {len(pairs)} pairs → {path}")


def main():
    # Load prompts to get object IDs
    with open(PROMPTS_PATH, "r") as f:
        prompts = json.load(f)
    obj_ids = [p["id"] for p in prompts]

    # Load all object metadata
    all_obj_meta = {}
    for obj_id in obj_ids:
        meta = load_object_metadata(obj_id)
        if meta:
            all_obj_meta[obj_id] = meta
        else:
            print(f"[WARN] No metadata for {obj_id}, skipping")

    print(f"[Stage 5] Loaded metadata for {len(all_obj_meta)}/{len(obj_ids)} objects")

    # Generate all pairs
    all_pairs = []
    for obj_id, meta in all_obj_meta.items():
        pairs = generate_pairs(meta)
        all_pairs.extend(pairs)
        print(f"  {obj_id}: {len(pairs)} pairs")

    print(f"[Stage 5] Total pairs: {len(all_pairs)}")

    # Split train/val/test
    train, val, test, train_ids, val_ids, test_ids = split_pairs(all_pairs)
    print(f"[Stage 5] Split: train={len(train)} ({len(train_ids)} objs), "
          f"val={len(val)} ({len(val_ids)} objs), test={len(test)} ({len(test_ids)} objs)")

    # Write CSVs
    write_csv(train, os.path.join(DATA_DIR, "pairs_train.csv"))
    write_csv(val, os.path.join(DATA_DIR, "pairs_val.csv"))
    write_csv(test, os.path.join(DATA_DIR, "pairs_test.csv"))

    # Write unified metadata
    unified_meta = {
        "version": "1.0",
        "total_objects": len(all_obj_meta),
        "total_renders": sum(m["total_renders"] for m in all_obj_meta.values()),
        "total_pairs": len(all_pairs),
        "split": {
            "train": {"pairs": len(train), "objects": sorted(train_ids)},
            "val": {"pairs": len(val), "objects": sorted(val_ids)},
            "test": {"pairs": len(test), "objects": sorted(test_ids)},
        },
        "objects": all_obj_meta,
    }
    with open(METADATA_OUT, "w") as f:
        json.dump(unified_meta, f, indent=2)
    print(f"[Stage 5] Metadata → {METADATA_OUT}")

    # Print summary
    print("\n[Stage 5] Summary:")
    print(f"  Objects:       {len(all_obj_meta)}")
    print(f"  Total renders: {unified_meta['total_renders']}")
    print(f"  Total pairs:   {len(all_pairs)}")
    print(f"  Train pairs:   {len(train)}")
    print(f"  Val pairs:     {len(val)}")
    print(f"  Test pairs:    {len(test)}")
    print("\n[Stage 5] ✓ Done")


if __name__ == "__main__":
    main()
