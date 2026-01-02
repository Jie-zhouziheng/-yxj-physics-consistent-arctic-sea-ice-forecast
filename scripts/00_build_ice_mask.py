from pathlib import Path
import numpy as np

from src.datasets import (
    build_index,
    split_index_by_year,
    SICWindowDataset,
    build_static_ice_mask,
)


def main():
    data_dir = Path("data/raw/nsidc_sic")
    out_dir = Path("data/eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold = 0.15
    mask_name = f"ice_mask_ice{int(threshold*100):02d}.npy"
    out_path = out_dir / mask_name

    # -------- Train Dataset --------
    index_all = build_index(data_dir, hemisphere="N")
    index_train = split_index_by_year(index_all, "train")

    ds_train = SICWindowDataset(
        index_train,
        input_window=12,
        lead_time=1,
    )

    print(f"[INFO] Train samples: {len(ds_train)}")

    # -------- Build mask --------
    ice_mask = build_static_ice_mask(ds_train, threshold=threshold)

    print(
        f"[INFO] Ice mask threshold={threshold} keeps "
        f"{int(ice_mask.sum())} / {ice_mask.size} grid cells"
    )

    # -------- Save --------
    np.save(out_path, ice_mask)
    print(f"[OK] Ice mask saved to: {out_path}")


if __name__ == "__main__":
    main()
