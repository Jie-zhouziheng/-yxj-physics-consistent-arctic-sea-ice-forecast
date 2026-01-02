from typing import List, Tuple
from .sic_dataset import SICIndexItem


def split_index_by_year(
    index: List[SICIndexItem],
    split: str,
) -> List[SICIndexItem]:
    """
    Split SICIndexItem list into train / val / test by year.

    Fixed protocol:
      Train : 1979–2010
      Val   : 2011–2016
      Test  : 2017–2022
    """
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")

    def year_of(item: SICIndexItem) -> int:
        return int(item.ym[:4])

    if split == "train":
        y0, y1 = 1979, 2010
    elif split == "val":
        y0, y1 = 2011, 2016
    else:  # test
        y0, y1 = 2017, 2022

    out = [
        item for item in index
        if y0 <= year_of(item) <= y1
    ]

    if len(out) == 0:
        raise RuntimeError(f"No samples found for split={split}")

    return out
