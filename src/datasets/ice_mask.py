import numpy as np


def build_static_ice_mask(
    ds_train,
    threshold: float = 0.15,
):
    """
    Build a static effective ice mask from training dataset.

    A grid cell is considered valid if it exhibits
    SIC >= threshold at least once during the training period.

    Parameters
    ----------
    ds_train : SICWindowDataset
        Dataset constructed from TRAIN split.
    threshold : float
        SIC threshold defining ice presence.

    Returns
    -------
    mask : np.ndarray, shape (H, W), dtype=bool
        True for effective ice region.
    """
    max_sic = None

    for i in range(len(ds_train)):
        _, y, _ = ds_train[i]  # y: (H, W)

        if max_sic is None:
            max_sic = y.copy()
        else:
            max_sic = np.maximum(max_sic, y)

    mask = max_sic >= threshold
    return mask
