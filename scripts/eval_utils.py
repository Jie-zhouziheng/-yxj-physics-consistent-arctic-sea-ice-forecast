import csv
from pathlib import Path
import numpy as np
import torch


def _month_from_tout(meta_tout) -> int:
    s = str(meta_tout)
    return int(s[4:6])


@torch.no_grad()
def evaluate_monthly_append(
    model,
    loader,
    device,
    wmask,
    denom,
    out_csv: Path,
    model_name: str,
    split: str,
    input_window: int,
    lead_time: int,
    mask_name: str,
):
    """
    Append monthly MAE/RMSE to CSV.

    CSV columns:
      model, split, input_window, lead_time, mask,
      month, n_samples, mae, rmse

    month:
      0            -> overall
      1..12        -> calendar month
      spring_3_6   -> aggregated March–June
    """
    model.eval()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    maes_by_m = {m: [] for m in range(1, 13)}
    rmses_by_m = {m: [] for m in range(1, 13)}
    maes_all, rmses_all = [], []

    # ---------- evaluation loop ----------
    for x, y, meta in loader:
        assert x.shape[0] == 1, "evaluate_monthly_append() assumes batch_size=1"

        x = x.unsqueeze(2).float().to(device)
        y = y.unsqueeze(1).float().to(device)

        tout = meta["t_out"]  # list[str], len=1
        m = _month_from_tout(tout[0])

        # support Phase1 / Phase2 / Phase3
        try:
            pred = model(x, tout)
        except TypeError:
            pred = model(x)

        y_np = y.squeeze(0).squeeze(0).cpu().numpy()
        p_np = pred.squeeze(0).squeeze(0).cpu().numpy()

        mae = float((np.abs(p_np - y_np) * wmask).sum() / denom)
        rmse = float(np.sqrt(((p_np - y_np) ** 2 * wmask).sum() / denom))

        maes_all.append(mae)
        rmses_all.append(rmse)

        if 1 <= m <= 12:
            maes_by_m[m].append(mae)
            rmses_by_m[m].append(rmse)

    rows = []

    # ---------- overall ----------
    rows.append({
        "model": model_name,
        "split": split,
        "input_window": input_window,
        "lead_time": lead_time,
        "mask": mask_name,
        "month": 0,
        "n_samples": len(maes_all),
        "mae": f"{np.mean(maes_all):.6f}",
        "rmse": f"{np.mean(rmses_all):.6f}",
    })

    # ---------- per month ----------
    for m in range(1, 13):
        if len(maes_by_m[m]) > 0:
            mae_m = np.mean(maes_by_m[m])
            rmse_m = np.mean(rmses_by_m[m])
        else:
            mae_m = float("nan")
            rmse_m = float("nan")

        rows.append({
            "model": model_name,
            "split": split,
            "input_window": input_window,
            "lead_time": lead_time,
            "mask": mask_name,
            "month": m,
            "n_samples": len(maes_by_m[m]),
            "mae": f"{mae_m:.6f}",
            "rmse": f"{rmse_m:.6f}",
        })

    # ---------- spring (Mar–Jun) ----------
    spring_months = [3, 4, 5, 6]
    spring_maes, spring_rmses = [], []

    for m in spring_months:
        spring_maes.extend(maes_by_m[m])
        spring_rmses.extend(rmses_by_m[m])

    if len(spring_maes) > 0:
        spring_mae = np.mean(spring_maes)
        spring_rmse = np.mean(spring_rmses)
    else:
        spring_mae = float("nan")
        spring_rmse = float("nan")

    rows.append({
        "model": model_name,
        "split": split,
        "input_window": input_window,
        "lead_time": lead_time,
        "mask": mask_name,
        "month": "spring_3_6",
        "n_samples": len(spring_maes),
        "mae": f"{spring_mae:.6f}",
        "rmse": f"{spring_rmse:.6f}",
    })

    # ---------- append to CSV ----------
    fields = [
        "model", "split", "input_window", "lead_time",
        "mask", "month", "n_samples", "mae", "rmse"
    ]
    write_header = not out_csv.exists()

    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return rows
