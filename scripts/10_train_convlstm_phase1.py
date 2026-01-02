# scripts/10_train_convlstm_phase1.py

from pathlib import Path
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import build_index, split_index_by_year, SICWindowDataset
from src.models.convlstm_baseline import ConvLSTMBaseline


# =========================
# Metrics (masked)
# =========================

def masked_mae_rmse(y_true: np.ndarray, y_pred: np.ndarray, wmask: np.ndarray, denom: float):
    diff = y_pred - y_true
    mae = float((np.abs(diff) * wmask).sum() / denom)
    rmse = float(np.sqrt(((diff * diff) * wmask).sum() / denom))
    return mae, rmse


def masked_mse_torch(pred: torch.Tensor, target: torch.Tensor, wmask: torch.Tensor, denom: float):
    """
    pred/target: (B, 1, H, W)
    wmask:       (1, 1, H, W) float32, 0/1
    denom: float = sum(mask) * B   (IMPORTANT: include batch)
    """
    diff = pred - target
    loss = (diff * diff) * wmask
    return loss.sum() / denom


def append_result_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "split", "input_window", "lead_time", "mask", "mae", "rmse"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)


# =========================
# Evaluation
# =========================

@torch.no_grad()
def evaluate(model, loader, device, wmask, denom):
    model.eval()
    maes, rmses = [], []

    for x, y, _ in loader:
        # keep current assumption explicit
        assert x.shape[0] == 1, "evaluate() assumes batch_size=1"

        # x: (B, T, H, W) -> (B, T, 1, H, W)
        x = x.unsqueeze(2).float().to(device)
        y = y.unsqueeze(1).float().to(device)

        # output already in [0,1] if model uses sigmoid head
        pred = model(x)

        y_np = y.squeeze(0).squeeze(0).cpu().numpy()
        p_np = pred.squeeze(0).squeeze(0).cpu().numpy()

        mae, rmse = masked_mae_rmse(y_np, p_np, wmask, denom)
        maes.append(mae)
        rmses.append(rmse)

    return float(np.mean(maes)), float(np.mean(rmses))


# =========================
# Main
# =========================

def main():
    # ---------- basic setup ----------
    data_dir = Path("data/raw/nsidc_sic")
    hemisphere = "N"
    input_window = 12
    lead_time = 1

    # training params
    epochs = 5
    batch_size = 2      # 如果显存不够，改成 1
    lr = 1e-3
    hidden_channels = 16

    # ---------- mask ----------
    mask_name = "ice15"
    mask_path = Path("data/eval/ice_mask_ice15.npy")
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask not found: {mask_path}\n"
            f"Please run: PYTHONPATH=. python scripts/00_build_ice_mask.py"
        )
    wmask = np.load(mask_path).astype(np.float32)
    denom = float(wmask.sum())
    if denom == 0:
        raise RuntimeError("Mask has zero valid cells.")
    print(f"[INFO] Loaded ice mask: keeps {int(denom)} / {wmask.size}")

    # ---------- device ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    # torch mask (broadcastable)
    wmask_t = torch.from_numpy(wmask).to(device).view(1, 1, *wmask.shape)

    # ---------- datasets ----------
    index_all = build_index(data_dir, hemisphere=hemisphere)
    index_train = split_index_by_year(index_all, "train")
    index_val = split_index_by_year(index_all, "val")
    index_test = split_index_by_year(index_all, "test")

    ds_train = SICWindowDataset(index_train, input_window=input_window, lead_time=lead_time)
    ds_val = SICWindowDataset(index_val, input_window=input_window, lead_time=lead_time)
    ds_test = SICWindowDataset(index_test, input_window=input_window, lead_time=lead_time)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0)

    print(f"[INFO] Train: {len(ds_train)} | Val: {len(ds_val)} | Test: {len(ds_test)}")

    # ---------- model ----------
    model = ConvLSTMBaseline(
        in_channels=1,
        hidden_channels=hidden_channels,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = None
    best_path = Path("experiments/convlstm_phase1_best.pt")
    best_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- training loop ----------
    for ep in range(1, epochs + 1):
        model.train()
        losses = []

        for x, y, _ in train_loader:
            x = x.unsqueeze(2).float().to(device)   # (B,T,1,H,W)
            y = y.unsqueeze(1).float().to(device)   # (B,1,H,W)

            # output already in [0,1] (sigmoid head)
            pred = model(x)

            # IMPORTANT: normalize by (sum(mask) * B) for batch-invariant scale
            denom_b = float(denom) * pred.shape[0]
            loss = masked_mse_torch(pred, y, wmask_t, denom_b)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        train_mse = float(np.mean(losses))

        val_mae, val_rmse = evaluate(model, val_loader, device, wmask, denom)
        print(f"[Epoch {ep:02d}] Train masked-MSE={train_mse:.6f} | Val MAE={val_mae:.4f} RMSE={val_rmse:.4f}")

        if best_val is None or val_rmse < best_val:
            best_val = val_rmse
            torch.save(model.state_dict(), best_path)
            print(f"  [SAVE] best model -> {best_path}")

    # ---------- test ----------
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_mae, test_rmse = evaluate(model, test_loader, device, wmask, denom)

    print("\n=== ConvLSTM Phase-1 (Test, masked) ===")
    print(f"MAE  : {test_mae:.4f}")
    print(f"RMSE : {test_rmse:.4f}")

    # ---------- save results ----------
    out_csv = Path("scripts/results/models_test_lead1.csv")
    append_result_row(out_csv, {
        "model": "convlstm_phase1",
        "split": "test",
        "input_window": input_window,
        "lead_time": lead_time,
        "mask": mask_name,
        "mae": f"{test_mae:.6f}",
        "rmse": f"{test_rmse:.6f}",
    })

    print(f"[OK] Results appended to: {out_csv}")


if __name__ == "__main__":
    main()
