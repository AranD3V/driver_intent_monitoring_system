"""
Gaze Predictor — train on DR(eye)VE, apply to HDD.

DR(eye)VE has real eye-tracker gaze; HDD has dead gaze dims (all-zero).
This MLP learns: scene_objects + vehicle_speed → gaze_x, gaze_y, is_fixation
so that hdd_train_scene.json sequences can have their gaze dims populated.

Usage:
    python scripts/train_gaze_predictor.py \
        --data data/dreyeve_train.json \
        --output models/gaze_predictor.pth \
        --epochs 200
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ------------------------------------------------------------------ #
#  Feature encoding constants (must match populate_gaze.py)           #
# ------------------------------------------------------------------ #

OBJECT_CLASSES = [
    'none', 'person', 'car', 'truck', 'bus',
    'motorcycle', 'bicycle', 'traffic_light', 'stop_sign'
]
TOP_K = 3
OBJ_DIM = len(OBJECT_CLASSES) + 3   # one-hot(9) + cx_norm + cy_norm + dist_norm
GAZE_INPUT_DIM = 1 + TOP_K * OBJ_DIM  # speed(1) + objects(3×12 = 36) = 37

SCENE_W, SCENE_H = 1280, 720
MAX_DIST = 80.0


# ------------------------------------------------------------------ #
#  Frame encoder                                                       #
# ------------------------------------------------------------------ #

def encode_frame(frame: dict) -> np.ndarray:
    feat = np.zeros(GAZE_INPUT_DIM, dtype=np.float32)

    v = frame.get('vehicle', {})
    feat[0] = min(float(v.get('speed_kmh', 0.0)) / 120.0, 1.0)

    objs = frame.get('detected_objects', [])
    objs = sorted(objs, key=lambda o: o.get('distance_estimate', 999.0))[:TOP_K]

    for i, obj in enumerate(objs):
        off = 1 + i * OBJ_DIM
        cls = obj.get('class', 'none')
        cls_idx = OBJECT_CLASSES.index(cls) if cls in OBJECT_CLASSES else 0
        feat[off + cls_idx] = 1.0
        bbox = obj.get('bbox', {})
        cx = float(bbox.get('center_x', SCENE_W / 2))
        cy = float(bbox.get('center_y', SCENE_H / 2))
        feat[off + len(OBJECT_CLASSES)]     = cx / SCENE_W
        feat[off + len(OBJECT_CLASSES) + 1] = cy / SCENE_H
        feat[off + len(OBJECT_CLASSES) + 2] = min(
            float(obj.get('distance_estimate', MAX_DIST)) / MAX_DIST, 1.0)

    return feat


def extract_gaze_target(frame: dict):
    gd = frame.get('gaze_data', {})
    gp = gd.get('gaze_point', [SCENE_W / 2, SCENE_H / 2])
    gx = float(gp[0]) / SCENE_W
    gy = float(gp[1]) / SCENE_H
    fix = 1.0 if gd.get('is_fixation', False) else 0.0
    conf = float(gd.get('confidence', 0.0))
    return np.array([gx, gy, fix], dtype=np.float32), conf


# ------------------------------------------------------------------ #
#  Dataset                                                             #
# ------------------------------------------------------------------ #

class GazeFrameDataset(Dataset):
    def __init__(self, sequences, min_confidence=0.5):
        X, y = [], []
        for seq in sequences:
            for frame in seq.get('frames', []):
                target, conf = extract_gaze_target(frame)
                if conf < min_confidence:
                    continue
                X.append(encode_frame(frame))
                y.append(target)

        if not X:
            raise RuntimeError(
                "No valid gaze frames found — check that DR(eye)VE data "
                "has confidence >= 0.5 in gaze_data fields.")

        self.X = torch.tensor(np.stack(X), dtype=torch.float32)
        self.y = torch.tensor(np.stack(y), dtype=torch.float32)
        print(f"  {len(self.X)} valid frames extracted")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ------------------------------------------------------------------ #
#  Model                                                               #
# ------------------------------------------------------------------ #

class GazePredictor(nn.Module):
    """
    MLP: scene_features → gaze_x, gaze_y, is_fixation  (all in [0, 1])
    """
    def __init__(self, input_dim=GAZE_INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


# ------------------------------------------------------------------ #
#  Training                                                            #
# ------------------------------------------------------------------ #

def train(args):
    with open(args.data) as f:
        sequences = json.load(f)
    print(f"Loaded {len(sequences)} DR(eye)VE sequences from {args.data}")

    dataset = GazeFrameDataset(sequences, min_confidence=0.5)

    n_val   = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}  |  train={n_train}  val={n_val}")

    model     = GazePredictor(input_dim=GAZE_INPUT_DIM).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_loss = float('inf')
    patience_left = 30

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            pred = model(X_b)
            loss = (nn.functional.mse_loss(pred[:, :2], y_b[:, :2]) +
                    0.3 * nn.functional.binary_cross_entropy(pred[:, 2], y_b[:, 2]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(X_b)
        t_loss /= n_train

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model(X_b)
                v_loss += (nn.functional.mse_loss(pred[:, :2], y_b[:, :2]) +
                           0.3 * nn.functional.binary_cross_entropy(
                               pred[:, 2], y_b[:, 2])).item() * len(X_b)
        v_loss /= n_val
        scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}/{args.epochs}  "
                  f"train={t_loss:.5f}  val={v_loss:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_left = 30
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': GAZE_INPUT_DIM,
                'val_loss': v_loss,
            }, args.output)
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  Early stop at epoch {epoch}")
                break

    print(f"\nBest val_loss = {best_val_loss:.5f}  ->  saved to {args.output}")


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Train gaze predictor on DR(eye)VE")
    parser.add_argument('--data',   default='data/dreyeve_train.json',
                        help='DR(eye)VE JSON with real gaze data')
    parser.add_argument('--output', default='models/gaze_predictor.pth')
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()
