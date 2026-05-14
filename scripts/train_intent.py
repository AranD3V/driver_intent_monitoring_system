"""
Train / Evaluate Driver Intent Model.

Subcommands:
  train        -- single 80/20 train/val split
  train-kfold  -- stratified 5-fold CV with OTC TTA (recommended)
  evaluate     -- accuracy + F1 on a dataset
  ensemble     -- average softmax over multiple checkpoints
"""

import sys, json, copy, argparse, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.temporal_model import (
    TemporalIntentModel, FeatureExtractor,
    FEATURE_DIM, INTENT_CLASSES,
    NATIVE_CLASSES, NATIVE_LABEL_ALIASES,
)
from modules.losses import AnticipatoryFocalLoss, build_warmup_cosine_scheduler
from modules.feature_augment import augment_train as _telemetry_aware_augment


# ── Constants ────────────────────────────────────────────────────────────────

OTC_MODES         = ('original', 'translate', 'cutout')
_RESTART_PATIENCE = 8
_MAX_RESTARTS     = 2


# ── OTC Test-Time Augmentation ───────────────────────────────────────────────

def _otc_transform(seqs: torch.Tensor, mode: str,
                   translate_frames: int = 3,
                   cutout_frames: int = 10) -> torch.Tensor:
    """Apply one OTC view to a batch of (B, T, F) sequences."""
    if mode == 'original':
        return seqs
    out = seqs.clone()
    T   = seqs.shape[1]
    if mode == 'translate':
        out = torch.roll(out, shifts=translate_frames, dims=1)
        out[:, :translate_frames, :] = 0.0
    elif mode == 'cutout':
        mid   = T // 2
        start = max(0, mid - cutout_frames // 2)
        end   = min(T, start + cutout_frames)
        out[:, start:end, :] = 0.0
    return out


@torch.no_grad()
def otc_predict(model: nn.Module, seqs: torch.Tensor,
                device: str, use_amp: bool = False) -> torch.Tensor:
    """Average softmax over all OTC views. Returns (B, C) probability tensor."""
    probs = None
    for mode in OTC_MODES:
        x = _otc_transform(seqs, mode).to(device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            p = torch.softmax(model(x), dim=1)
        probs = p if probs is None else probs + p
    return probs / len(OTC_MODES)


# ── Augmentation ─────────────────────────────────────────────────────────────

def _jitter(seq: np.ndarray) -> np.ndarray:
    """Zero-pad temporal shift +/-5 frames + small Gaussian noise."""
    T, F  = seq.shape
    shift = random.randint(-5, 5)
    if shift > 0:
        seq = np.concatenate([np.zeros((shift, F), np.float32), seq[:-shift]])
    elif shift < 0:
        seq = np.concatenate([seq[-shift:], np.zeros((-shift, F), np.float32)])
    return np.clip(seq + np.random.normal(0, 0.005, (T, F)).astype(np.float32),
                   0.0, 1.0)


# ── Dataset ──────────────────────────────────────────────────────────────────

class IntentSequenceDataset(Dataset):
    def __init__(self, data_paths, seq_len: int, stride: int = 10,
                 augment: bool = False, classes=None,
                 use_native_labels: bool = False,
                 max_wins_per_seq: int = 0,
                 weak_aug: bool = False,
                 use_weak_weights: bool = False,
                 min_weak_conf: float = 0.0):
        self.seq_len = seq_len
        self.augment = augment
        self.weak_aug = weak_aug                  # use modules.feature_augment pipeline
        self.use_weak_weights = use_weak_weights  # populate sample_weights from weak_label_meta
        self.min_weak_conf = float(min_weak_conf) # drop entries below this consensus_confidence
        self.classes = list(classes) if classes is not None else list(INTENT_CLASSES)

        self._sequences: list = []   # (ndarray T x F, label_idx, ttm)
        self._seq_ids:   list = []   # one entry per window, for CV grouping
        self.sample_weights: list = []  # per-window float, parallel to _sequences
        self._dropped_low_conf = 0

        paths = sorted(data_paths) if isinstance(data_paths, (list, tuple)) else [data_paths]
        for p in paths:
            self._load(str(p), stride, use_native_labels, max_wins_per_seq)

        if self.min_weak_conf > 0 and self._dropped_low_conf:
            print(f"  [min-weak-conf={self.min_weak_conf:.2f}] "
                  f"dropped {self._dropped_low_conf} sequences below threshold")

        counts = Counter(lbl for _, lbl, _ in self._sequences)
        print("Dataset: {:d} windows | ".format(len(self._sequences)) +
              " | ".join("{:s}={:d}".format(self.classes[c], counts[c])
                         for c in sorted(counts)))

    def _load(self, path: str, stride: int, use_native_labels: bool,
              max_wins_per_seq: int = 0):
        with open(path) as f:
            data = json.load(f)

        dropped = Counter()
        for idx, entry in enumerate(data):
            # ── Label ─────────────────────────────────────────────────────
            if use_native_labels:
                raw = entry.get('native_label') or entry.get('intent_label', '')
            else:
                raw = entry.get('intent_label', '')
                raw = NATIVE_LABEL_ALIASES.get(raw, raw)

            if raw not in self.classes:
                dropped[raw or '(missing)'] += 1
                continue
            label_idx = self.classes.index(raw)

            # ── Drop low-confidence weak labels (audit-driven filter) ─────
            if self.min_weak_conf > 0:
                meta = entry.get('weak_label_meta') or {}
                conf = float(meta.get('consensus_confidence', 1.0) or 1.0)
                if conf < self.min_weak_conf:
                    self._dropped_low_conf += 1
                    continue

            # ── Sequence ID (for group-aware CV) ──────────────────────────
            seq_id = (entry.get('seq_id')
                      or entry.get('clip_id')
                      or '{:s}_{:d}'.format(Path(path).stem, idx))

            ttm = float(entry.get('ttm', 0.0))

            # ── Per-sequence weight from weak_label_meta (optional) ───────
            seq_weight = 1.0
            if self.use_weak_weights:
                meta = entry.get('weak_label_meta') or {}
                conf = float(meta.get('consensus_confidence', 0.0) or 0.0)
                n_voters = int(meta.get('n_voters', 0) or 0)
                if n_voters >= 1:
                    # Linear blend: 0 voters -> 1.0 (trust label), 2 voters -> 1.0 + conf
                    seq_weight = 1.0 + 0.5 * n_voters * conf
                else:
                    # No labeler agreed — original label might be noisy, downweight
                    seq_weight = 0.5

            # ── Feature extraction (fresh extractor = reset _prev_vehicle) ─
            ext   = FeatureExtractor()
            feats = np.array([ext.extract(fd) for fd in entry.get('frames', [])],
                             dtype=np.float32)

            if len(feats) < self.seq_len:
                dropped[raw] += 1
                continue

            # ── Sliding window ─────────────────────────────────────────────
            starts = list(range(0, len(feats) - self.seq_len + 1, stride))
            if max_wins_per_seq and len(starts) > max_wins_per_seq:
                # Evenly spaced subset — preserves coverage of the sequence
                idx = np.linspace(0, len(starts) - 1, max_wins_per_seq, dtype=int)
                starts = [starts[i] for i in idx]
            for start in starts:
                self._sequences.append(
                    (feats[start:start + self.seq_len], label_idx, ttm))
                self._seq_ids.append(seq_id)
                self.sample_weights.append(seq_weight)

        if dropped:
            print("  [!] {:s}: skipped ".format(Path(path).name) +
                  ", ".join("{:s}={:d}".format(k, v)
                             for k, v in sorted(dropped.items())))

    def __len__(self):
        return len(self._sequences)

    def __getitem__(self, idx):
        seq, label, ttm = self._sequences[idx]
        if self.augment:
            seq = (_telemetry_aware_augment(seq) if self.weak_aug
                   else _jitter(seq))
        return (torch.FloatTensor(seq),
                torch.tensor(label, dtype=torch.long),
                torch.tensor(ttm,   dtype=torch.float))


class _Subset(Dataset):
    """Index-based view of IntentSequenceDataset with optional augmentation."""

    def __init__(self, base: IntentSequenceDataset,
                 indices: list, augment: bool = False):
        self.base    = base
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        seq, label, ttm = self.base._sequences[self.indices[i]]
        if self.augment:
            seq = (_telemetry_aware_augment(seq) if getattr(self.base, 'weak_aug', False)
                   else _jitter(seq))
        return (torch.FloatTensor(seq),
                torch.tensor(label, dtype=torch.long),
                torch.tensor(ttm,   dtype=torch.float))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _class_weights(label_indices, n_classes: int, device: str) -> torch.Tensor:
    counts = [max(Counter(label_indices)[i], 1) for i in range(n_classes)]
    return AnticipatoryFocalLoss.compute_alpha_from_counts(
        counts, scheme='sqrt_inv_freq').to(device)


def _build_model(args, n_classes: int, device: str) -> TemporalIntentModel:
    return TemporalIntentModel(
        input_size    = FEATURE_DIM,
        stream_hidden = args.hidden,
        num_layers    = 2,
        num_heads     = 4,
        num_classes   = n_classes,
        dropout       = args.dropout,
    ).to(device)


def _load_model_from_ckpt(ckpt: dict, device: str) -> tuple:
    """Returns (model, classes, uses_native_labels)."""
    arch        = ckpt.get('arch', {})
    classes     = ckpt.get('classes') or list(INTENT_CLASSES)
    uses_native = ckpt.get('uses_native_labels', False)
    if 'stream_hidden' not in arch:
        w = (ckpt['model_state_dict'].get('lstm_in.weight_ih_l0')
             or ckpt['model_state_dict'].get('lstm.weight_ih_l0'))
        arch['stream_hidden'] = int(w.shape[0] // 4) if w is not None else 64
    model = TemporalIntentModel(
        input_size    = FEATURE_DIM,
        stream_hidden = arch.get('stream_hidden', 64),
        num_layers    = arch.get('num_layers', 2),
        num_heads     = arch.get('num_heads', 4),
        num_classes   = len(classes),
        dropout       = arch.get('dropout', 0.3),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return model, classes, uses_native


def _ckpt_payload(model, epoch, val_acc, val_loss, classes,
                  args, fold=None) -> dict:
    d = {
        'epoch':              epoch,
        'model_state_dict':   model.state_dict(),
        'val_acc':            val_acc,
        'val_loss':           val_loss,
        'classes':            list(classes),
        'uses_native_labels': getattr(args, 'native_labels', False),
        'arch': {
            'stream_hidden': args.hidden,
            'num_layers':    2,
            'num_heads':     4,
            'dropout':       args.dropout,
            'feature_dim':   FEATURE_DIM,
            'num_classes':   len(classes),
        },
    }
    if fold is not None:
        d['fold'] = fold
    return d


def _save_confusion_matrix(labels, preds, out_dir: Path, class_names):
    cm  = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(max(4, len(class_names) * 1.6),
                                    max(3, len(class_names) * 1.6)))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ticks = list(range(len(class_names)))
    ax.set(xticks=ticks, yticks=ticks,
           xticklabels=class_names, yticklabels=class_names,
           xlabel='Predicted', ylabel='True', title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
    mx = cm.max() or 1
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > mx * 0.5 else 'black',
                    fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrix.png', dpi=150)
    plt.close()
    print("Confusion matrix -> {:s}".format(str(out_dir / 'confusion_matrix.png')))


def _save_training_curves(history: dict, out_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'],   label='Val')
    ax1.set(title='Loss', xlabel='Epoch'); ax1.legend()
    ax2.plot(history['val_acc'])
    ax2.set(title='Val Accuracy (%)', xlabel='Epoch')
    plt.tight_layout()
    plt.savefig(out_dir / 'training_curves.png', dpi=150)
    plt.close()
    print("Training curves -> {:s}".format(str(out_dir / 'training_curves.png')))


def _seq_level_split(full_ds: IntentSequenceDataset, val_frac: float = 0.2,
                     seed: int = 42):
    """Stratified seq-level split. Returns (train_indices, val_indices)."""
    groups    = defaultdict(list)
    sid_label = {}
    for i, (sid, (_, lbl, _)) in enumerate(
            zip(full_ds._seq_ids, full_ds._sequences)):
        groups[sid].append(i)
        sid_label[sid] = lbl

    by_class = defaultdict(list)
    for sid, lbl in sid_label.items():
        by_class[lbl].append(sid)

    rng      = random.Random(seed)
    val_sids = set()
    for lbl, sids in by_class.items():
        rng.shuffle(sids)
        val_sids.update(sids[:max(1, round(len(sids) * val_frac))])

    train_idx = [i for sid, idxs in groups.items()
                 if sid not in val_sids for i in idxs]
    val_idx   = [i for sid in val_sids for i in groups[sid]]
    return train_idx, val_idx


def _make_loaders(full_ds, train_idx, val_idx, batch, balance_classes: bool = False):
    nw = 0 if sys.platform == 'win32' else 4
    pin = torch.cuda.is_available()
    train_set = _Subset(full_ds, train_idx, augment=True)

    sampler = None
    shuffle = True
    sw = getattr(full_ds, 'sample_weights', None)

    if balance_classes:
        # Per-sample weight = 1 / class_count. Each batch becomes class-balanced
        # in expectation, regardless of raw frequency. Multiplies with any
        # weak-supervision weights already on the dataset.
        from torch.utils.data import WeightedRandomSampler
        train_labels = [full_ds._sequences[i][1] for i in train_idx]
        counts = Counter(train_labels)
        n_classes = len(getattr(full_ds, 'classes', [])) or (max(counts) + 1)
        inv = [1.0 / max(counts.get(c, 0), 1) for c in range(n_classes)]
        base = sw if sw else [1.0] * len(full_ds._sequences)
        weights = torch.tensor(
            [inv[full_ds._sequences[i][1]] * base[i] for i in train_idx],
            dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                        replacement=True)
        shuffle = False
        print(f"  balance-classes sampler: per-class counts={dict(counts)} "
              f"(weight range {weights.min():.4f}..{weights.max():.4f})")
    elif sw and getattr(full_ds, 'use_weak_weights', False) and any(w != 1.0 for w in sw):
        from torch.utils.data import WeightedRandomSampler
        weights = torch.tensor([sw[i] for i in train_idx], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                        replacement=True)
        shuffle = False
        print(f"  using WeightedRandomSampler (weight range: "
              f"{weights.min():.2f}..{weights.max():.2f})")

    return (
        DataLoader(train_set, batch_size=batch, shuffle=shuffle,
                   sampler=sampler, num_workers=nw, pin_memory=pin),
        DataLoader(_Subset(full_ds, val_idx,   augment=False),
                   batch_size=batch, shuffle=False,
                   num_workers=0, pin_memory=pin),
    )


# ── Training: single split ────────────────────────────────────────────────────

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {:s}".format(device))
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    classes = NATIVE_CLASSES if args.native_labels else INTENT_CLASSES
    full_ds = IntentSequenceDataset(
        sorted(args.data), args.seq_len, stride=args.stride,
        augment=False, classes=classes, use_native_labels=args.native_labels,
        max_wins_per_seq=args.max_wins_per_seq,
        weak_aug=getattr(args, 'weak_aug', False),
        use_weak_weights=getattr(args, 'use_weak_weights', False),
        min_weak_conf=getattr(args, 'min_weak_conf', 0.0))

    train_idx, val_idx = _seq_level_split(full_ds, seed=42)
    print("Split: {:d} train / {:d} val windows".format(
        len(train_idx), len(val_idx)))

    train_loader, val_loader = _make_loaders(
        full_ds, train_idx, val_idx, args.batch,
        balance_classes=getattr(args, 'balance_classes', False))
    model = _build_model(args, len(classes), device)

    ckpt_path = Path(args.output)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    prior_best = -1.0
    if ckpt_path.exists() and not args.overwrite:
        try:
            prior_best = float(
                torch.load(ckpt_path, map_location='cpu').get('val_acc', -1.0))
            print("Existing checkpoint val_acc={:.1f}% -- "
                  "will only overwrite on improvement.".format(prior_best))
        except Exception:
            pass
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        prior_best = max(prior_best, float(ckpt.get('val_acc', -1.0)))
        print("Resumed (prior val_acc={:.1f}%)".format(prior_best))

    train_labels = [full_ds._sequences[i][1] for i in train_idx]
    cw = _class_weights(train_labels, len(classes), device)
    print("Class weights: " +
          " ".join("{:s}={:.3f}".format(classes[i], cw[i].item())
                   for i in range(len(classes))))

    criterion    = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    optimizer    = optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    total_steps  = len(train_loader) * args.epochs
    scheduler    = build_warmup_cosine_scheduler(
        optimizer, max(1, int(0.05 * total_steps)), total_steps, min_lr_ratio=0.01)
    use_amp = (device == 'cuda')
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val_acc  = prior_best
    best_state    = None
    patience      = 0
    restarts_left = _MAX_RESTARTS
    history       = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    out_dir       = ckpt_path.parent

    for epoch in range(args.epochs):
        model.train()
        tl = tc = tt = 0
        for seqs, labels, _ in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(seqs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            tl += loss.item()
            tc += (logits.detach().argmax(1) == labels).sum().item()
            tt += labels.size(0)
        avg_tl    = tl / max(1, len(train_loader))
        train_acc = 100.0 * tc / max(1, tt)

        model.eval()
        vl = vc = vt = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for seqs, labels, _ in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    vl += criterion(model(seqs), labels).item()
                probs = otc_predict(model, seqs, device, use_amp)
                preds = probs.argmax(1)
                vc += (preds == labels).sum().item(); vt += labels.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        avg_vl  = vl / max(1, len(val_loader))
        val_acc = 100.0 * vc / max(1, vt)
        history['train_loss'].append(avg_tl)
        history['val_loss'].append(avg_vl)
        history['val_acc'].append(val_acc)

        print("Epoch {:3d}/{:d} | Train {:.4f} {:.1f}% "
              "| Val {:.4f} OTC {:.1f}% | LR {:.2e} | Best {:.1f}%".format(
              epoch + 1, args.epochs, avg_tl, train_acc,
              avg_vl, val_acc, scheduler.get_last_lr()[0], best_val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(model.state_dict())
            patience     = 0
            torch.save(_ckpt_payload(model, epoch, val_acc, avg_vl, classes, args),
                       ckpt_path)
            print("  -> Saved (val_acc={:.1f}%)".format(val_acc))
            present = sorted(set(all_labels))
            print(classification_report(
                all_labels, all_preds,
                labels=present,
                target_names=[classes[i] for i in present],
                zero_division=0))
            _save_confusion_matrix(all_labels, all_preds, out_dir,
                                   [classes[i] for i in present])
        else:
            patience += 1
            if (best_state is not None and restarts_left > 0
                    and patience >= _RESTART_PATIENCE):
                model.load_state_dict(best_state)
                new_lr = args.lr / (2 ** (_MAX_RESTARTS - restarts_left + 1))
                optimizer = optim.AdamW(model.parameters(), lr=new_lr,
                                        weight_decay=args.weight_decay)
                elapsed  = (epoch + 1) * len(train_loader)
                scheduler = build_warmup_cosine_scheduler(
                    optimizer, 0, max(1, total_steps - elapsed), min_lr_ratio=0.01)
                restarts_left -= 1; patience = 0
                print("  Warm restart ({:d} left) -- LR {:.2e}".format(
                    restarts_left, new_lr))
            elif args.early_stop > 0 and patience >= args.early_stop:
                print("Early stop after {:d} epochs without improvement.".format(patience))
                break

    _save_training_curves(history, out_dir)
    print("\nBest val_acc={:.1f}% -> {:s}".format(best_val_acc, str(ckpt_path)))


# ── Training: k-fold CV ───────────────────────────────────────────────────────

def train_kfold(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: {:s}".format(device))
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    classes = NATIVE_CLASSES if args.native_labels else INTENT_CLASSES
    print("{:d}-Fold CV | {:d} classes".format(args.n_folds, len(classes)))

    full_ds = IntentSequenceDataset(
        sorted(args.data), args.seq_len, stride=args.stride,
        augment=False, classes=classes, use_native_labels=args.native_labels,
        max_wins_per_seq=args.max_wins_per_seq,
        weak_aug=getattr(args, 'weak_aug', False),
        use_weak_weights=getattr(args, 'use_weak_weights', False),
        min_weak_conf=getattr(args, 'min_weak_conf', 0.0))

    n     = len(full_ds._sequences)
    y_arr = np.array([lbl for _, lbl, _ in full_ds._sequences])
    g_arr = np.array(full_ds._seq_ids)

    skf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True,
                               random_state=args.seed)

    ckpt_path   = Path(args.output)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_stem   = ckpt_path.stem
    ckpt_suffix = ckpt_path.suffix
    out_dir     = ckpt_path.parent

    global_best_val_acc = -1.0
    if ckpt_path.exists() and not args.overwrite:
        try:
            prev = float(torch.load(ckpt_path, map_location='cpu')
                         .get('val_acc', -1.0))
            global_best_val_acc = prev
            print("Locked floor: existing checkpoint val_acc={:.1f}% "
                  "-- use --overwrite to ignore.".format(prev))
        except Exception:
            pass

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(n), y_arr, g_arr)):

        print("\n{}\nFold {:d}/{:d}\n{}".format(
            '=' * 64, fold_idx + 1, args.n_folds, '=' * 64))
        val_counts = Counter(y_arr[val_idx].tolist())
        print("  Train: {:d} windows | Val: {:d} windows".format(
            len(train_idx), len(val_idx)))
        print("  Val class counts: " +
              ", ".join("{:s}={:d}".format(classes[c], val_counts.get(c, 0))
                        for c in range(len(classes))))

        train_loader, val_loader = _make_loaders(
            full_ds, train_idx.tolist(), val_idx.tolist(), args.batch,
            balance_classes=getattr(args, 'balance_classes', False))

        model     = _build_model(args, len(classes), device)
        cw        = _class_weights(y_arr[train_idx].tolist(), len(classes), device)
        print("  Class weights: " +
              " ".join("{:s}={:.3f}".format(classes[i], cw[i].item())
                       for i in range(len(classes))))

        criterion    = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
        optimizer    = optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
        total_steps  = len(train_loader) * args.epochs
        scheduler    = build_warmup_cosine_scheduler(
            optimizer, max(1, int(0.05 * total_steps)), total_steps,
            min_lr_ratio=0.01)
        use_amp = (device == 'cuda')
        scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

        fold_ckpt     = out_dir / '{:s}_fold{:d}{:s}'.format(
            ckpt_stem, fold_idx, ckpt_suffix)
        best_val_acc  = -1.0
        best_state    = None
        best_preds    = best_labels = None
        best_report   = None
        patience      = 0
        restarts_left = _MAX_RESTARTS

        for epoch in range(args.epochs):
            model.train()
            tl = tc = tt = 0
            for seqs, labels, _ in train_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(seqs)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); scheduler.step()
                tl += loss.item()
                tc += (logits.detach().argmax(1) == labels).sum().item()
                tt += labels.size(0)
            avg_tl    = tl / max(1, len(train_loader))
            train_acc = 100.0 * tc / max(1, tt)

            model.eval()
            vl = vc = vt = 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for seqs, labels, _ in val_loader:
                    seqs, labels = seqs.to(device), labels.to(device)
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        vl += criterion(model(seqs), labels).item()
                    probs = otc_predict(model, seqs, device, use_amp)
                    preds = probs.argmax(1)
                    vc += (preds == labels).sum().item(); vt += labels.size(0)
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
            avg_vl  = vl / max(1, len(val_loader))
            val_acc = 100.0 * vc / max(1, vt)

            print("  [F{:d}] Epoch {:3d}/{:d} "
                  "| Train {:.4f} {:.1f}% "
                  "| Val {:.4f} OTC {:.1f}% "
                  "| LR {:.2e} | Best {:.1f}%".format(
                  fold_idx, epoch + 1, args.epochs,
                  avg_tl, train_acc, avg_vl, val_acc,
                  scheduler.get_last_lr()[0], best_val_acc))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = copy.deepcopy(model.state_dict())
                best_preds   = list(all_preds)
                best_labels  = list(all_labels)
                best_report  = classification_report(
                    all_labels, all_preds,
                    labels=list(range(len(classes))),
                    target_names=classes, zero_division=0, output_dict=True)
                patience = 0
                payload  = _ckpt_payload(model, epoch, val_acc, avg_vl,
                                         classes, args, fold=fold_idx)
                torch.save(payload, fold_ckpt)
                if val_acc > global_best_val_acc:
                    global_best_val_acc = val_acc
                    torch.save(payload, ckpt_path)
                    print("  *** GLOBAL BEST: fold {:d} epoch {:d} "
                          "val_acc={:.1f}% -> {:s}".format(
                          fold_idx + 1, epoch + 1, val_acc, str(ckpt_path)))
            else:
                patience += 1
                if (best_state is not None and restarts_left > 0
                        and patience >= _RESTART_PATIENCE):
                    model.load_state_dict(best_state)
                    new_lr = args.lr / (2 ** (_MAX_RESTARTS - restarts_left + 1))
                    optimizer = optim.AdamW(model.parameters(), lr=new_lr,
                                            weight_decay=args.weight_decay)
                    elapsed  = (epoch + 1) * len(train_loader)
                    scheduler = build_warmup_cosine_scheduler(
                        optimizer, 0, max(1, total_steps - elapsed),
                        min_lr_ratio=0.01)
                    restarts_left -= 1; patience = 0
                    print("  Warm restart ({:d} left) -- LR {:.2e}".format(
                        restarts_left, new_lr))
                elif args.early_stop > 0 and patience >= args.early_stop:
                    print("  Early stop (best val_acc={:.1f}%)".format(best_val_acc))
                    break

        fold_results.append({'fold': fold_idx, 'val_acc': best_val_acc,
                             'report': best_report})
        if best_labels is not None:
            present = sorted(set(best_labels))
            print(classification_report(
                best_labels, best_preds,
                labels=present,
                target_names=[classes[i] for i in present],
                zero_division=0))

    # ── Summary ───────────────────────────────────────────────────────────
    accs = [r['val_acc'] for r in fold_results]
    print("\n{}\n5-Fold CV Summary\n{}".format('=' * 64, '=' * 64))
    print("  Per-fold OTC val acc: {:s}".format(
        ', '.join("{:.2f}%".format(a) for a in accs)))
    print("  Mean: {:.2f}% +/- {:.2f}%".format(
        float(np.mean(accs)), float(np.std(accs))))

    all_f1 = defaultdict(list)
    for r in fold_results:
        if r['report']:
            for c in classes:
                all_f1[c].append(r['report'].get(c, {}).get('f1-score', 0.0))
    if all_f1:
        print("\n  Per-class F1 (mean +/- std across folds):")
        for c in classes:
            vals = all_f1[c]
            print("    {:<30} {:.3f} +/- {:.3f}".format(
                c, float(np.mean(vals)), float(np.std(vals))))

    summary = {
        'mean_val_acc': float(np.mean(accs)),
        'std_val_acc':  float(np.std(accs)),
        'fold_accs':    [float(a) for a in accs],
        'per_class_f1': {c: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                         for c, v in all_f1.items()},
    }
    summary_path = out_dir / 'kfold_summary.json'
    with open(summary_path, 'w') as fp:
        json.dump(summary, fp, indent=2)
    print("\n  Summary saved -> {:s}".format(str(summary_path)))
    print("  Best model (val_acc={:.1f}%) saved -> {:s}".format(
        global_best_val_acc, str(ckpt_path)))

    if not getattr(args, 'keep_folds', False):
        for fi in range(args.n_folds):
            p = out_dir / '{:s}_fold{:d}{:s}'.format(ckpt_stem, fi, ckpt_suffix)
            if p.exists():
                p.unlink()
        print("  Per-fold checkpoints removed.")
    else:
        kept = [out_dir / '{:s}_fold{:d}{:s}'.format(ckpt_stem, fi, ckpt_suffix)
                for fi in range(args.n_folds)]
        kept = [p for p in kept if p.exists()]
        print("  Kept {:d} per-fold checkpoints for ensembling:".format(len(kept)))
        for p in kept:
            print("    {:s}".format(str(p)))


# ── Evaluate ─────────────────────────────────────────────────────────────────

def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt   = torch.load(args.output, map_location=device)
    model, classes, uses_native = _load_model_from_ckpt(ckpt, device)
    model.eval()
    print("Loaded {:s} (val_acc={:.1f}%, classes={:s})".format(
        args.output, ckpt.get('val_acc', -1.0), str(classes)))

    ds = IntentSequenceDataset(
        sorted(args.data), args.seq_len, stride=args.seq_len,
        classes=classes, use_native_labels=uses_native)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, labels, _ in loader:
            seqs = seqs.to(device)
            if args.otc:
                probs = otc_predict(model, seqs, device)
            else:
                probs = torch.softmax(model(seqs), dim=1)
            all_preds.extend(probs.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())

    present = sorted(set(all_labels) | set(all_preds))
    names   = [classes[i] for i in present]
    print(classification_report(all_labels, all_preds,
          labels=present, target_names=names, zero_division=0))
    _save_confusion_matrix(all_labels, all_preds,
                           Path(args.output).parent, names)


# ── Ensemble ──────────────────────────────────────────────────────────────────

def ensemble(args):
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    models  = []
    classes = None
    uses_native = False

    for mp in args.models:
        ckpt = torch.load(mp, map_location=device)
        m, cls, un = _load_model_from_ckpt(ckpt, device)
        if classes is None:
            classes = cls; uses_native = un
        elif cls != classes:
            raise ValueError(
                "Class mismatch between ensemble members: {:s} vs {:s}".format(
                    str(classes), str(cls)))
        m.eval()
        models.append(m)
        print("  Loaded {:s} (val_acc={:.1f}%)".format(
            mp, ckpt.get('val_acc', -1.0)))

    ds = IntentSequenceDataset(
        sorted(args.data), args.seq_len, stride=args.seq_len,
        classes=classes, use_native_labels=uses_native)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, labels, _ in loader:
            seqs  = seqs.to(device)
            probs = sum(
                (otc_predict(m, seqs, device) if args.otc
                 else torch.softmax(m(seqs), dim=1))
                for m in models
            ) / len(models)
            all_preds.extend(probs.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())

    present = sorted(set(all_labels) | set(all_preds))
    names   = [classes[i] for i in present]
    print(classification_report(all_labels, all_preds,
          labels=present, target_names=names, zero_division=0))
    _save_confusion_matrix(all_labels, all_preds,
                           Path(args.models[0]).parent, names)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train / Evaluate Intent Model')
    sub    = parser.add_subparsers(dest='cmd')

    # ── train ──────────────────────────────────────────────────────────────
    tr = sub.add_parser('train')
    tr.add_argument('--data',         nargs='+', required=True)
    tr.add_argument('--output',       default='models/intent_model.pth')
    tr.add_argument('--epochs',       type=int,   default=80)
    tr.add_argument('--batch',        type=int,   default=8)
    tr.add_argument('--lr',           type=float, default=3e-4)
    tr.add_argument('--seq-len',      type=int,   default=50,  dest='seq_len')
    tr.add_argument('--stride',            type=int,   default=10)
    tr.add_argument('--max-wins-per-seq',  type=int,   default=0,
                    dest='max_wins_per_seq',
                    help='Cap windows per sequence (0=unlimited)')
    tr.add_argument('--early-stop',   type=int,   default=25,  dest='early_stop')
    tr.add_argument('--hidden',       type=int,   default=128)
    tr.add_argument('--dropout',      type=float, default=0.3)
    tr.add_argument('--weight-decay', type=float, default=1e-2, dest='weight_decay')
    tr.add_argument('--native-labels', action='store_true', dest='native_labels')
    tr.add_argument('--resume',       action='store_true')
    tr.add_argument('--overwrite',    action='store_true')
    tr.add_argument('--seed',         type=int,   default=42)
    tr.add_argument('--weak-aug',     action='store_true', dest='weak_aug',
                    help='Use telemetry-aware augmentation (modules.feature_augment)')
    tr.add_argument('--use-weak-weights', action='store_true', dest='use_weak_weights',
                    help='Read consensus_confidence from weak_label_meta as sample weight')
    tr.add_argument('--balance-classes', action='store_true', dest='balance_classes',
                    help='Per-class WeightedRandomSampler so each batch is class-balanced')
    tr.add_argument('--min-weak-conf', type=float, default=0.0, dest='min_weak_conf',
                    help='Drop sequences with weak_label_meta.consensus_confidence below this')

    # ── train-kfold ────────────────────────────────────────────────────────
    kf = sub.add_parser('train-kfold')
    kf.add_argument('--data',         nargs='+', required=True)
    kf.add_argument('--output',       default='models/intent_model.pth')
    kf.add_argument('--n-folds',      type=int,   default=5,   dest='n_folds')
    kf.add_argument('--epochs',       type=int,   default=80)
    kf.add_argument('--batch',        type=int,   default=8)
    kf.add_argument('--lr',           type=float, default=3e-4)
    kf.add_argument('--seq-len',      type=int,   default=50,  dest='seq_len')
    kf.add_argument('--stride',            type=int,   default=10)
    kf.add_argument('--max-wins-per-seq',  type=int,   default=0,
                    dest='max_wins_per_seq',
                    help='Cap windows per sequence (0=unlimited)')
    kf.add_argument('--early-stop',   type=int,   default=25,  dest='early_stop')
    kf.add_argument('--hidden',       type=int,   default=128)
    kf.add_argument('--dropout',      type=float, default=0.3)
    kf.add_argument('--weight-decay', type=float, default=1e-2, dest='weight_decay')
    kf.add_argument('--native-labels', action='store_true', dest='native_labels')
    kf.add_argument('--overwrite',    action='store_true')
    kf.add_argument('--seed',         type=int,   default=42)
    kf.add_argument('--weak-aug',     action='store_true', dest='weak_aug',
                    help='Use telemetry-aware augmentation (modules.feature_augment)')
    kf.add_argument('--use-weak-weights', action='store_true', dest='use_weak_weights',
                    help='Read consensus_confidence from weak_label_meta as sample weight')
    kf.add_argument('--keep-folds',   action='store_true', dest='keep_folds',
                    help='Preserve per-fold checkpoints for ensembling')
    kf.add_argument('--balance-classes', action='store_true', dest='balance_classes',
                    help='Per-class WeightedRandomSampler so each batch is class-balanced')
    kf.add_argument('--min-weak-conf', type=float, default=0.0, dest='min_weak_conf',
                    help='Drop sequences with weak_label_meta.consensus_confidence below this')

    # ── evaluate ────────────────────────────────────────────────────────────
    ev = sub.add_parser('evaluate')
    ev.add_argument('--data',    nargs='+', required=True)
    ev.add_argument('--output',  required=True)
    ev.add_argument('--batch',   type=int, default=32)
    ev.add_argument('--seq-len', type=int, default=50, dest='seq_len')
    ev.add_argument('--otc',     action='store_true')

    # ── ensemble ────────────────────────────────────────────────────────────
    en = sub.add_parser('ensemble')
    en.add_argument('--data',    nargs='+', required=True)
    en.add_argument('--models',  nargs='+', required=True)
    en.add_argument('--batch',   type=int, default=32)
    en.add_argument('--seq-len', type=int, default=50, dest='seq_len')
    en.add_argument('--otc',     action='store_true')

    args = parser.parse_args()
    if   args.cmd == 'train':        train(args)
    elif args.cmd == 'train-kfold':  train_kfold(args)
    elif args.cmd == 'evaluate':     evaluate(args)
    elif args.cmd == 'ensemble':     ensemble(args)
    else:                            parser.print_help()


if __name__ == '__main__':
    main()
