"""
Build a prioritized review queue for human re-annotation.

Combines three signals:
  1. Audit warnings  (scripts/audit_labels.py)        : telemetry inconsistency
  2. Consensus disagreements (scripts/consensus_label.py) : weak labelers vote against original
  3. Model uncertainty (optional)                       : top-1 vs top-2 logit margin

Each sequence gets a priority score in [0, 1]. Higher = more informative to
re-annotate. Output CSV is sorted high-priority first so you can crop the
top-N for a one-day annotation sprint.

Usage:
  # No model needed (audit + disagreements only):
  python scripts/build_review_queue.py \\
      --audit reports/audit_hdd_train.csv \\
      --disagreements reports/consensus_disagreements.csv \\
      --out reports/review_queue.csv

  # With model uncertainty (recommended):
  python scripts/build_review_queue.py \\
      --audit reports/audit_hdd_train.csv \\
      --disagreements reports/consensus_disagreements.csv \\
      --data data/hdd_train.json \\
      --model models/intent_combined.pth \\
      --out reports/review_queue.csv \\
      --top-n 300
"""

import sys, csv, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_csv(path):
    if not path or not Path(path).exists():
        return []
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _model_uncertainty(data_path: str, model_path: str, seq_len: int):
    """
    Returns {seq_id: margin}. Smaller margin = more uncertain.
    Margin = top1_softmax - top2_softmax averaged over all windows of the seq.
    """
    import numpy as np, torch
    from modules.temporal_model import (
        TemporalIntentModel, FeatureExtractor, FEATURE_DIM, INTENT_CLASSES,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    arch = ckpt.get('arch', {})
    classes = ckpt.get('classes') or list(INTENT_CLASSES)

    model = TemporalIntentModel(
        input_size    = FEATURE_DIM,
        stream_hidden = arch.get('stream_hidden', 64),
        num_layers    = arch.get('num_layers', 2),
        num_heads     = arch.get('num_heads', 4),
        num_classes   = len(classes),
        dropout       = arch.get('dropout', 0.3),
    ).to(device).eval()
    model.load_state_dict(ckpt['model_state_dict'])

    margins = {}
    with open(data_path, encoding='utf-8') as f:
        data = json.load(f)
    for entry in data:
        seq_id = entry.get('seq_id', '')
        ext = FeatureExtractor()
        feats = np.array([ext.extract(fd) for fd in entry.get('frames', [])],
                         dtype=np.float32)
        if len(feats) < seq_len:
            continue
        wins = []
        for s in range(0, len(feats) - seq_len + 1, max(1, seq_len // 2)):
            wins.append(feats[s:s + seq_len])
        if not wins:
            continue
        x = torch.from_numpy(np.stack(wins)).to(device)
        with torch.no_grad():
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)
            top    = probs.topk(2, dim=1).values
            margin = (top[:, 0] - top[:, 1]).mean().item()
        margins[seq_id] = float(margin)
    return margins, classes


def _norm(x, lo=0.0, hi=1.0):
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit',          help='audit CSV path')
    ap.add_argument('--disagreements',  help='consensus disagreements CSV path')
    ap.add_argument('--data',           help='source dataset JSON (for model uncertainty)')
    ap.add_argument('--model',          help='trained model checkpoint (.pth)')
    ap.add_argument('--seq-len',        type=int, default=50, dest='seq_len')
    ap.add_argument('--out',            default='reports/review_queue.csv')
    ap.add_argument('--top-n',          type=int, default=0,
                    help='Truncate to top-N by priority (0=keep all)')
    args = ap.parse_args()

    rows: dict = {}   # composite key -> aggregated row

    def _key(source, seq_id, idx):
        return f"{source or '?'}::{seq_id or '?'}::{idx or '?'}"

    # ── Audit warnings ────────────────────────────────────────────────────
    for r in _load_csv(args.audit):
        sid = r.get('seq_id', '')
        idx = r.get('idx', '')
        src = r.get('source', '')
        k   = _key(src, sid, idx)
        rows[k] = {
            'key':          k,
            'seq_id':       sid,
            'row_idx':      idx,
            'source':       src,
            'intent_label': r.get('intent_label', ''),
            'raw_label':    r.get('raw_label', ''),
            'audit_warns':  r.get('warnings', ''),
            'audit_flag':   1,
            'consensus':    '',
            'consensus_voters': '',
            'consensus_conf':   '',
            'model_margin': '',
        }

    # ── Consensus disagreements ───────────────────────────────────────────
    for r in _load_csv(args.disagreements):
        sid = r.get('seq_id', '')
        idx = r.get('row_idx', '')
        src = r.get('source', '')
        k   = _key(src, sid, idx)
        if k not in rows:
            rows[k] = {
                'key':          k,
                'seq_id':       sid,
                'row_idx':      idx,
                'source':       src,
                'intent_label': r.get('original', ''),
                'raw_label':    r.get('raw_label', ''),
                'audit_warns':  '',
                'audit_flag':   0,
                'consensus':    '',
                'consensus_voters': '',
                'consensus_conf':   '',
                'model_margin': '',
            }
        rows[k]['consensus']        = r.get('consensus', '')
        rows[k]['consensus_voters'] = r.get('voters', '') or r.get('n_voters', '')
        rows[k]['consensus_conf']   = r.get('consensus_conf', '')

    # ── Model uncertainty (optional) ──────────────────────────────────────
    margins = {}
    if args.model and args.data:
        print(f"Computing model uncertainty on {args.data} ...")
        margins, _classes = _model_uncertainty(args.data, args.model, args.seq_len)
        for sid, m in margins.items():
            if sid not in rows:
                rows[sid] = {
                    'seq_id': sid, 'source': '', 'intent_label': '',
                    'raw_label': '', 'audit_warns': '', 'audit_flag': 0,
                    'consensus': '', 'consensus_voters': '',
                    'consensus_conf': '', 'model_margin': '',
                }
            rows[sid]['model_margin'] = f"{m:.3f}"

    # ── Compute priority ──────────────────────────────────────────────────
    # priority = w1 * audit_flag + w2 * disagree_signal + w3 * uncertainty
    # disagree_signal = consensus_conf (if differs from original)
    # uncertainty = 1 - normalized_margin
    margin_values = list(margins.values()) if margins else []
    margin_lo = min(margin_values) if margin_values else 0.0
    margin_hi = max(margin_values) if margin_values else 1.0

    for r in rows.values():
        audit_score = 1.0 if r['audit_flag'] else 0.0
        try:
            cons_score = float(r['consensus_conf']) if r['consensus_conf'] else 0.0
        except ValueError:
            cons_score = 0.0
        try:
            m = float(r['model_margin']) if r['model_margin'] else None
        except ValueError:
            m = None
        unc_score = (1.0 - _norm(m, margin_lo, margin_hi)) if m is not None else 0.0

        # Weight: audit is the most actionable signal (clear telemetry mismatch)
        priority = 0.45 * audit_score + 0.35 * cons_score + 0.20 * unc_score
        r['priority'] = round(priority, 4)

    # ── Sort and write ────────────────────────────────────────────────────
    sorted_rows = sorted(rows.values(), key=lambda r: -r['priority'])
    if args.top_n:
        sorted_rows = sorted_rows[:args.top_n]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    field_order = ['priority', 'key', 'seq_id', 'row_idx', 'source',
                   'intent_label', 'raw_label',
                   'audit_flag', 'audit_warns',
                   'consensus', 'consensus_voters', 'consensus_conf',
                   'model_margin']
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        w.writeheader()
        for r in sorted_rows:
            w.writerow({k: r.get(k, '') for k in field_order})

    print(f"\nWrote {len(sorted_rows)} review-queue rows -> {args.out}")
    if sorted_rows:
        print(f"  priority range: {sorted_rows[0]['priority']:.3f} (top) "
              f"-> {sorted_rows[-1]['priority']:.3f} (bottom)")
        # Class distribution of top-100
        from collections import Counter
        top = sorted_rows[:100]
        cls = Counter(r['intent_label'] for r in top)
        print("  top-100 class distribution:")
        for k, v in cls.most_common():
            print(f"    {k:25s} {v}")


if __name__ == '__main__':
    main()
