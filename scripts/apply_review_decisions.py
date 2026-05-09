"""
Auto-apply consensus disagreements to clean the training data.

Rules (rank-ordered):

  1. KEEP   — sequence is in agreement with weak labelers, OR labelers abstain.
  2. RELABEL — consensus disagreement with confidence >= conf-thresh AND
               n_voters >= voter-thresh. New label = consensus_label.
  3. DROP   — audit flagged AND no consensus support. Likely noisy.
  4. DOWNWEIGHT — covered by `--use-weak-weights` at training time, no action here.

This is the no-human-in-the-loop fast path. For higher precision, hand-review
`reports/review_queue.csv` first and use a manual decision CSV instead.

Usage:
  python scripts/apply_review_decisions.py \\
      --in   data/hdd_train.json \\
      --out  data/hdd_cleaned.json \\
      --audit reports/audit_hdd_train.csv \\
      --disagreements reports/consensus_disagreements.csv

  # Stricter:
  python scripts/apply_review_decisions.py \\
      --in data/hdd_train.json --out data/hdd_cleaned.json \\
      --audit reports/audit_hdd_train.csv \\
      --disagreements reports/consensus_disagreements.csv \\
      --conf-thresh 0.90 --voter-thresh 2
"""

import sys, json, csv, argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_csv(path):
    if not path or not Path(path).exists():
        return []
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in',  dest='input',  required=True)
    ap.add_argument('--out', dest='output', required=True)
    ap.add_argument('--audit',          help='audit CSV — used only for DROP decisions')
    ap.add_argument('--disagreements', required=True,
                    help='consensus disagreements CSV from consensus_label.py')
    ap.add_argument('--conf-thresh',  type=float, default=0.85,
                    help='min consensus_conf to RELABEL (default 0.85)')
    ap.add_argument('--voter-thresh', type=int, default=1,
                    help='min n_voters to RELABEL (default 1)')
    ap.add_argument('--drop-on-audit-and-no-consensus', action='store_true',
                    default=True,
                    help='Drop sequences flagged by audit with no consensus support')
    ap.add_argument('--no-drop', action='store_true',
                    help='Disable DROP rule — only relabel, never drop')
    ap.add_argument('--protect-labels', nargs='*',
                    default=['pedestrian_monitor'],
                    help='Original labels that must NOT be relabeled '
                         '(default: pedestrian_monitor — telemetry alone '
                         'cannot disprove pedestrian observation)')
    args = ap.parse_args()
    protect_set = set(args.protect_labels or [])

    # Build lookup: (source, seq_id, idx) -> consensus row
    relabel_map = {}
    for r in _load_csv(args.disagreements):
        try:
            conf = float(r.get('consensus_conf', '0') or '0')
            n_voters = int(r.get('n_voters', '0') or '0')
        except ValueError:
            continue
        if conf < args.conf_thresh or n_voters < args.voter_thresh:
            continue
        key = (r.get('source', ''), r.get('seq_id', ''), r.get('row_idx', ''))
        relabel_map[key] = r.get('consensus', '')

    # Build drop set: audit-flagged AND not in relabel_map
    drop_set = set()
    if not args.no_drop:
        for r in _load_csv(args.audit):
            key = (r.get('source', ''), r.get('seq_id', ''), r.get('idx', ''))
            if key not in relabel_map:
                drop_set.add(key)

    # Load + apply
    data = json.load(open(args.input, encoding='utf-8'))
    src_name = Path(args.input).name

    out = []
    n_relabel = n_drop = n_keep = 0
    relabel_flow = Counter()
    dropped_by_orig = Counter()

    for idx, entry in enumerate(data):
        key = (src_name, entry.get('seq_id', ''), str(idx))
        original = entry.get('intent_label', '')

        if key in relabel_map and original not in protect_set:
            new_label = relabel_map[key]
            entry = dict(entry)
            entry['intent_label']      = new_label
            entry['original_label']    = original
            entry['relabel_reason']    = 'consensus_auto'
            relabel_flow[(original, new_label)] += 1
            out.append(entry)
            n_relabel += 1
        elif key in drop_set:
            dropped_by_orig[original] += 1
            n_drop += 1
        else:
            out.append(entry)
            n_keep += 1

    # Write
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f)

    print(f"\nWrote {len(out)} sequences -> {args.output}")
    print(f"  kept     : {n_keep}")
    print(f"  relabeled: {n_relabel}")
    print(f"  dropped  : {n_drop}")
    if relabel_flow:
        print("\nRelabel flow:")
        for (a, b), c in sorted(relabel_flow.items(), key=lambda x: -x[1]):
            print(f"  {a:25s} -> {b:25s}  {c}")
    if dropped_by_orig:
        print("\nDropped (by original label):")
        for k, v in dropped_by_orig.most_common():
            print(f"  {k:25s} {v}")

    # Final class balance
    final = Counter(s.get('intent_label', '') for s in out)
    print("\nFinal class distribution:")
    for k, v in sorted(final.items(), key=lambda x: -x[1]):
        print(f"  {k:25s} {v}")


if __name__ == '__main__':
    main()
