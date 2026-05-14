"""
Auto-resolve low-priority entries in the review queue.

Reads `reports/review_queue.csv` (produced by build_review_queue.py) and
splits rows into three bins:

  AUTO_KEEP    - priority < keep-thresh AND no audit flag.
                 Safe to accept the original label without human review.
  AUTO_RELABEL - consensus_conf >= relabel-conf AND consensus differs
                 from original. Trust the weak-label consensus.
  NEEDS_REVIEW - everything else. Top-N is what a human should actually
                 look at.

Output: two CSVs.
  - resolved CSV (auto-decisions, columns: key, decision, new_label, ...)
  - residual CSV (rows still needing human review, sorted high priority first)

Usage:
  python scripts/auto_resolve_queue.py \\
      --queue   reports/review_queue.csv \\
      --resolved reports/auto_resolved.csv \\
      --residual reports/needs_review.csv

  # Tighter — only accept very-low-priority items, push more to review:
  python scripts/auto_resolve_queue.py \\
      --queue reports/review_queue.csv \\
      --resolved reports/auto_resolved.csv \\
      --residual reports/needs_review.csv \\
      --keep-thresh 0.30 --relabel-conf 0.90
"""

import sys, csv, argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))


def _float(s, default=0.0):
    try:
        return float(s) if s not in (None, '') else default
    except ValueError:
        return default


def _int(s, default=0):
    try:
        return int(s) if s not in (None, '') else default
    except ValueError:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue',    required=True,
                    help='Input review queue CSV from build_review_queue.py')
    ap.add_argument('--resolved', required=True,
                    help='Output CSV of auto-decisions')
    ap.add_argument('--residual', required=True,
                    help='Output CSV of items still requiring human review')
    ap.add_argument('--keep-thresh', type=float, default=0.40,
                    dest='keep_thresh',
                    help='Priority below this -> AUTO_KEEP (default 0.40)')
    ap.add_argument('--relabel-conf', type=float, default=0.85,
                    dest='relabel_conf',
                    help='Min consensus_conf for AUTO_RELABEL (default 0.85)')
    ap.add_argument('--relabel-voters', type=int, default=2,
                    dest='relabel_voters',
                    help='Min n_voters for AUTO_RELABEL (default 2)')
    ap.add_argument('--top-n', type=int, default=0,
                    help='Cap residual CSV at top-N priority (0=keep all)')
    args = ap.parse_args()

    queue_path = Path(args.queue)
    if not queue_path.exists():
        print(f"Queue file not found: {queue_path}")
        sys.exit(1)

    with open(queue_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    auto, residual = [], []
    decision_counts = Counter()
    relabel_flow = Counter()

    for r in rows:
        priority = _float(r.get('priority'))
        audit    = _int(r.get('audit_flag'))
        cons     = r.get('consensus', '').strip()
        cons_conf = _float(r.get('consensus_conf'))
        cons_voters = _int(r.get('consensus_voters'))
        original = r.get('intent_label', '')

        decision = None
        new_label = ''

        # AUTO_RELABEL takes precedence — high-confidence consensus is the
        # strongest signal we have when it disagrees with the original.
        if (cons and cons != original
                and cons_conf >= args.relabel_conf
                and cons_voters >= args.relabel_voters):
            decision = 'AUTO_RELABEL'
            new_label = cons
            relabel_flow[(original, cons)] += 1

        # AUTO_KEEP — low priority and no audit flag means nothing meaningful
        # is wrong with this entry.
        elif priority < args.keep_thresh and not audit:
            decision = 'AUTO_KEEP'

        if decision:
            decision_counts[decision] += 1
            auto.append({
                'key':         r.get('key', ''),
                'seq_id':      r.get('seq_id', ''),
                'row_idx':     r.get('row_idx', ''),
                'source':      r.get('source', ''),
                'decision':    decision,
                'original_label': original,
                'new_label':   new_label,
                'priority':    r.get('priority', ''),
                'consensus_conf': r.get('consensus_conf', ''),
            })
        else:
            decision_counts['NEEDS_REVIEW'] += 1
            residual.append(r)

    # Sort residual high priority first; optional truncation
    residual.sort(key=lambda r: -_float(r.get('priority')))
    if args.top_n:
        residual = residual[:args.top_n]

    Path(args.resolved).parent.mkdir(parents=True, exist_ok=True)
    Path(args.residual).parent.mkdir(parents=True, exist_ok=True)

    auto_fields = ['key', 'seq_id', 'row_idx', 'source', 'decision',
                   'original_label', 'new_label', 'priority', 'consensus_conf']
    with open(args.resolved, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=auto_fields)
        w.writeheader()
        for r in auto:
            w.writerow(r)

    if residual:
        with open(args.residual, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(residual[0].keys()))
            w.writeheader()
            for r in residual:
                w.writerow(r)
    else:
        # Still write an empty file with original headers
        with open(args.residual, 'w', newline='', encoding='utf-8') as f:
            f.write(','.join(rows[0].keys()) + '\n' if rows else '')

    total = len(rows)
    print(f"\nProcessed {total} queue entries")
    for k, v in decision_counts.most_common():
        pct = 100.0 * v / max(1, total)
        print(f"  {k:15s} {v:6d}  ({pct:5.1f}%)")
    if relabel_flow:
        print("\nAUTO_RELABEL flow:")
        for (a, b), c in sorted(relabel_flow.items(), key=lambda x: -x[1]):
            print(f"  {a:25s} -> {b:25s}  {c}")
    print(f"\n  Auto-decisions saved -> {args.resolved}")
    print(f"  Residual review queue -> {args.residual} ({len(residual)} rows)")


if __name__ == '__main__':
    main()
