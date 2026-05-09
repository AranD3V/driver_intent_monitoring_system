"""
Consensus pseudo-labeling pipeline.

Three modes:

  --mode validate  (default for *labeled* data)
      Keep every sequence and its original label. Annotate each with weak-
      labeler votes and a derived `label_confidence` for sample weighting.
      Use for improving an *already-labeled* training set without throwing
      sequences away.

  --mode relabel
      Apply two-of-N consensus and OVERRIDE original labels where the weak
      labelers disagree. Sequences with no consensus are dropped.

  --mode augment
      Treat input as unlabeled (session logs etc). Window into seq_len
      windows, run labelers, keep only windows that pass `min_voters`.

Outputs:
  out_json                     -- training-ready JSON with `weak_label_meta`
  reports/consensus_disagreements.csv  -- original vs. consensus mismatches
  reports/consensus_abstained.csv      -- sequences with no consensus

Usage:
  # Annotate existing HDD labels with confidence
  python scripts/consensus_label.py --in data/hdd_train.json \\
      --out data/hdd_validated.json --mode validate

  # Aggressive re-labeling (only keeps sequences with strong consensus)
  python scripts/consensus_label.py --in data/hdd_train.json \\
      --out data/hdd_relabeled.json --mode relabel --min-voters 2

  # Mine pseudo-labels from session logs
  python scripts/consensus_label.py --in logs/*.json \\
      --out data/pseudo_logs.json --mode augment --min-voters 2
"""

import sys, json, csv, glob, argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.weak_labelers import run_all, consensus


def _load_labeled(paths):
    for p in paths:
        try:
            data = json.load(open(p, encoding='utf-8'))
        except Exception as e:
            print(f"[!] {p}: {e}")
            continue
        if isinstance(data, list):
            for idx, entry in enumerate(data):
                # Stash the per-file row index so downstream tools can build
                # unique keys (HDD seq_id is per-session, not per-window).
                entry.setdefault('_row_idx', idx)
                yield p, entry


def _iter_session_log_windows(path: str, seq_len: int = 90, stride: int = 45):
    """Convert a flat session log into pseudo-sequences."""
    try:
        data = json.load(open(path, encoding='utf-8'))
    except Exception as e:
        print(f"[!] {path}: {e}"); return
    if not isinstance(data, list) or not data:
        return
    if 'frames' in data[0]:
        for entry in data:
            yield path, entry
        return

    for start in range(0, len(data) - seq_len + 1, stride):
        chunk = data[start:start + seq_len]
        yield path, {
            'seq_id':    f"{Path(path).stem}_w{start:06d}",
            'source':    'session_log',
            'raw_label': '',
            'frames':    chunk,
        }


def _annotate(seq: dict, votes: dict, cons_label, cons_conf, voters):
    """Attach weak_label_meta to a copy of seq."""
    out = dict(seq)
    out.pop('_row_idx', None)  # don't leak internal bookkeeping into output
    out['weak_label_meta'] = {
        'consensus_label':      cons_label,
        'consensus_confidence': round(cons_conf, 3) if cons_conf else 0.0,
        'voters':               voters,
        'n_voters':             len(voters),
        'original_label':       seq.get('intent_label', '') or '',
        'all_votes': {n: {'label': lbl, 'conf': round(c, 3)}
                      for n, (lbl, c) in votes.items()},
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in',  dest='inputs',   nargs='+', required=True)
    ap.add_argument('--out', dest='out_json', required=True)
    ap.add_argument('--mode', choices=('validate', 'relabel', 'augment'),
                    default='validate')
    ap.add_argument('--min-voters', type=int, default=1)
    ap.add_argument('--seq-len',    type=int, default=90)
    ap.add_argument('--stride',     type=int, default=45)
    ap.add_argument('--disagreements', default='reports/consensus_disagreements.csv')
    ap.add_argument('--no-consensus',  default='reports/consensus_abstained.csv')
    args = ap.parse_args()

    paths = []
    for pat in args.inputs:
        m = sorted(glob.glob(pat))
        paths.extend(m if m else [pat])
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        print("No matching files."); sys.exit(1)

    if args.mode == 'augment':
        iterator = (item
                    for p in paths
                    for item in _iter_session_log_windows(p, args.seq_len, args.stride))
    else:
        iterator = _load_labeled(paths)

    out_seqs       = []
    disagreements  = []
    abstained      = []

    confirmed = relabeled = new_label = no_label = 0
    voter_freq        = Counter()
    consensus_classes = Counter()
    original_classes  = Counter()
    n_voter_hist      = Counter()

    for path, seq in iterator:
        original = seq.get('intent_label', '') or ''
        original_classes[original or '(none)'] += 1

        votes = run_all(seq)
        cons_label, cons_conf, voters = consensus(votes, min_voters=1)
        n_voter_hist[len(voters)] += 1

        for name, (lbl, _) in votes.items():
            if lbl is not None:
                voter_freq[name] += 1

        # ── Mode: VALIDATE — keep every sequence, annotate ──────────────
        if args.mode == 'validate':
            entry = _annotate(seq, votes, cons_label, cons_conf, voters)
            # If consensus disagrees with original AND meets min_voters, log it
            if cons_label and original and cons_label != original \
                    and len(voters) >= args.min_voters:
                relabeled += 1
                disagreements.append({
                    'source':         Path(path).name,
                    'seq_id':         seq.get('seq_id', ''),
                    'row_idx':        seq.get('_row_idx', ''),
                    'raw_label':      seq.get('raw_label', ''),
                    'original':       original,
                    'consensus':      cons_label,
                    'consensus_conf': f"{cons_conf:.3f}",
                    'n_voters':       len(voters),
                    'voters':         ','.join(voters),
                })
            elif cons_label == original:
                confirmed += 1
            out_seqs.append(entry)
            continue

        # ── Modes: RELABEL / AUGMENT — must hit min_voters ─────────────
        if cons_label is None or len(voters) < args.min_voters:
            no_label += 1
            abstained.append({
                'source':    Path(path).name,
                'seq_id':    seq.get('seq_id', ''),
                'raw_label': seq.get('raw_label', ''),
                'original':  original,
                'votes':     ' | '.join(f"{n}={lbl}@{c:.2f}" if lbl else f"{n}=ABSTAIN"
                                        for n, (lbl, c) in votes.items()),
            })
            continue

        consensus_classes[cons_label] += 1
        entry = _annotate(seq, votes, cons_label, cons_conf, voters)
        entry['intent_label'] = cons_label
        out_seqs.append(entry)

        if not original:
            new_label += 1
        elif original == cons_label:
            confirmed += 1
        else:
            relabeled += 1
            disagreements.append({
                'source':         Path(path).name,
                'seq_id':         seq.get('seq_id', ''),
                'raw_label':      seq.get('raw_label', ''),
                'original':       original,
                'consensus':      cons_label,
                'consensus_conf': f"{cons_conf:.3f}",
                'n_voters':       len(voters),
                'voters':         ','.join(voters),
            })

    # ── Write outputs ──────────────────────────────────────────────────────
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(out_seqs, f)
    print(f"\nWrote {len(out_seqs)} sequences -> {args.out_json}")

    def _csv(rows, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            print(f"  (no rows for {path})"); return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"  {len(rows)} rows -> {path}")

    _csv(disagreements, args.disagreements)
    _csv(abstained,     args.no_consensus)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n=== Summary [mode={args.mode}, min_voters={args.min_voters}] ===")
    print(f"  confirmed       = {confirmed}")
    print(f"  disagreements   = {relabeled}")
    print(f"  newly labeled   = {new_label}")
    print(f"  no consensus    = {no_label}")
    print("\nVoter participation:")
    for n, c in voter_freq.most_common():
        print(f"  {n:15s} {c}")
    print("\nVoter-count histogram:")
    for k in sorted(n_voter_hist):
        print(f"  {k} voters: {n_voter_hist[k]}")
    if args.mode != 'validate':
        print("\nConsensus label distribution:")
        for k, v in consensus_classes.most_common():
            print(f"  {k:25s} {v}")
    print("\nOriginal label distribution:")
    for k, v in original_classes.most_common():
        print(f"  {k:25s} {v}")


if __name__ == '__main__':
    main()
