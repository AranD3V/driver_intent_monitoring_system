# Day-of checklist — one-shot Colab training

Total time: ~4 hrs. Most of it is Colab running unattended.

## 0. Confirm overnight push (1 min)

The following files must be on `origin/main` of the GitHub repo. Auto-push hook should have handled it; verify at https://github.com/AranD3V/driver_intent_monitoring_system

- `scripts/audit_labels.py`
- `scripts/consensus_label.py`
- `scripts/build_review_queue.py`
- `scripts/apply_review_decisions.py`
- `modules/weak_labelers.py`
- `modules/feature_augment.py`
- `requirements-colab.txt`
- `notebooks/colab_train.ipynb`
- modified `scripts/train_intent.py` (with `--weak-aug` and `--use-weak-weights` flags)

## 1. Upload data to Drive (5 min)

Create folder `MyDrive/intent_data/` and upload these from `D:/TW/Final_Project/data/`:

- `hdd_train.json`              (~70 MB) — vanilla baseline
- `hdd_cleaned_validated.json`  (~60 MB) — cleaned + weak-meta annotated

That's it — only two files. Don't upload the rest.

(Optional) create empty folders `MyDrive/intent_models/` and `MyDrive/intent_reports/`. The notebook creates them if missing, but pre-creating means the symlinks succeed on first try.

## 2. Open Colab and runtime (2 min)

1. Open https://colab.research.google.com
2. File → Open notebook → GitHub tab → paste repo URL `AranD3V/driver_intent_monitoring_system` → select `notebooks/colab_train.ipynb`
3. Runtime → Change runtime type → **T4 GPU** → Save
4. (Hidden tip) Runtime → Manage sessions to confirm you got a GPU not a TPU.

## 3. Run the cells (~3.5 hrs unattended)

Cell-by-cell, top to bottom. The cells:

| Cell | What it does | Time |
|------|--------------|------|
| 1 | Mount Drive (asks for OAuth) | 30 s |
| 2 | Clone repo | 10 s |
| 3 | `pip install -r requirements-colab.txt` | 1-2 min |
| 4 | Symlink data/models/reports to Drive | 5 s |
| 5 | GPU + data sanity check | 5 s |
| 6 | **Run A** — vanilla baseline kfold | ~90 min |
| 7 | **Run B** — weak pipeline kfold | ~90 min |
| 8 | Eval baseline on hdd_train | ~3 min |
| 9 | Eval weak on hdd_train | ~3 min |
| 10 | Pick winner, copy to `intent_canonical.pth` | 1 s |
| 11 | Print summary table | 1 s |

**Anti-disconnect tips:**
- Keep the browser tab open. Don't close the laptop lid.
- Free Colab idle-disconnects after 90 min of inactivity, but a running cell counts as activity.
- If it disconnects mid-run B: re-mount Drive, re-symlink data/models, re-run from cell 7. The baseline (cell 6) checkpoint survives in Drive.

## 4. Pull artifacts back (5 min)

Everything is already in Drive (because of the symlinks). Specifically:

- `MyDrive/intent_models/baseline.pth` + `_latest.pth` + per-fold checkpoints
- `MyDrive/intent_models/weak.pth` + per-fold checkpoints
- `MyDrive/intent_models/intent_canonical.pth`
- `MyDrive/intent_models/confusion_matrix*.png`
- `MyDrive/intent_reports/eval_baseline.txt`
- `MyDrive/intent_reports/eval_weak.txt`

Download the `.pth` files and `.txt` files locally for the writeup.

## 5. Writeup template (15 min)

After both evals print, you'll have:

```
=========================================================
Run                       Eval acc        Delta
=========================================================
baseline                       XX.XX
+ weak pipeline                XX.XX        +X.XX
=========================================================
```

Open `README.md` (or wherever your project writeup lives), drop in:

- The number table.
- A line about the pipeline: "Audit + 3-labeler consensus + auto-relabel of high-confidence Background mislabels (n=275) + weak-confidence sample weighting + telemetry-aware augmentation."
- The two confusion matrices side-by-side.
- One paragraph caveat: "Validation split is window-level; a session-level split would likely lower the headline number by 3-5pts. Future work: driver-level split + time-to-detection metric."

That's the project. Done.

## If things go wrong

| Symptom | Fix |
|---------|-----|
| `mediapipe`/`carla`/`metadrive` install error | Make sure cell 3 uses `requirements-colab.txt` not `requirements.txt` |
| `RuntimeError: ... train_intent.py` import fail | The `_jitter` etc. are in train_intent — check git pulled latest |
| `No GPU available` | Runtime → Change runtime type → T4 GPU. If pool exhausted, wait 20 min, try again |
| Run A finishes < 50% acc | Sanity issue — paste the last 30 lines into Claude and we'll debug |
| Run B *worse* than Run A | Weak pipeline made things worse. Likely the `--use-weak-weights` was too aggressive on the 165-sample `normal_forward` class. Re-run B without that flag, keep `--weak-aug` only |

---

**Estimate of final number:** baseline ~58-62% val_acc, weak pipeline +3 to +6 pts. Anything bigger is suspicious; anything smaller means cleanup didn't help and we'd want a follow-up day.
