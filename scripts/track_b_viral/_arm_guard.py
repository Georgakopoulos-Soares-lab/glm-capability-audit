"""Refuse to score a capacity-sweep arm that is still being written.

`capacity_sweep.py` writes its per-seed prediction files DURING the run and its
summary json only at the end. So a glob over `*__cnn_ceiling__s*__preds.npz`
while a sweep is live returns a mixture: one seed from the current run and five
stale ones from whatever ran before. That mixture has no interpretation, and it
silently sets a floor.

This happened. During the transformer-overwrite incident the phage CNN arm was a
blend of a fresh seed 42 and five seeds from a superseded uncapped run, and two
audits scored against it before anyone looked at the timestamps.

The completion signal is the summary json: it is written last. So an arm counts
as usable only when its json exists and every prediction file predates it.
"""
import glob, os


def sweep_arm_ready(sweep_json, pred_glob, span_limit_s=12 * 3600):
    """(ready, reason). `span_limit_s` is the widest a genuine refit set may straddle."""
    preds = sorted(glob.glob(pred_glob))
    if not preds:
        return False, "no prediction files"
    if not os.path.exists(sweep_json):
        return False, (f"sweep still running -- {os.path.basename(sweep_json)} absent, so the "
                       f"{len(preds)} prediction file(s) are a mid-write mixture")
    # Predictions legitimately POSTDATE the json: `--only_selected` re-runs the
    # dev-selected cell at more seeds and deliberately does not rewrite the
    # ladder. So newness is not the signal -- INTERNAL INCONSISTENCY is. A real
    # refit set is written in one sitting; the contaminated phage arm had seed 42
    # from a live sweep and seeds 43-47 from a superseded run 33 h earlier.
    ts = [os.path.getmtime(p) for p in preds]
    span = max(ts) - min(ts)
    if span > span_limit_s:
        return False, (f"prediction files span {span/3600:.1f} h -- not one refit set but a "
                       f"mixture of runs; re-run rather than score it")
    return True, f"{len(preds)} seeds, span {span/60:.0f} min"
