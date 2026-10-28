#!/usr/bin/env python
"""The single entry point for every experiment in the paper.

    python pipelines/run.py <stage>... [--tasks ...] [--models ...] [flags]

Stages run in dependency order; each one is also a legal standalone target.

    homology    MMseqs2 train-vs-test search -> the NN-lookup arm
    baselines   k-mer, k-mer GBT, aa composition, taxonomy prior, CNN and
                from-scratch transformer ladders
    finetune    full fine-tuning, identical budget per model
    hidden      nested homology audit, nucleotide then six-frame -> hasN/hasT,
                which define L1 and L2
    layered     graded relatedness curve and group bootstrap

Every stage here feeds a reported number. Analyses that were run and not
reported live under scripts/_not_in_manuscript/ and are not wired to a stage.

Examples

    python pipelines/run.py --list                       # what is configured
    python pipelines/run.py baselines --dry-run          # print the commands
    python pipelines/run.py finetune --tasks gue_phage --models nt_v2_500m
    python pipelines/run.py homology hidden --tasks paper
    python pipelines/run.py all --gpus 0,1,2,3

Nothing is ever overwritten: a stage whose output exists is skipped unless
--force, and --force writes under a distinct tag.
"""
from __future__ import annotations
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from registry import Registry                     # noqa: E402
from scheduler import Job, run as run_jobs        # noqa: E402

S = os.path.join(REPO, "scripts")
# These must match scripts/common/paths.py exactly. When the pipeline resolved
# outputs independently of the scripts, it wrote predictions into one tree while
# the scripts globbed another, and a baseline arm read as simply absent.
RESULTS = os.environ.get("VB_VIRAL_OUT", os.path.join(REPO, "results", "viral_bench"))
PREDS = os.environ.get("VB_PREDS", os.path.join(RESULTS, "preds_rescore"))
MASKS = os.path.join(RESULTS, "regime_masks")
REPORTS = os.environ.get("VB_REPORTS", os.path.join(REPO, "reports"))
STAGES = ["homology", "baselines", "finetune", "hidden", "layered"]


# --------------------------------------------------------------------------- #
# stage builders: each returns a list of Job
# --------------------------------------------------------------------------- #
def st_homology(R, tasks, models, a):
    jobs = []
    for t in tasks:
        if not t.dir:
            continue          # virobench builds its splits in-script
        jobs.append(Job(
            name=f"homology__{t.key}", gpu=False, stage="homology",
            produces=os.path.join(PREDS, f"hvue_v2__{t.kmer_tag}__nnlookup__s0__preds.npz"),
            cmd=[R.python(), "-u", f"{S}/track_b_viral/nn_lookup_as_baseline.py",
                 "--train-csv", t.csv("train"), "--test-csv", t.csv("test"),
                 "--name", t.name, "--esc", t.kmer_tag,
                 "--threads", str(a.threads), "--predictions-dir", PREDS]))
    return jobs


def st_baselines(R, tasks, models, a):
    P, jobs = R.protocol["baselines"], []
    for t in tasks:
        FILE_TAG = {"kmer": "kmer", "kmer_gbt": "kmergbt", "aa_gbt": "aagbt",
                    "tax_prior": "taxprior"}
        for arm in ("kmer", "kmer_gbt", "aa_gbt", "tax_prior"):
            spec = R.baselines[arm]
            # The taxonomy prior needs a lineage label, which 8 of 9 benchmarks do
            # not ship. That is a finding, not a gap -- see reports/RESULTS.md S4.
            if spec.get("requires") == "taxonomy" and not (t.tax_train and t.tax_test):
                continue
            cmd = [R.python(), "-u", f"{S}/{spec['script']}"]
            if arm == "tax_prior":
                cmd += ["--train-tax", t.tax_train, "--test-tax", t.tax_test,
                        "--esc", t.kmer_tag, "--name", t.name]
            elif arm == "aa_gbt":
                cmd += ["--train-csv", t.csv("train"), "--test-csv", t.csv("test"),
                        "--esc", t.kmer_tag, "--name", t.name]
                if t.train_cap:
                    cmd += ["--train-cap", str(t.train_cap)]
            else:
                cmd += ["--task", t.task_arg or t.esc]
                if t.loader == "geneb":
                    cmd += ["--csvdir", t.dir]
                if t.train_cap:
                    cmd += ["--train-cap", str(t.train_cap)]
            cmd += ["--predictions-dir", PREDS]
            jobs.append(Job(name=f"{arm}__{t.key}", gpu=False, stage="baselines", cmd=cmd,
                            produces=os.path.join(
                                PREDS, f"hvue_v2__{t.kmer_tag}__{FILE_TAG[arm]}__s0__preds.npz")))
        # capacity ladders -- the CNN and the from-scratch transformer.
        # arch_family MUST reach the output name: it did not, and the transformer
        # ladder overwrote the CNN's json and all six prediction files on two
        # tasks before the collision was noticed.
        for arm in ("cnn", "transformer"):
            spec = R.baselines[arm]
            suffix = "" if arm == "cnn" else "__transformer"
            cmd = [R.python(), "-u", f"{S}/{spec['script']}",
                   "--dataset", t.loader, "--task", t.task_arg or t.esc,
                   "--class_weight", P["class_weight"],
                   "--arch_family", spec["arch_family"],
                   "--seeds", *[str(s) for s in P["ladder_seeds"]],
                   "--refit_seeds", str(P["refit_seeds"]),
                   "--topk", str(P["topk"]), "--predictions_dir", PREDS]
            if t.train_cap:
                cmd += ["--train_cap", str(t.train_cap)]
            if t.level:
                cmd += ["--level", t.level]
            if t.seqlen:
                cmd += ["--seqlen", str(t.seqlen)]
            if arm == "transformer":
                cmd += ["--kernels", *[str(k) for k in P["transformer_kernels"]],
                        "--lrs", *[str(l) for l in P["transformer_lrs"]]]
            env = {"CSVDIR": t.dir} if t.loader == "csvdir" else {}
            jobs.append(Job(name=f"{arm}__{t.key}", stage="baselines", cmd=cmd, env=env,
                            need_mib=a.need_mib,
                            produces=os.path.join(RESULTS, "capacity_sweep",
                                                  f"{t.sweep_tag}{suffix}.json")))
    return jobs


def st_finetune(R, tasks, models, a):
    P, jobs = R.protocol["finetune"], []
    for t in tasks:
        for m in models:
            if not (m.runs_on(t.key) and m.runs_in("finetune")):
                continue
            ep = P[t.epochs_key]
            for lr in P["lrs"]:
                cmd = [R.python(), "-u", f"{S}/track_a_benchmarks/geneb_finetune.py",
                       "--source", "geneb" if t.loader == "geneb" else t.loader,
                       "--task", t.task_arg or t.esc, "--model", m.key,
                       "--lr", f"{lr:g}",
                       "--seeds", *[str(s) for s in P["seeds"]], "--epochs", str(ep)]
                if t.train_cap:
                    cmd += ["--cap_train", str(t.train_cap)]
                if P.get("class_weight"):
                    cmd += ["--class_weight", "balanced", "--tag_suffix", "__cwbal"]
                jobs.append(Job(
                    name=f"ft__{t.key}__{m.key}__lr{lr:g}", stage="finetune", cmd=cmd,
                    need_mib=a.need_mib,
                    produces=os.path.join(RESULTS, "geneb_finetune",
                                          f"{t.esc}__{m.key}__lr{lr:g}__fullft.json")))
    return jobs


def st_hidden(R, tasks, models, a):
    """Nested homology audit: nucleotide -> translated (six-frame) -> protein.

    The stage that matters most. A nucleotide-only screen calls 65-97% of these
    test sets novel when translated search says otherwise, so every downstream
    novelty claim depends on running this.
    """
    jobs = []
    for t in tasks:
        if not t.dir:
            continue
        cmd = [R.python(), "-u", f"{S}/track_b_viral/hidden_homology_audit.py",
               "--name", t.name, "--train-csv", t.csv("train"), "--test-csv", t.csv("test"),
               "--esc", t.esc, "--threads", str(a.threads), "--reps", "2000",
               "--cache", os.path.join(MASKS, f"{t.name}.npz")]
        if t.kmer_esc:
            cmd += ["--kmer-esc", t.kmer_esc]
        if t.sweep_tag:
            cmd += ["--sweep-tag", t.sweep_tag]
        if t.cluster_csv:
            cmd += ["--cluster-csv", t.cluster_csv]
        if t.tax_train and t.tax_test:
            cmd += ["--tax-train", t.tax_train, "--tax-test", t.tax_test]
        jobs.append(Job(name=f"hidden__{t.key}", gpu=False, stage="hidden", cmd=cmd,
                        produces=os.path.join(REPORTS, "hidden_homology",
                                              f"{t.name}__hidden_homology.json")))
    return jobs


def st_layered(R, tasks, models, a):
    """Layered-novelty figure, graded relatedness curve, group bootstrap, equivalence.

    --require-arms is deliberate: this fails loudly rather than scoring against a
    ladder arm that is still being written.
    """
    jobs = []
    for t in tasks:
        if not t.dir:
            continue
        cmd = [R.python(), "-u", f"{S}/track_b_viral/layered_novelty_figure.py",
               "--name", t.name, "--test-csv", t.csv("test"), "--esc", t.esc,
               "--masks", os.path.join(MASKS, f"{t.name}.npz"),
               "--hits-npz", os.path.join(MASKS, f"{t.name}.npz"),
               "--reps", "5000", "--require-arms", "--focus", t.focus_model or "hvilm"]
        if t.kmer_esc:
            cmd += ["--kmer-esc", t.kmer_esc]
        if t.sweep_tag:
            cmd += ["--sweep-tag", t.sweep_tag]
        if t.cluster_csv:
            cmd += ["--units-csv", t.cluster_csv, "--unit-col", "cluster"]
        if t.tax_train and t.tax_test:
            cmd += ["--tax-train", t.tax_train, "--tax-test", t.tax_test]
        jobs.append(Job(name=f"layered__{t.key}", gpu=False, stage="layered", cmd=cmd,
                        produces=os.path.join(REPORTS, "layered", f"{t.name}__layered.json")))
    return jobs


BUILDERS = {"homology": st_homology, "baselines": st_baselines, "finetune": st_finetune,
            "hidden": st_hidden, "layered": st_layered}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="*", help=f"one or more of: {', '.join(STAGES)}, or 'all'")
    ap.add_argument("--tasks", default="paper",
                    help="comma list of task keys, or 'paper' (default) / 'all'")
    ap.add_argument("--models", default="all", help="comma list of model keys, or 'all'")
    ap.add_argument("--gpus", default="", help="e.g. 0,1,2 (default: every visible GPU)")
    ap.add_argument("--need-mib", type=int, default=20000,
                    help="free memory a GPU must have before a job is placed on it")
    ap.add_argument("--threads", type=int, default=16, help="MMseqs2 threads")
    ap.add_argument("--stagger", type=int, default=120,
                    help="seconds between GPU launches; a 944M model needs ~90s to "
                         "reserve memory, and probing sooner stacks jobs on one card")
    ap.add_argument("--force", action="store_true",
                    help="redo stages whose output already exists (never overwrites in place)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    ap.add_argument("--list", action="store_true", help="show the configuration and exit")
    a = ap.parse_args()

    R = Registry()
    if a.list or not a.stages:
        import subprocess
        subprocess.run([sys.executable, os.path.join(HERE, "registry.py")])
        print(f"\nmodels: {', '.join(R.models)}")
        print(f"stages: {', '.join(STAGES)}")
        print("\nnothing to do -- name at least one stage (or 'all')")
        return 0

    stages = STAGES if "all" in a.stages else a.stages
    bad = [s for s in stages if s not in BUILDERS]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; known: {', '.join(STAGES)}")
    stages = [s for s in STAGES if s in stages]          # dependency order

    tasks = R.select_tasks(a.tasks)
    models = R.select_models(a.models)
    gpus = [int(g) for g in a.gpus.split(",")] if a.gpus else None

    print(f"tasks:  {', '.join(t.key for t in tasks)}")
    print(f"models: {', '.join(m.key for m in models)}")
    print(f"stages: {' -> '.join(stages)}\n")

    jobs = []
    for s in stages:
        js = BUILDERS[s](R, tasks, models, a)
        print(f"  {s:11s} {len(js):3d} job(s)")
        jobs += js
    print()
    fails = run_jobs(jobs, gpus=gpus, force=a.force, dry_run=a.dry_run, stagger=a.stagger)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
