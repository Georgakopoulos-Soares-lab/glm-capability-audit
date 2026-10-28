# Copy to env.sh, edit, `source env.sh`.  Nothing in this repository hard-codes a
# machine path; everything resolves through these, with repo-relative defaults in
# scripts/common/paths.py.

# --- where things live --------------------------------------------------------
export VB_DATA=$PWD/data                   # parent of hvue_v2/ vhdb_tropism/ gue_devcarve/ ...
export VB_OUT=$PWD/results                 # run outputs: json summaries + .npz predictions
export VB_VIRAL_OUT=$VB_OUT/viral_bench    # large regenerable outputs: ladders, predictions, masks
export VB_PREDS=$VB_VIRAL_OUT/preds_rescore
export VB_REPORTS=$PWD/reports             # hidden_homology/, layered/, tables/
export VB_SCRATCH=/tmp/$USER/vb            # mmseqs temp; needs tens of GB
export VB_LOGS=$PWD/logs                   # one .log per job, named after the job
export HF_HOME=$VB_DATA/hf_cache
export VB_MMSEQS=$(command -v mmseqs)

# --- interpreter ----------------------------------------------------------------
# One environment runs everything.  Build it from environment.yml and point this
# at its python, or leave it unset to use whichever interpreter launches run.py.
export VB_PYTHON=$(command -v python)
