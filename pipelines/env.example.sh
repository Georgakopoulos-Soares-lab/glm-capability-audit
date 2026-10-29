# Copy to env.sh, edit, `source env.sh`.  Every path the pipeline and the analysis
# layer use resolves through these.  Individual scripts under scripts/ carry
# absolute fallbacks from the cluster this work ran on; exporting these overrides
# all of them, so set them before running anything.

# --- where things live --------------------------------------------------------
export VB_DATA=$PWD/data                   # parent of hvue_v2/ vhdb_tropism/ gue_devcarve/ ...
# The two GUE benchmarks ship train.csv/test.csv with no dev split, so they load
# through the same reader as GENEB and resolve under $VB_GENEB_DIR, not $VB_DATA.
# GENEB itself is not a benchmark in this study; only the reader is shared.
export VB_GENEB_DIR=$VB_DATA/gue_devcarve
export VB_OUT=$PWD/results                 # run outputs: json summaries + .npz predictions
export VB_VIRAL_OUT=$VB_OUT/viral_bench    # large regenerable outputs: ladders, predictions, masks
export VB_PREDS=$VB_VIRAL_OUT/preds_rescore
export VB_RESULTS=$VB_VIRAL_OUT            # the tree the analysis layer reads
export VB_REPORTS=$PWD/reports             # hidden_homology/, layered/, tables/
export VB_SCRATCH=/tmp/$USER/vb            # mmseqs temp; needs tens of GB
export VB_LOGS=$PWD/logs                   # one .log per job, named after the job
export HF_HOME=$VB_DATA/hf_cache
export VB_MMSEQS=$(command -v mmseqs)

# --- interpreter ----------------------------------------------------------------
# One environment runs everything.  Build it from environment.yml and point this
# at its python, or leave it unset to use whichever interpreter launches run.py.
export VB_PYTHON=$(command -v python)
