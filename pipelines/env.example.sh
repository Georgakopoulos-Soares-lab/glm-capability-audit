# Copy to env.sh, edit, `source env.sh`.  Nothing in this repository hard-codes a
# machine path; everything resolves through these, with repo-relative defaults in
# scripts/common/paths.py.

# --- where things live --------------------------------------------------------
export VB_DATA=/data/nvidia                # parent of hvue_v2/ vhdb_tropism/ gue_devcarve/ ...
export VB_OUT=$PWD/results                 # run outputs: json summaries + .npz predictions
export VB_VIRAL_OUT=$VB_OUT/viral_bench    # large regenerable outputs: ladders, predictions, masks
export VB_PREDS=$VB_VIRAL_OUT/preds_rescore
export VB_REPORTS=$PWD/reports             # regime/, frozen_probe/, trend/, the final tables
export VB_SCRATCH=/tmp/$USER/vb            # mmseqs temp; needs tens of GB
export VB_LOGS=$PWD/logs                   # one .log per job, named after the job
export HF_HOME=/data/nvidia/hf_cache
export VB_MMSEQS=$(command -v mmseqs)

# --- interpreters -------------------------------------------------------------
# These models need mutually incompatible dependency sets.  Splitting them is not
# fastidiousness: runs have died on a missing pandas, a missing transformers and a
# missing evo2, each time silently until the log was read.  pipelines/run.py
# refuses to launch if one of these does not exist.
export VB_ENV_DNABERT=$HOME/miniconda3/envs/dnabert/bin/python    # transformers + torch
export VB_ENV_BIOJEPA=$HOME/miniconda3/envs/biojepa/bin/python    # pandas/sklearn/xgboost
export VB_ENV_EVO_7B=/data/nvidia/envs/evo_7b/bin/python          # evo2 + turboquant-bio
export VB_ENV_EVO=$HOME/miniconda3/envs/evo/bin/python            # evo 1
export VB_ENV_GENERATOR=$HOME/miniconda3/envs/generator/bin/python

# --- model weights not on the hub ---------------------------------------------
export VB_LUCAVIRUS_DIR=/data/nvidia/models/lucavirus_step38M/207da0192381a3cedcb1cf82a9850239d5fdd9d5
