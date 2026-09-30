"""Import-check stub, NOT a deepspeed implementation.

GENA-LM's `modeling_bert.py` imports `deepspeed.ops.sparse_attention` inside
`if self.is_sparse:`. gena-lm-bert-base-t2t's config has no `is_sparse` key, so it defaults
to False and the import is never executed -- but transformers' `check_imports` scans the file
STATICALLY and refuses to load the module without deepspeed present.

Installing real deepspeed to satisfy a branch that never runs would pull a compiler toolchain
into the env for nothing. This stub satisfies `find_spec`. Touching any attribute raises, so
if a future config ever does set is_sparse the run fails loudly instead of silently degrading.
"""
__version__ = "0.0.0-stub"
