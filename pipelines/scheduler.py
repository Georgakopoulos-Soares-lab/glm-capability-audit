"""GPU-aware job runner.

Every guard here exists because its absence cost a run:

* **Admission on free MEMORY, not on process count or a marker.**  Busy-marker
  locking proved fragile -- a leftover subshell from a killed dispatcher deleted
  a marker it did not own, two 34 GB jobs landed on one card and a third OOMed.
  Free memory is the property we actually care about and it cannot go stale.
* **A stagger after each launch.**  A 944M model takes ~90 s to reserve its
  memory; probe again immediately and the next pick reads the same card as free.
  Five jobs once stacked on one GPU this way and the slowest crawled at 46
  min/epoch until it was moved.
* **Exit status captured immediately.**  `rc=$?` after an `echo` reports the
  echo's status, which is always 0.  Two crashed jobs were recorded DONE rc=0.
* **`CUDA_VISIBLE_DEVICES` exported into the child**, not prefixed onto a `cd`.
  `VAR=x cd a && python b` applies VAR to `cd`; every job silently ran on GPU 0.
* **Skip, never overwrite.**  A finished job is evidence; re-running it needs
  `--force`, and then the new output carries a distinct tag.
"""
from __future__ import annotations
import json, os, shlex, subprocess, sys, time
from dataclasses import dataclass, field

LOGS = os.environ.get("VB_LOGS", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"))


@dataclass
class Job:
    name: str
    cmd: list           # argv, already interpreter-resolved
    gpu: bool = True
    need_mib: int = 20000
    produces: str = ""   # a path; if it exists the job is skipped
    env: dict = field(default_factory=dict)
    stage: str = ""

    def done(self):
        return bool(self.produces) and os.path.exists(self.produces)

    def shell(self):
        return " ".join(shlex.quote(c) for c in self.cmd)


def gpu_free_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        free = {}
        for line in out.stdout.strip().splitlines():
            i, tot, used = (int(x) for x in line.split(","))
            free[i] = tot - used
        return free
    except Exception:
        return {}


def run(jobs, gpus=None, force=False, dry_run=False, stagger=120, poll=30, max_cpu=4):
    """Execute `jobs`, respecting GPU capacity.  Returns the list of failures."""
    os.makedirs(LOGS, exist_ok=True)
    gpus = gpus if gpus is not None else sorted(gpu_free_mib())
    todo, skipped = [], []
    for j in jobs:
        (skipped if (j.done() and not force) else todo).append(j)

    if skipped:
        print(f"  {len(skipped)} already complete, skipped "
              f"(--force to redo; output is never overwritten in place)")
    if dry_run:
        for j in todo:
            print(f"  [{j.stage or '-'}] {j.name}\n      {j.shell()}")
        print(f"\n  {len(todo)} job(s) would run, {len(skipped)} skipped")
        return []

    live, failures, queue = {}, [], list(todo)
    cpu_live = 0
    while queue or live:
        # reap
        for name, (p, g, t0, j) in list(live.items()):
            rc = p.poll()
            if rc is None:
                continue
            rc_now = rc                      # capture BEFORE anything else runs
            del live[name]
            if g is None:
                cpu_live -= 1
            mins = (time.time() - t0) / 60
            print(f"  {'OK  ' if rc_now == 0 else 'FAIL'} {name}  "
                  f"({mins:.0f} min, rc={rc_now}, {os.path.join(LOGS, name)}.log)")
            if rc_now != 0:
                failures.append((name, rc_now))

        launched = False
        for j in list(queue):
            if j.gpu:
                free = gpu_free_mib()
                busy = {g for _, (_, g, _, _) in live.items() if g is not None}
                pick = next((g for g in gpus
                             if g not in busy and free.get(g, 0) >= j.need_mib), None)
                if pick is None:
                    continue
            else:
                if cpu_live >= max_cpu:
                    continue
                pick = None

            queue.remove(j)
            env = dict(os.environ, **j.env)
            if pick is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(pick)
            log = open(os.path.join(LOGS, f"{j.name}.log"), "w")
            p = subprocess.Popen(j.cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            live[j.name] = (p, pick, time.time(), j)
            if pick is None:
                cpu_live += 1
            print(f"  START {'gpu%d' % pick if pick is not None else 'cpu '}  {j.name}")
            launched = True
            if pick is not None:
                time.sleep(stagger)   # let it actually reserve the memory
            break

        if not launched and (queue or live):
            time.sleep(poll)

    print(f"\n  {len(todo) - len(failures)}/{len(todo)} succeeded")
    for n, rc in failures:
        print(f"    FAILED {n} (rc={rc}) -- see {os.path.join(LOGS, n)}.log")
    return failures
