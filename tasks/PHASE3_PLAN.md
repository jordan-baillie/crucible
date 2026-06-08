# Phase 3 — Multi-agent scale-out (1 → N smiths)

Goal: run N forge agents in parallel **without** corrupting the shared multiple-testing
accounting, duplicating work, or letting agent-written code escape the sandbox.

## The 3 concurrency races to fix (correctness-critical)
1. **FDR registry read-modify-write** (THE non-negotiable). Today the harness does
   `n_fam = distinct_families()` → `bar = promote_dsr(n_fam)` → `registry.append_run()`
   with NO lock. Two agents both read n_fam=5, both bar-for-5, both append → the FDR
   bar fails to rise → false-discovery protection breaks. Must be atomic across processes.
2. **Duplicate hypotheses.** N agents each call `propose()` over the same wiki+candidates
   → they pick overlapping ideas → wasted compute + double-counting.
3. **Strategy-file / run-log collisions.** `sid` uses `time()%100000`; two agents in the
   same second collide. run_log/registry appends are O_APPEND-atomic (<4KB) so OK.

## Build pieces
- [ ] **P3.1 `sdk/locks.py`** — cross-process `FileLock` (atomic O_CREAT|O_EXCL under
      `research-wiki/.locks/`, owner+expiry, stale-steal, context manager). Tiny, no deps.
- [ ] **P3.2 Registry lock in harness** — wrap ONLY the `distinct_families → promote_dsr →
      append_run` sequence in `FileLock("fdr-registry")`. Gates/CPCV stay outside the lock
      (hold-time stays milliseconds). The correctness fix.
- [ ] **P3.3 `sdk/queue.py` + claim** — `research-wiki/.queue/queue.jsonl`: items with
      status queued/claimed/done + agent + ts. `enqueue`, `claim_next(agent)` (under lock:
      oldest queued → claimed), `complete(id, verdict)`. Decouples generation from execution.
- [ ] **P3.4 `agent/director.py`** — under a director lock, keep the queue topped to ≥K
      deduped, prioritized hypotheses (calls propose(), excludes registry+in-flight+queue
      titles). Priorities = combinations of validated legs / complementary premia / corners.
- [ ] **P3.5 `agent/run_worker.py`** — worker loop: claim_next → codegen → run (harness
      now registry-locked) → complete + run_log. Unique `agent_id`=host+pid; honors
      LOOP_DISABLED; bounded iterations; sid includes agent_id (no collisions).
- [ ] **P3.6 Sandbox hardening** — before running generated code: (a) AST denylist scan
      (reject os.system/subprocess/socket/open-for-write/eval/__import__ of net/fs) — the
      harness, not the agent, owns all I/O; (b) `resource.setrlimit` in preexec_fn
      (RLIMIT_CPU, RLIMIT_AS mem cap, RLIMIT_NPROC, RLIMIT_FSIZE); keep timeout+confined cwd.
- [ ] **P3.7 systemd** — templated `hephaestus-worker@.service` (1..N) + optional
      `hephaestus-director.service`. nice'd/idle-IO. **Installed DISABLED by default**; the
      existing single nightly `hephaestus-cycle` keeps running until N is proven.

## Rollout (cautious)
1. Build P3.1–P3.6. 2. Local test: run 2 workers + director against the live wiki, confirm
   no dup hypotheses, FDR bar rises monotonically, sandbox rejects a malicious signal.
3. Only then wire systemd worker@ (disabled). Keep nightly cycle as the live path meanwhile.

## Acceptance criteria
- 2 workers run concurrently with ZERO duplicate experiments (queue claim works).
- Under a forced concurrent burst, `n_families` is monotonic + every append reflects a
  unique serialized count (registry lock works).
- A planted malicious signal (file write / network / subprocess) is REJECTED pre-run.
- `LOOP_DISABLED` halts all workers. Rails remain non-bypassable (harness owns gates).
- Default host state unchanged until I explicitly enable worker@ (nightly cycle still live).

## Key design decisions (confirm before build)
- D1 Queue model: **director-fills-queue + workers-claim** (PLAN-specified) vs each-worker-
     proposes-with-dedup-lock. → recommend director+queue.
- D2 Sandbox: **rlimits + AST denylist** ($0, pragmatic) vs container/namespace isolation
     (heavier). → recommend rlimits + AST denylist now.
- D3 Rollout: build+local-2-worker-test, systemd worker@ **disabled by default**, nightly
     cycle stays live. → recommend cautious path.
