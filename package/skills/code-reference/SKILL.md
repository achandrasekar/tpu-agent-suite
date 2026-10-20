---
name: code-reference
description: Fetches the upstream kernel repos — jax-ml/jax and openxla/tokamax — into a gitignored code-reference/ directory at the workspace root, then routes a kernel-development question to the specific upstream unit test or op directory that answers it. Use when you need a worked example of a TPU Pallas pattern (SparseCore, grouped GEMM or MoE, splash or paged attention, manual pipelining, ragged ops), the unit test that would prove such a kernel correct, the file layout to copy for a new kernel plus its test, or to check a pinned tokamax or jax citation in the write-tpu-kernels references. Reports an existing clone's sha and leaves it alone; updates only when passed --update.
references:
- references/code-reference/MAP.md
---

You make two upstream repositories — `jax-ml/jax` and `openxla/tokamax` — reachable as a *lookup
table*, not as reading. They hold the worked examples this package only cites: `<jax>/tests/pallas/` is
the largest body of TPU Pallas unit tests that exists, and tokamax's per-op directories are the
best available template for how a kernel and its test are laid out together.

They are also large. The `jax` checkout is ~76 MB and its TPU test files run to thousands of lines each.
**You are the boundary that keeps them out of the caller's context**: you return
`path:line-line@sha` citations, never pasted source.

## Paths used on this page

- `$ROOT`: the root of the user's workspace, resolved below.
- `<code-reference>`: `$ROOT/code-reference/`, the gitignored directory that holds both clones and
  the manifest. Every command on this page runs from `$ROOT`, so the commands name it
  `code-reference/`.
- `<jax>`: `<code-reference>/jax`, the clone of `jax-ml/jax`.
- `<tokamax>`: `<code-reference>/tokamax`, the clone of `openxla/tokamax`.

## Decide which path to take

Resolve the clone root once, from git, never from the cwd — a subagent's cwd is not the repo root:

```bash
ROOT="$(git rev-parse --show-toplevel)"; echo "$ROOT/code-reference"
```

Then take the **first** branch that matches:

| condition | go to |
| --- | --- |
| The caller asked a kernel **question** and both clones exist | *Answering a kernel question*. Do not fetch, do not stat the remote. |
| `<code-reference>/<repo>/` is **missing** | *Fetch* |
| Present, no `--update` in the arguments | *Report the pin, do not update* |
| Present, `--update` given | *Update* |
| Present but **broken** — no `.git`, an interrupted checkout, `git status` errors | *Repair* |

*Repair* is: report exactly what is wrong, name the directory, and **stop**. Never `rm -rf` a clone
the user did not ask you to delete — a half-written `jax` is a slow re-download, not a crisis, and
the user may have put something in there.

The two repos are independent. One may need a fetch while the other only needs its pin reported.

## Preconditions to verify up front

Check these before doing work, and report clearly if one fails rather than pushing through.

| probe | if it fails |
| --- | --- |
| `git --version` — need ≥ 2.19 for `--filter` | Fall back to a plain `--depth 1` clone, and record in the manifest that arbitrary-pin resolution is **unavailable**. Say so in your reply; it silently breaks the citation-checking use case. |
| `git ls-remote --heads https://github.com/openxla/tokamax.git main` (and the same for `jax-ml/jax`) | HTTPS to GitHub is down or proxied. Report the actual error text and stop. Do not retry-loop. |
| `git -C "$ROOT" check-ignore -v code-reference` must print a rule | **Stop and refuse.** No rule ignores the directory, or somebody has negated the ignore, and you are one `git add` away from committing a nested git repo into the workspace — a mistake that is unrecoverable by anything short of a history rewrite. Report it, name the rule that is missing (`/code-reference/` in `$ROOT/.gitignore` or `$ROOT/.git/info/exclude`), and do not clone. |
| `df -h "$ROOT"` — need ≥ 1 GB free | Report the free space. The failure mode is a half-written clone, which then looks like *Repair* to the next caller. |

## Fetch

Two different commands, on purpose. `--filter=blob:none` drops *history* blobs, not the working
tree, so dropping `--depth 1` costs only commits and trees:

```bash
cd "$ROOT" && mkdir -p code-reference && cd code-reference

# tokamax — blobless, FULL history. ~1240 commits; measured 14 MB on disk, ~4 s.
git clone --filter=blob:none https://github.com/openxla/tokamax.git

# jax — blobless, depth 1. 187 MB of upstream git history nobody here needs.
git clone --depth 1 --filter=blob:none https://github.com/jax-ml/jax.git
```

Measured 2026-09-10: tokamax 14 MB in 4 s, jax 76 MB in 16 s. A result far off those is a wrong
clone — check the filter was accepted before you trust the tree.

**Do not "fix" tokamax to `--depth 1`.** Full history on tokamax is what makes
`git show 9da2369:<path>` resolve, and that is the only way to check the ~76 pinned
`[source: tokamax/…@9da2369]` citations in the {{skill:write-tpu-kernels}} references. On a 13 MB repo the
history is free. jax is depth-1 because its pinned citations are tag-shaped (`@jax 0.10.2`) and a
tag is one cheap fetch away — see *Deepen onto a pinned sha or tag*.

Then write the manifest, then run *Verify and report*.

## Report the pin, do not update

```bash
git -C code-reference/tokamax log -1 --format='%H %cI %s'
git -C code-reference/jax     log -1 --format='%H %cI %s'
```

State both shas in your reply and change nothing else.

**An existing clone is a pin somebody may already be citing against.** Moving it silently
invalidates every `path:line@sha` citation written since it landed — including line ranges, which
drift on any edit upstream. Refreshing is an explicit act with an explicit flag.

## Update

Only when the caller passed `--update`.

First refuse on a dirty tree:

```bash
git -C code-reference/<repo> status --porcelain    # any output → stop and report the files
```

Then, per repo — they differ because one is shallow:

```bash
# tokamax (full history)
git -C code-reference/tokamax pull --ff-only

# jax (shallow; `pull` on a depth-1 clone is not what you want)
git -C code-reference/jax fetch --depth 1 origin main
git -C code-reference/jax reset --hard FETCH_HEAD
```

Rewrite the manifest, and **print old → new sha per repo in your reply**. That line is the whole
point of the update path: it is what tells the caller whether citations they are holding need
re-checking. If a repo did not move, say "already current" and give the sha anyway.

## Deepen onto a pinned sha or tag

Use this when checking a `[source: …@<sha>]` or `[source: …@jax 0.10.2]` citation, not for browsing.
GitHub serves arbitrary reachable SHAs and tags.

```bash
# jax, onto a release tag, without moving the working tree
git -C code-reference/jax fetch --depth 1 origin refs/tags/jax-v0.10.2:refs/tags/jax-v0.10.2
git -C code-reference/jax show jax-v0.10.2:jax/_src/pallas/mosaic/tpu_info.py | sed -n '1,60p'

# jax, onto an arbitrary sha
git -C code-reference/jax fetch --depth 1 origin <sha>
git -C code-reference/jax show FETCH_HEAD:<path>
```

**tokamax needs none of this** — it has full history, so `git show 9da2369:<path>` works directly.

If you want more than two or three files at a pin, `git worktree add` at that pin rather than
checking the main tree out backwards; leaving `<code-reference>/<repo>` on a detached old commit is
the fastest way to make the next caller's citations wrong.

## The manifest

Write `<code-reference>/MANIFEST.md` — **beside** the clones, never inside one, where it would show as
untracked in *that* repo. One block per repo:

```markdown
## jax
url:          https://github.com/jax-ml/jax.git
clone mode:   depth-1 blobless          # or: blobless full | depth-1 plain
sha:          <full sha>
committed:    <%cI>
subject:      <%s>
fetched:      <YYYY-MM-DD>
size:         <du -sh>
resolves an arbitrary pinned sha:  no — fetch --depth 1 origin <sha> first
```

The manifest is gitignored, so it is in nobody's context and nobody's diff. **Your reply repeats the
shas.** The manifest is the record; the reply is the delivery.

## Answering a kernel question

1. **Classify** the question into one row of {{reference:code-reference/MAP.md}}. The eight highest-traffic rows are
   inlined below, so the common case costs no second file read.
2. **Grep the named subtree first, then Read a window.** Never Read one of these files whole —
   `tpu_pallas_test.py` alone is thousands of lines. Grep for the symbol or the test name, then Read
   ±40 lines around the hit.
3. **Budget: at most 5 files opened per question.** If you need more, the question was really
   several questions — split it and say so rather than blowing through the budget.
4. **If no row fits**, say so plainly, name the nearest row, and *propose the new row*. A new
   routing-map row is a real deliverable of this skill, not a side effect.
5. Emit the report block from *Verify and report*.

## The routing map

The full table is {{reference:code-reference/MAP.md}} — about thirty rows, plus an annotated inventory of every
TPU-relevant file in `<jax>/tests/pallas/`, the shipped kernels in `<jax>/jax/experimental/pallas/ops/tpu/`,
the tokamax op directories, and the per-op file convention to copy when laying out a new kernel and
its test. Read it when the question is not one of these:

| Question | Go to |
| --- | --- |
| A first Pallas TPU kernel, any shape | `<jax>/jax/experimental/pallas/ops/tpu/example_kernel.py`, `<jax>/docs/pallas/quickstart.md` |
| Mosaic TPU semantics — DMA, semaphores, scratch, `pltpu` primitives | `<jax>/tests/pallas/tpu_pallas_test.py` — grep it, never read it whole |
| Manual pipelining, `emit_pipeline`, overlap | `<jax>/tests/pallas/tpu_pallas_pipeline_test.py` and `tpu_pallas_async_test.py`; `<jax>/docs/pallas/pipelining.md` |
| A **SparseCore** Pallas kernel | `<jax>/docs/pallas/tpu/sparsecore.md` first — runnable worked examples; then `<jax>/tests/pallas/tpu_sparsecore_pallas_test.py`; API in `<jax>/jax/experimental/pallas/tpu_sc.py`. tokamax has none. Pair with §10 of the {{skill:write-tpu-kernels}} skill's `hardware-facts.md` |
| **Grouped GEMM, MoE, ragged dot** | `<jax>/jax/experimental/pallas/ops/tpu/megablox/` + `<jax>/tests/pallas/tpu_gmm_test.py` + `<tokamax>/tokamax/_src/ops/ragged_dot/` |
| **Attention** — splash, flash | `<jax>/jax/experimental/pallas/ops/tpu/splash_attention/` and `flash_attention.py`; tests `tpu_splash_attention_kernel_test.py`, `tpu_splash_attention_kernel_sharded_test.py`, `tpu_splash_attention_mask_test.py`; `<tokamax>/tokamax/_src/ops/attention/` + `<tokamax>/docs/splash_attention.md` |
| **Paged / ragged-paged attention** — vLLM's decode kernel | `<jax>/jax/experimental/pallas/ops/tpu/paged_attention/` and `ragged_paged_attention/`; tests `tpu_paged_attention_kernel_test.py`, `tpu_ragged_paged_attention_test.py` |
| How to structure a new kernel **and its test** | the tokamax per-op layout — MAP.md § "The tokamax per-op file convention" |

## Never grep the clones repo-wide

These rules are the difference between this skill saving context and destroying it.

- **No repo-wide grep, ever.** Every search names a subtree. Find the file first, then search inside
  it:

  ```bash
  grep -rn "<symbol>" code-reference/jax/tests/pallas/ --include='tpu_*_test.py' -l
  grep -n  "<symbol>" code-reference/jax/tests/pallas/<the one file>
  ```

- **Skip `gpu_*`, `mgpu_*`, `pallas_triton*`, `pallas_mosaic_gpu*`.** Roughly half of
  `<jax>/tests/pallas/` and most files in each tokamax op directory are GPU, and they lower through a
  different compiler. One named trap:
  `<tokamax>/tokamax/_src/pallas/block.py` is Triton/GPU code, not TPU — do
  not read it for TPU.

- **Citations, not source.** The unit of output is
  `` `<jax>/tests/pallas/tpu_gmm_test.py:412-448@<sha>` `` plus one line saying what is there. In
  the citation you return, the path is relative to `<code-reference>`: write the clone name, `jax`
  or `tokamax`, as its first segment. Paste at
  most ~20 lines, and only when the exact text *is* the answer — a signature, a decorator, an
  argument order. If the caller wants more they can Read that range themselves; they now have the
  path.

- **Never Read a large file whole.** Grep, then Read with an offset and a limit.

- **Upstream source is a rung on the evidence ladder, not the top of it.** It proves an API exists
  and a pattern compiles. It says nothing about what that pattern costs at your shapes, on this
  hardware, in this workload. On the {{skill:write-tpu-kernels}} ladder it sits beside a `[verified]` fact measured
  at another shape — below a probe you ran this session. Do not let a worked example upstream settle
  a question that only a measurement can settle.

## Verify and report

After a *Fetch* or an *Update*, run the functional checks. `git clone` can exit 0 having produced a
repo that does not do what you need — a filter the remote declined, or somebody's `--depth 1` on
tokamax — so **never report success on the strength of an exit code alone**:

```bash
cd "$ROOT"
test -f code-reference/jax/tests/pallas/tpu_pallas_test.py
test -d code-reference/tokamax/tokamax/_src/ops/ragged_dot
git -C code-reference/tokamax cat-file -e 9da23691ea784df734946dab03587949477e8573^{commit}
git -C code-reference/jax     log -1 --format='%H %cI %s'
git -C code-reference/tokamax log -1 --format='%H %cI %s'
du -sh code-reference/jax code-reference/tokamax
```

The `cat-file -e` is the one that matters: it is what distinguishes a working tokamax clone from a
plausible one. If it fails, the clone cannot resolve pinned citations — say so explicitly and record
it in the manifest.

Then report, concisely: which path you took per repo, both shas with their dates, the clone modes,
and anything that failed with its actual error text.

For a **kernel question**, report in this fixed block, so the reply stays bounded and skimmable:

```
question:   <restated in one line>
route:      <the MAP.md row used, or "no row — nearest is X, proposed row: …">
citations:  <path:lines@sha> — <what is there>        (1 to 5 of these)
pattern:    <3-8 lines: the shape of the upstream solution, in prose>
caveats:    <version skew, GPU-only siblings, anything you did not verify>
shas:       jax <sha> (<date>) · tokamax <sha> (<date>)
```

The `shas:` line is not decoration — a citation without the sha it was read at is exactly the kind
of claim that drifted last time.

## What this skill does not do

- **It does not install jax or tokamax.** These are checkouts to read, not environments to import
  from. Nothing here touches a venv.
- **It does not run their tests.** The TPU ones need TPUs. If a test genuinely has to run, run it
  with `{{runtime}} compute`, and only with a reason worth the cluster time.
- **It does not edit the {{skill:write-tpu-kernels}} references.** Promoting a finding into an area file is
  that skill's own "What you owe `references/`" protocol. This skill supplies the evidence; the
  caller does the promotion.
- **It reads only the clones under `<code-reference>`.** Any other tokamax or jax checkout on the
  machine is not the one the citations resolve against. If you find yourself reading one, you took
  a wrong path.
