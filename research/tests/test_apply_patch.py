#!/usr/bin/env python3
"""Tests for src/apply.py's patching.

No llama.cpp clone is needed. A minimal fake tree holding only the anchor
strings the patch looks for is built, and the patch is applied to that.

There are two things to protect.

  1. **Failing immediately when an anchor is not found.** The worst case, when
     upstream rewrites the region in question, is that nothing applies and the
     build passes silently -- not noticed until the output is broken. apply.py
     guards this with asserts, so what is checked here is whether those asserts
     really fire.
  2. **Applying twice does not break it.** A rebuild, or the way layer caching
     happens to hit, can run it twice.
"""
import pathlib, re, shutil, subprocess, sys, tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
APPLY = REPO / 'src' / 'apply.py'

npass = nfail = 0
def check(ok, what):
    global npass, nfail
    if ok: npass += 1; print(f"  ok   {what}")
    else:  nfail += 1; print(f"  FAIL {what}")

def anchors():
    """Extract the strings apply.py substitutes and the files it writes."""
    src = APPLY.read_text()
    files = re.findall(r"p = root / '([^']+)'", src)
    return files

LIT = r"(\"\"\".*?\"\"\"|'''.*?'''|'[^']*'|\"[^\"]*\")"

def _lit(x):
    return x[3:-3] if x[:3] in ("'''", '"""') else x[1:-1]

def build_fake_tree(base: pathlib.Path):
    """Build a minimal tree containing apply.py's anchors.

    The anchor text is taken from apply.py itself, so the test follows
    automatically when apply.py is edited and nothing is maintained twice.
    """
    src = APPLY.read_text()
    olds = [_lit(m) for m in re.findall(r"s = s\.replace\(\s*" + LIT, src, re.S)]
    # also picks up the form where it is bound to a variable first
    # (old / old2 / old_loop / old_end)
    olds += [_lit(m) for m in re.findall(r"^\s*old[a-z0-9_]* = " + LIT, src, re.S | re.M)]
    # regex substitutions demand a match count (n_mm >= 3), so add lines to satisfy it
    olds += ["build_lora_mm_id(gate_exps, cur, selected_experts, nullptr, n_expert_used, il);",
             "build_lora_mm_id(up_exps, cur, selected_experts, nullptr, n_expert_used, il);",
             "build_lora_mm_id(down_exps, cur, selected_experts, nullptr, n_expert_used, il);"]
    for f in anchors():
        p = base / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('\n\n'.join(olds) + '\n')
    return base

def run(tree):
    return subprocess.run([sys.executable, str(APPLY), str(tree)],
                          capture_output=True, text=True)

# ---- 1. it applies to a well-formed tree -----------------------------------
with tempfile.TemporaryDirectory() as d:
    tree = build_fake_tree(pathlib.Path(d))
    r = run(tree)
    check(r.returncode == 0, f"applies to a tree with all anchors present ({r.stderr.strip()[:60]})")
    check('applied' in r.stdout, f"reports how many were applied ({r.stdout.strip()})")
    after_first = {f: (tree / f).read_text() for f in anchors()}

    # ---- 2. applying twice changes nothing ---------------------------------
    r2 = run(tree)
    same = all((tree / f).read_text() == after_first[f] for f in anchors())
    check(r2.returncode == 0 and same, "a second application leaves the content unchanged (idempotent)")

# ---- 3. a missing anchor fails ---------------------------------------------
#   Reproduces upstream having rewritten the region. It must not pass silently.
for target in anchors():
    with tempfile.TemporaryDirectory() as d:
        tree = build_fake_tree(pathlib.Path(d))
        (tree / target).write_text("// upstream rewrote this file\n")
        r = run(tree)
        check(r.returncode != 0,
              f"fails when {target}'s anchor disappears")

# ---- 4. the patched files match llama.cpp's actual layout ------------------
expected_prefixes = ('src/', 'ggml/')
check(all(f.startswith(expected_prefixes) for f in anchors()),
      "patch targets are confined to llama.cpp's src/ and ggml/")

print(f"  ---- {npass} passed / {nfail} failed")
sys.exit(1 if nfail else 0)
