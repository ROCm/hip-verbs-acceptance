#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""propose_tests.py -- decision engine for extract-hip/verbs-requirements.

Pure-Python (stdlib only) replacement for the old jq-based propose_tests.sh, so
the skill needs no jq (python3 is available everywhere; jq often is not).

Reads the embedded KB, the joined symbol table, and the coverage corpus, and emits
  PROPOSED_TESTS.md   (human + agent readable)
  proposed_tests.json (machine readable)

A KB scenario is COVERED only when a test implementing it exists (its match_names
appear in a corpus test file / plan doc) -- API overlap is only a reuse hint.
Symbols the source uses that no KB scenario lists are UNMODELED; for each, we
include the code CONTEXT (the lines around its call sites) so a name like
ibv_create_srq is understandable without guessing.

Usage:
  propose_tests.py --out DIR --kb KB.json --joined joined.tsv --corpus corpus.tsv \
                   [--corpus-roots A:B:C] [--src SRC_DIR]
"""
import argparse, json, os, re, subprocess, sys

def read_tsv(path):
    rows = []
    try:
        with open(path) as f:
            for ln in f:
                rows.append(ln.rstrip("\n").split("\t"))
    except FileNotFoundError:
        pass
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--kb", required=True)
    ap.add_argument("--joined", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--corpus-roots", default="")
    ap.add_argument("--src", default="")
    a = ap.parse_args()

    kb = json.load(open(a.kb))
    reqs = kb.get("requirements", [])

    # joined.tsv: symbol, canonical, family, count, status, covered_in, paths, locs
    joined = read_tsv(a.joined)
    header = joined[0] if joined else []
    jrows = joined[1:] if joined else []
    # canonical symbol -> (count, sample_locs)
    used = {}
    for r in jrows:
        if len(r) < 8:
            continue
        canon = r[1].split(":")[0]
        used[canon] = (r[3], r[7])

    # corpus.tsv: symbol, tag, path
    corpus = read_tsv(a.corpus)
    covered_sym = {}
    for r in corpus:
        if len(r) < 3:
            continue
        covered_sym.setdefault(r[0], []).append((r[1], r[2]))

    # --- scenario index: test names + plan/spec text from corpus roots ---------
    scen_text = []
    for r in corpus:           # corpus file paths carry test names too
        if len(r) >= 3:
            scen_text.append(r[2])
    roots = [x for x in a.corpus_roots.split(":") if x]
    TESTRE = re.compile(r'TEST(?:_F|_P|_CASE)?\(([^,)]+)')
    for root in roots:
        if os.path.isfile(root):
            try: scen_text.append(open(root, errors="ignore").read())
            except OSError: pass
            continue
        if not os.path.isdir(root):
            continue
        for dp, _, fns in os.walk(root):
            for fn in fns:
                p = os.path.join(dp, fn)
                if fn.endswith((".cc", ".cpp")):
                    try: txt = open(p, errors="ignore").read()
                    except OSError: continue
                    for m in TESTRE.finditer(txt):
                        scen_text.append(m.group(1))
                elif fn.endswith((".md", ".txt")):
                    try: scen_text.append(open(p, errors="ignore").read())
                    except OSError: pass
    scen_blob = "\n".join(scen_text).lower()

    def scenario_covered(mnames):
        return any(mn and mn.lower() in scen_blob for mn in mnames)

    # --- classify each KB requirement -----------------------------------------
    items = []
    for req in reqs:
        apis = req.get("signal_apis", [])
        mnames = req.get("match_names") or [req.get("title",""), req.get("id","")]
        any_used = any(s in used for s in apis)
        reuse = []
        for s in apis:
            if s in covered_sym:
                reuse += [f"{s}:{t}({p})" for t, p in covered_sym[s]]
        if scenario_covered(mnames):
            status = "COVERED"
        elif any_used:
            status = "PROPOSE"
        else:
            status = "N/A"
        it = dict(req)
        it["status"] = status
        it["covered_in"] = " ".join(reuse)
        # For PROPOSE items, ground the Description in the real code: pull one
        # call-site excerpt per signal API that the source actually uses, so the
        # test author sees exactly which call/operands/order to reproduce.
        if status == "PROPOSE" and a.src and os.path.isdir(a.src):
            sites = []
            for s in apis:
                if s in used:
                    for ln in grep_context(a.src, s)[:1]:
                        sites.append(ln)
            # de-dup preserving order, cap to keep the block readable
            seen_s = set(); it["call_sites"] = []
            for ln in sites:
                if ln not in seen_s:
                    seen_s.add(ln); it["call_sites"].append(ln)
                if len(it["call_sites"]) >= 4:
                    break
        items.append(it)

    # --- UNMODELED: used canonical symbols no KB signal_api lists --------------
    kb_apis = set()
    for req in reqs:
        kb_apis.update(req.get("signal_apis", []))
    SKIP = ("_func",)  # plumbing trampolines
    def norm(s):  # collapse RCCL's dlsym indirection onto the public API name
        return s.replace("ibv_internal_", "ibv_").replace("mlx5dv_internal_", "mlx5dv_")
    seen = {}
    for canon, (cnt, locs) in sorted(used.items()):
        raw = canon.split(":")[0]
        if not raw or canon.endswith(":NO_HIP_EQUIV"):
            continue
        if any(raw.endswith(s) for s in SKIP):
            continue
        base = norm(raw)
        if base in kb_apis or base in covered_sym:
            continue
        # merge internal+public into one entry under the public name
        e = seen.get(base)
        if e is None:
            seen[base] = {"symbol": base, "calls": cnt, "locations": locs}
        else:
            try: e["calls"] = str(int(e["calls"]) + int(cnt))
            except ValueError: pass
    unmodeled = [seen[k] for k in sorted(seen)]

    # code context + a plain-language purpose for each unmodeled symbol, so a
    # reviewer can decide "do we need a test for this or not" without guessing.
    if a.src and os.path.isdir(a.src):
        for u in unmodeled:
            u["context"] = grep_context(a.src, u["symbol"])
            u["purpose"] = purpose_of(u["symbol"], u.get("context") or [])

    write_md(a.out, items, unmodeled)
    write_json(a.out, items, unmodeled)

    p = sum(1 for i in items if i["status"] == "PROPOSE")
    c = sum(1 for i in items if i["status"] == "COVERED")
    n = sum(1 for i in items if i["status"] == "N/A")
    print(f"propose_tests: PROPOSE={p} COVERED={c} N/A={n} UNMODELED={len(unmodeled)} "
          f"-> {a.out}/PROPOSED_TESTS.md , {a.out}/proposed_tests.json", file=sys.stderr)

def grep_context(src, sym):
    """Return up to 2 short code excerpts showing how `sym` is called."""
    try:
        out = subprocess.run(
            ["grep", "-rEn", "--include=*.cc", "--include=*.cpp", "--include=*.h",
             "--include=*.hpp", r"\b%s\s*\(" % re.escape(sym), src],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    def _rel(ln):
        return (ln[len(src):].lstrip("/") if ln.startswith(src) else ln).strip()[:200]
    def _is_comment(rel):
        code = rel.split(":", 2)[-1].lstrip()
        return code.startswith(("*", "//", "/*", "#"))
    real, weak = [], []   # real call sites first; wrappers/comments as fallback
    for ln in out.splitlines():
        rel = _rel(ln)
        if "_internal_" in ln or "wrapper.h" in ln or _is_comment(rel):
            weak.append(rel)
        else:
            real.append(rel)
    res = (real + weak)[:2]
    return res

def purpose_of(sym, ctx):
    """Best-effort one-liner: what this API does + why the source calls it, so a
    reviewer can judge whether it needs an acceptance test. Heuristics from the
    symbol family/name first, refined by the call-site comment/region if useful.
    No network, no man-pages -- just a readable hint, never a hard claim."""
    s = sym
    # family/role classification (HIP/CUDA-driver/HSA + verbs/rdma/mlx5dv)
    FAM = [
        (("cuCtx",),            "CUDA-driver context management (create/set/destroy a primary context)"),
        (("cuDevice",),         "CUDA-driver device query/handle (enumerate devices, query attributes)"),
        (("cuMem","cuMemory"),  "CUDA-driver low-level VMM (reserve/create/map/set-access physical memory)"),
        (("cuPointer",),        "CUDA-driver pointer attribute query"),
        (("cuStream",),         "CUDA-driver stream control"),
        (("cuLaunch","cuFunc"), "CUDA-driver kernel launch path"),
        (("cuMulticast",),      "CUDA-driver multicast object (NVLS/multimem)"),
        (("hipIpc",),           "HIP inter-process memory/event handle sharing"),
        (("hipHost",),          "HIP pinned/mapped host memory"),
        (("hipMalloc","hipFree","hipMem"), "HIP device-memory allocation / transfer"),
        (("hipStream",),        "HIP stream lifecycle / ordering"),
        (("hipEvent",),         "HIP event record/sync/timing"),
        (("hipGraph",),         "HIP graph capture/launch"),
        (("hipModule","hipFunc"),"HIP module load / kernel launch"),
        (("hsa_amd","hsa_"),    "HSA runtime / ROCr low-level (agents, signals, dmabuf, memory pools)"),
        (("ibv_create_qp","ibv_modify_qp","ibv_query_qp","ibv_destroy_qp"),
                                "RDMA QP lifecycle / state transition"),
        (("ibv_reg","ibv_dereg"),"RDMA memory-region registration"),
        (("ibv_post","ibv_poll"),"RDMA work-request post / completion poll (datapath)"),
        (("ibv_create_cq","ibv_destroy_cq"),"RDMA completion-queue lifecycle"),
        (("ibv_create_ah","ibv_destroy_ah"),"RDMA address handle (UD addressing)"),
        (("ibv_create_srq","ibv_modify_srq"),"RDMA shared receive queue"),
        (("ibv_event","ibv_get_async","ibv_ack"),"RDMA async-event / error reporting"),
        (("ibv_query","ibv_get_device","ibv_open","ibv_close","ibv_free_device"),
                                "RDMA device/port/GID discovery & open"),
        (("rdma_",),            "RDMA-CM connection management (resolve/connect/listen)"),
        (("mlx5dv_devx",),      "mlx5 DEVX direct-verbs object/UAR/UMEM (vendor low-level)"),
        (("mlx5dv_",),          "mlx5 direct-verbs vendor extension"),
    ]
    role = None
    for keys, desc in FAM:
        if any(s.startswith(k) for k in keys):
            role = desc; break
    if role is None:
        role = "unclassified API symbol — inspect the call site to decide its role"
    # where/why: name the first real call-site (skip header doc-comments).
    where = ""
    def _is_comment(ln):
        code = ln.split(":", 2)[-1].lstrip()
        return code.startswith(("*", "//", "/*", "#"))
    real = next((c for c in ctx if not _is_comment(c)), ctx[0] if ctx else "")
    if real:
        loc = real.split(":")[0:2]
        if len(loc) == 2:
            where = f" Used in {loc[0]}:{loc[1]}."
    return role + "." + where + " Decide: acceptance-relevant (a path that can " \
           "silently break on FW/driver/runtime updates) or pure plumbing (skip)."

def md_block(it):
    gate = f"  ·  gate: {it['gate']}" if it.get("gate") else ""
    seq = " -> ".join(it.get("sequence", []))
    # Richer Description: the KB description, then the impl_hint (concrete steps a
    # test author follows), so the proposed test is precise, not just a title.
    desc = it.get("description", "")
    hint = it.get("impl_hint", "")
    if hint:
        desc = f"{desc}\n  - **How to implement:** {hint}"
    s = [f"\n### [{it['status']}] {it.get('id','')} — `{it.get('title','')}`  ·  {it.get('class','')}{gate}",
         f"- **Area:** {it.get('area','')}",
         f"- **Sequence:** {seq}",
         f"- **Description:** {desc}",
         f"- **Verify:** {it.get('verify','')}",
         f"- **Signal APIs:** {' '.join(it.get('signal_apis', []))}",
         f"- **Provenance:** {it.get('provenance_hint','')}"]
    # Real call-sites from the analyzed source: exactly which calls to reproduce.
    if it.get("call_sites"):
        s.append("- **Real call-sites in this source (reproduce these):**")
        s.append("  ```")
        s += [f"  {c}" for c in it["call_sites"]]
        s.append("  ```")
    if it.get("covered_in"):
        s.append(f"- **Reuse hint (same APIs unit-tested elsewhere; scenario may still be needed):** {it['covered_in']}")
    return "\n".join(s)

def write_md(out, items, unmodeled):
    P = [i for i in items if i["status"] == "PROPOSE"]
    C = [i for i in items if i["status"] == "COVERED"]
    NA = [i for i in items if i["status"] == "N/A"]
    L = ["# Proposed Acceptance Tests", "",
         "Generated by extract-*-requirements. PROPOSE = add this test · COVERED = a",
         "test implementing the scenario already exists · N/A = the analyzed source",
         "does not use these APIs · UNMODELED = used in code but no scenario yet.", "",
         "## TO ADD (PROPOSE)"]
    L.append("".join(md_block(i) + "\n" for i in P) or "_none — every applicable requirement is already covered._")
    L += ["", "## UNMODELED — used in code, no scenario yet (candidates to add)", ""]
    if unmodeled:
        for u in unmodeled:
            L.append(f"### `{u['symbol']}`  ({u['calls']} call sites)")
            if u.get("purpose"):
                L.append(f"- **What it's for:** {u['purpose']}")
            ctx = u.get("context") or []
            if ctx:
                L.append("How it is used in the source:")
                L.append("```")
                L += ctx
                L.append("```")
            else:
                L.append(f"- locations: {u.get('locations','')}")
            L.append("")
    else:
        L.append("_none — every used API maps to a scenario._")
    L += ["", "## Already covered (scenario exists)", ""]
    L.append("".join(md_block(i) + f"\n  - **Covered in:** {i.get('covered_in') or 'corpus'}\n" for i in C) or "_none_")
    L += ["", "## Not applicable to this source", ""]
    L.append("".join(f"- {i.get('id','')} `{i.get('title','')}`\n" for i in NA) or "_none_")
    open(os.path.join(out, "PROPOSED_TESTS.md"), "w").write("\n".join(L) + "\n")

def write_json(out, items, unmodeled):
    summary = {
        "propose": sum(1 for i in items if i["status"] == "PROPOSE"),
        "covered": sum(1 for i in items if i["status"] == "COVERED"),
        "na":      sum(1 for i in items if i["status"] == "N/A"),
        "unmodeled": len(unmodeled),
    }
    obj = {"generated_by": "extract-requirements", "items": items,
           "unmodeled_symbols": unmodeled, "summary": summary}
    json.dump(obj, open(os.path.join(out, "proposed_tests.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
