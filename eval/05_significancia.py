#!/usr/bin/env python3
"""
05_significancia.py
═══════════════════
Testes de significância citados no artigo (Seções 4 e 5):

  • Recuperação — nDCG@5 pareado por pergunta: fusão vs. melhor isolado,
    densos vs. BM25, tradução da consulta (CLIR) e BGE-m3 vs. Gemini.
  • Não-inferioridade (BGE-m3 vs. Gemini) — IC95% da diferença de nDCG@5
    contra a margem `noninferiority_margin`. Julga pelo IC, não pelo p.
  • RQ1 — cada cenário COM RAG (sem ICL) contra o baseline sem RAG, em
    todos os critérios dos juízes (média dos 2 juízes por pergunta).
  • RQ3 — ICL on vs. off, por cenário, em fidelidade, citação e alucinação.

Cada comparação traz Δ médio + IC95% (bootstrap), p de randomização pareada,
p de Wilcoxon e δ de Cliff. A correção de Holm é aplicada sobre TODA a família
de comparações de uma vez. Sementes fixas: a saída é determinística.

Saída: results/significancia.csv (e a tabela impressa no terminal).

Uso:  python eval/05_significancia.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.common import load_config, load_jsonl, resolve, setup_io
import lib.stats as st

import json

# `hallucination` é 1..5 com 5 = nenhuma alucinação: "reduzir alucinação" é
# nota MAIOR, mesma direção dos demais critérios.
ICL_CRITERIA = ("faithfulness", "citation", "hallucination")


def _aligned(d1, d2):
    keys = sorted(set(d1) & set(d2))
    return [d1[k] for k in keys], [d2[k] for k in keys]


def _compare(metric, label, x, y, n_perm, n_boot):
    if not x:
        return None
    delta, p_r = st.paired_randomization(x, y, n=n_perm)
    _, lo, hi = st.bootstrap_mean_diff_ci(x, y, n_boot=n_boot)
    return {"comparison": label, "metric": metric, "delta": delta, "lo": lo, "hi": hi,
            "p_rand": p_r, "p_wil": st.wilcoxon_p(x, y), "cliff": st.cliffs_delta(x, y),
            "n": len(x)}


def _noninferiority(ret_perq, margin, n_boot, ref="B_dense_gemini"):
    out = []
    for s in sorted(ret_perq):
        if s.startswith("B_dense") and s != ref and ref in ret_perq:
            x, y = _aligned(ret_perq[s], ret_perq[ref])
            if x:
                delta, lo, hi = st.bootstrap_mean_diff_ci(x, y, n_boot=n_boot)
                out.append({"embedding": s, "delta": delta, "lo": lo, "hi": hi,
                            "noninf": lo >= -margin, "n": len(x)})
    return out


def generation_per_query(gen_dir, judges, criteria):
    """perq[config][criterion][qid] = média dos juízes que deram nota."""
    perq = {}
    for fp in sorted(glob.glob(str(Path(gen_dir) / "judged_*.jsonl"))):
        pq = defaultdict(dict)
        for r in load_jsonl(fp):
            sc = r.get("scores", {})
            for crit in criteria:
                vals = [(sc.get(j) or {}).get(crit) for j in judges]
                vals = [v for v in vals if isinstance(v, int)]
                if vals:
                    pq[crit][r["qid"]] = float(np.mean(vals))
        perq[Path(fp).stem.replace("judged_", "")] = pq
    return perq


def main():
    setup_io()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = resolve(cfg["paths"]["results_dir"])
    judges = [m["name"] for m in cfg["judges"]["models"]][:2]
    criteria = cfg["judges"]["criteria"]
    sigcfg = cfg["metrics"]["significance"]
    NP, NB = sigcfg["n_permutations"], sigcfg.get("n_bootstrap", 10000)

    with open(res / "retrieval" / "per_query_ndcg.json", encoding="utf-8") as f:
        ret_perq = json.load(f)
    gperq = generation_per_query(res / "generation", judges, criteria)

    comps = []
    # Recuperação: fusão vs. o melhor sistema isolado
    if "C_fusion" in ret_perq:
        singles = [s for s in ret_perq if s.startswith(("A_bm25", "B_dense"))]
        if singles:
            best = max(singles, key=lambda s: sum(ret_perq[s].values()) / len(ret_perq[s]))
            comps.append(_compare("recuperação nDCG", f"C_fusion − {best}",
                                  *_aligned(ret_perq["C_fusion"], ret_perq[best]), NP, NB))
    # densos vs. BM25
    if "A_bm25" in ret_perq:
        for b in [s for s in ret_perq if s.startswith("B_dense")]:
            comps.append(_compare("recuperação nDCG", f"{b} − A_bm25",
                                  *_aligned(ret_perq[b], ret_perq["A_bm25"]), NP, NB))
    # tradução da consulta (CLIR)
    if "A_bm25" in ret_perq and "A_bm25_mt" in ret_perq:
        comps.append(_compare("recup. nDCG (CLIR)", "A_bm25_mt − A_bm25",
                              *_aligned(ret_perq["A_bm25_mt"], ret_perq["A_bm25"]), NP, NB))
    # embeddings entre si
    bs = sorted(s for s in ret_perq if s.startswith("B_dense"))
    for i in range(len(bs)):
        for j in range(i + 1, len(bs)):
            comps.append(_compare("recuperação nDCG", f"{bs[i]} − {bs[j]}",
                                  *_aligned(ret_perq[bs[i]], ret_perq[bs[j]]), NP, NB))
    # RQ1: com RAG (sem ICL) vs. sem RAG
    if "no_rag" in gperq:
        for cen in sorted(k for k in gperq if k.endswith("_icl0")):
            for crit in criteria:
                if crit in gperq.get(cen, {}) and crit in gperq["no_rag"]:
                    comps.append(_compare(f"geração {crit}", f"[RQ1] {cen[:-5]} − sem RAG",
                                          *_aligned(gperq[cen][crit], gperq["no_rag"][crit]),
                                          NP, NB))
    # RQ3: ICL on vs. off
    for c1 in sorted(k for k in gperq if k.endswith("_icl1")):
        c0 = c1[:-1] + "0"
        if c0 in gperq:
            for crit in ICL_CRITERIA:
                if crit in gperq[c0] and crit in gperq[c1]:
                    comps.append(_compare(f"geração {crit}", f"[RQ3] {c1[:-5]}: ICL on − off",
                                          *_aligned(gperq[c1][crit], gperq[c0][crit]), NP, NB))

    comps = [c for c in comps if c]
    corr = sigcfg.get("correction", "holm")
    for c, pa in zip(comps, st.correct([c["p_rand"] for c in comps], corr)):
        c["p_adj"] = pa
        c["sig"] = pa < sigcfg["alpha"]

    out = res / "significancia.csv"
    fields = ["comparison", "metric", "delta", "lo", "hi", "p_rand", "p_adj",
              "p_wil", "cliff", "n", "sig"]
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in comps:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in c.items()})

    print(f"| comparação | métrica | Δ [IC95%] | p | p({corr}) | Wilcoxon | δ Cliff | n |")
    print("|---|---|---|---|---|---|---|---|")
    for c in comps:
        print(f"| {c['comparison']} | {c['metric']} | {c['delta']:+.4f} "
              f"[{c['lo']:+.3f}, {c['hi']:+.3f}] | {c['p_rand']:.4f} | {c['p_adj']:.4f} | "
              f"{c['p_wil']:.4f} | {c['cliff']:+.3f} | {c['n']} |")

    nim = sigcfg.get("noninferiority_margin", 0.03)
    print(f"\nNão-inferioridade vs. Gemini (margem −{nim}, julgada pelo IC95%):")
    for r in _noninferiority(ret_perq, nim, NB):
        veredito = "não-inferior" if r["noninf"] else "NÃO demonstrada"
        print(f"  {r['embedding']}: Δ={r['delta']:+.4f} [{r['lo']:+.3f}, {r['hi']:+.3f}] "
              f"n={r['n']} → {veredito}")
    print(f"\n✓ {out}")


if __name__ == "__main__":
    main()
