#!/usr/bin/env python3
"""
Cenário B (geração) — Qwen responde a partir do contexto do DENSO implantado
(retrieval.fusion.dense_for_fusion, ex.: gemini), COM e SEM ICL. Requer a
recuperação (ranking B_dense_<emb>) já rodada.

Uso:  python eval/generation_eval/scenario_B_dense.py [--limit N] [--no-judge]
"""
from __future__ import annotations
import argparse, sys
from _common import Setup, run_and_save


def run(S=None, cfg=None):
    S = S or Setup(cfg)
    system = f"B_dense_{S.dense_for_gen}"
    rk = S.rankings(system)
    if rk is None:
        sys.exit(f"! ranking {system} ausente — rode a recuperação (retrieval_eval) antes.")
    for icl in S.icl_variants:
        run_and_save(S, f"{system}_icl{int(icl)}", rk, icl)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int); ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()
    run(S=Setup(limit=a.limit, no_judge=a.no_judge))
