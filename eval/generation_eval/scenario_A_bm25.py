#!/usr/bin/env python3
"""
Cenário A (geração) — Qwen responde a partir do contexto recuperado pelo BM25,
avaliado COM e SEM ICL. Requer a recuperação (ranking A_bm25) já rodada.

Uso:  python eval/generation_eval/scenario_A_bm25.py [--limit N] [--no-judge]
"""
from __future__ import annotations
import argparse, sys
from _common import Setup, run_and_save


def run(S=None, cfg=None):
    S = S or Setup(cfg)
    rk = S.rankings("A_bm25")
    if rk is None:
        sys.exit("! ranking A_bm25 ausente — rode a recuperação (retrieval_eval) antes.")
    for icl in S.icl_variants:
        run_and_save(S, f"A_bm25_icl{int(icl)}", rk, icl)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int); ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()
    run(S=Setup(limit=a.limit, no_judge=a.no_judge))
