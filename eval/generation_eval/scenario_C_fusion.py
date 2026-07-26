#!/usr/bin/env python3
"""
Cenário C (geração) — Qwen responde a partir do contexto da FUSÃO de deploy
(C_fusion), COM e SEM ICL. É a configuração que a ferramenta implanta.
Requer a recuperação (ranking C_fusion) já rodada.

Uso:  python eval/generation_eval/scenario_C_fusion.py [--limit N] [--no-judge]
"""
from __future__ import annotations
import argparse, sys
from _common import Setup, run_and_save


def run(S=None, cfg=None):
    S = S or Setup(cfg)
    rk = S.rankings("C_fusion")
    if rk is None:
        sys.exit("! ranking C_fusion ausente — rode a recuperação (retrieval_eval) antes.")
    for icl in S.icl_variants:
        run_and_save(S, f"C_fusion_icl{int(icl)}", rk, icl)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int); ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()
    run(S=Setup(limit=a.limit, no_judge=a.no_judge))
