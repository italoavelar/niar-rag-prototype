#!/usr/bin/env python3
"""
Cenário A (geração) — Qwen responde a partir do contexto recuperado pelo BM25,
avaliado COM e SEM ICL. Qual BM25 vem de retrieval.bm25.bm25_for_gen: por padrão
o A_bm25_mt (consulta traduzida), que é o mais forte e o mesmo que a fusão usa —
assim o cenário A é o melhor BM25 possível.

Uso:  python eval/generation_eval/scenario_A_bm25.py [--limit N] [--no-judge]
"""
from __future__ import annotations
import argparse, sys
from _common import Setup, run_and_save


def run(S=None, cfg=None):
    S = S or Setup(cfg)
    system = S.bm25_for_gen
    rk = S.rankings(system)
    if rk is None and system != "A_bm25":
        print(f"! ranking {system} ausente — caindo para A_bm25 (sem tradução). "
              f"Para usar o traduzido, rode a recuperação com translate_query=true.")
        system, rk = "A_bm25", S.rankings("A_bm25")
    if rk is None:
        sys.exit(f"! ranking {system} ausente — rode a recuperação (retrieval_eval) antes.")
    for icl in S.icl_variants:
        run_and_save(S, f"{system}_icl{int(icl)}", rk, icl)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int); ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()
    run(S=Setup(limit=a.limit, no_judge=a.no_judge))
