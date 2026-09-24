"""
eval/tools/contraste_por_juiz.py
════════════════════════════════
Contraste com RAG (denso, sem ICL) vs. sem RAG, analisado POR JUIZ.

Com concordância baixa entre os juízes (κ ponderado ≥ 0,6 em só 1 de 7
critérios), a média dos dois pode esconder que eles discordam até da direção do
efeito. Este script testa cada juiz isoladamente: Wilcoxon pareado por
pergunta, com correção de Holm sobre todos os testes (2 juízes × 7 critérios).
Também imprime a distribuição das notas de cada juiz, pooled nas 7
configurações, para mostrar a concentração em 5 que derruba o κ.

Uso:
  python eval/tools/contraste_por_juiz.py
"""
from __future__ import annotations

import collections
import glob
import json
from pathlib import Path

from scipy.stats import wilcoxon

EVAL = Path(__file__).resolve().parent.parent
GEN = EVAL / "results" / "generation"
JUDGES = ["llama", "gptoss"]
CRITERIA = ["faithfulness", "hallucination", "answer_relevance", "correctness",
            "completeness", "citation", "refusal"]


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return {r["qid"]: r for r in map(json.loads, f)}


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adj, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adj[i] = running
    return adj


def main():
    rag = load(GEN / "judged_B_dense_gemini_icl0.jsonl")
    base = load(GEN / "judged_no_rag.jsonl")

    rows = []
    for j in JUDGES:
        for c in CRITERIA:
            pares = [(rag[q]["scores"][j][c], base[q]["scores"][j][c])
                     for q in rag if q in base
                     and rag[q]["scores"][j][c] is not None
                     and base[q]["scores"][j][c] is not None]
            a = [x for x, _ in pares]
            b = [y for _, y in pares]
            delta = (sum(a) - sum(b)) / len(pares)
            rows.append([j, c, len(pares), sum(a) / len(a), sum(b) / len(b),
                         delta, wilcoxon(a, b).pvalue])

    for row, p_holm in zip(rows, holm([r[-1] for r in rows])):
        j, c, n, m_rag, m_base, delta, p = row
        print(f"{j:7s} {c:17s} n={n:3d}  RAG={m_rag:.2f}  sem RAG={m_base:.2f}  "
              f"delta={delta:+.2f}  p={p:.2g}  p_Holm={p_holm:.2g}")

    print("\nDistribuição das notas (7 configurações):")
    for c in CRITERIA:
        for j in JUDGES:
            cont = collections.Counter()
            for f in glob.glob(str(GEN / "judged_*.jsonl")):
                for r in load(Path(f)).values():
                    v = r["scores"][j][c]
                    if v is not None:
                        cont[v] += 1
            total = sum(cont.values())
            dist = {k: round(v / total, 2) for k, v in sorted(cont.items())}
            print(f"{c:17s} {j:7s} n={total}  nota 5={cont[5] / total:.2f}  {dist}")


if __name__ == "__main__":
    main()
