#!/usr/bin/env python3
"""
eval/tools/corrige_qrels_q0015_q0016.py
═══════════════════════════════════════
Correção pontual do gabarito de DUAS consultas, para o artigo do WFA.

MOTIVO
------
q0015 e q0016 tinham 91 e 100 trechos rotulados relevantes (mediana do gold set = 2).
Não era julgamento de anotação: a política de anotação marcou o intervalo de páginas
da resposta, e essas páginas caíram em cima do bug de chunking de
`src/extract_to_jsonl.py:580` (`new_start = max(end - overlap, start + 1)`), que
degenera para passo de 1 caractere e emite dezenas de fragmentos quase idênticos.
UNESCO p45 tem 83 chunks; GDPR p14 tem 87. A mediana do acervo é 3 chunks por página.

O gabarito herdou a explosão. Efeito: recall@5 mecanicamente limitado a ~2% nessas
duas consultas, e nDCG@5 inflado (quase qualquer trecho do documento certo pontua).

Além disso, o gabarito de q0016 marcava apenas CONSIDERANDOS (págs. 12-14) e não o
Art. 20 — o texto vinculante — que está no acervo em `p45_c0`/`p45_c1` e era
recuperado pelos sistemas (2ª posição no B_dense_gemini). Os sistemas achavam o
artigo e não recebiam crédito.

CRITÉRIO (fixado ANTES de olhar o efeito nas métricas)
------------------------------------------------------
grau 2 = trecho que responde diretamente a pergunta
grau 1 = trecho que sustenta ou completa a resposta
Fragmentos quase idênticos a um trecho já rotulado saem.

ESCOPO
------
Altera SOMENTE `eval/data/qrels.csv` e `eval/data/golden_qa.jsonl`, e SOMENTE as
duas consultas. Não roda avaliação, não toca `eval/results/`, não chama API.
Faz backup `.bak-pre-correcao-qrels` de cada arquivo antes de escrever.

A correção de raiz — o bug do chunker e a reindexação — é outra tarefa, em branch
separada. Este script trata apenas do gabarito, com o índice atual.

Uso:
    python eval/tools/corrige_qrels_q0015_q0016.py            # aplica
    python eval/tools/corrige_qrels_q0015_q0016.py --dry-run  # só mostra
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import shutil
from pathlib import Path

csv.field_size_limit(10 ** 9)

ROOT = Path(__file__).resolve().parents[2]
QRELS = ROOT / "eval" / "data" / "qrels.csv"
GOLD = ROOT / "eval" / "data" / "golden_qa.jsonl"
SUFIXO_BKP = ".bak-pre-correcao-qrels"

NOVO: dict[str, dict[str, int]] = {
    # "Como a ferramenta de avaliação de impacto ético da UNESCO classifica a
    #  gravidade de um impacto negativo?"  →  quatro níveis
    "q0015": {
        # "a continuum of four Gravity level: Moderate/ Minor, Serious, Critical
        #  and Catastrophic" + definição de Catastrophic
        "ethical_impact_assessment_UNESCO_2023_p45_c0": 2,
        # continuação da tabela: definições de Critical e Serious
        "ethical_impact_assessment_UNESCO_2023_p45_c81": 1,
    },
    # "O direito à portabilidade do GDPR se aplica quando o tratamento se baseia
    #  em obrigação legal do controlador?"  →  não
    "q0016": {
        # Art. 20 completo: §1(a) consentimento/contrato e §3 "shall not apply to
        # processing necessary for the performance of a task carried out in the
        # public interest or in the exercise of official authority"
        "gdpr_regulation_EU_2016_p45_c1": 2,
        # considerando 68: "It should not apply where processing is based on a
        # legal ground other than consent or contract"
        "gdpr_regulation_EU_2016_p13_c3": 2,
        "gdpr_regulation_EU_2016_p45_c0": 1,   # cabeçalho do Art. 20 + abertura do §1
        "gdpr_regulation_EU_2016_p13_c2": 1,   # abertura do considerando 68
        "gdpr_regulation_EU_2016_p13_c4": 1,   # cauda do considerando 68
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="mostra o que mudaria e não escreve nada")
    args = ap.parse_args()

    print(f"repositório: {ROOT}")
    print(f"  {QRELS.relative_to(ROOT)}")
    print(f"  {GOLD.relative_to(ROOT)}\n")

    # ── qrels.csv ───────────────────────────────────────────────────────────
    with io.open(QRELS, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        linhas = list(r)

    antes = collections.Counter(l[0] for l in linhas)
    saida, inserido = [], set()
    for l in linhas:
        qid = l[0]
        if qid in NOVO:
            if qid not in inserido:          # substitui o bloco na posição original
                saida.extend([qid, cid, str(g)] for cid, g in NOVO[qid].items())
                inserido.add(qid)
            continue                          # descarta as linhas antigas
        saida.append(l)

    faltando = set(NOVO) - inserido
    if faltando:
        raise SystemExit(f"ERRO: {sorted(faltando)} não existem em qrels.csv — nada foi escrito.")

    depois = collections.Counter(l[0] for l in saida)
    print(f"qrels.csv: {len(linhas)} → {len(saida)} linhas")
    for q in NOVO:
        print(f"   {q}: {antes[q]} → {depois[q]} trechos")

    # ── golden_qa.jsonl ─────────────────────────────────────────────────────
    recs, mudados = [], []
    with io.open(GOLD, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if d["qid"] in NOVO:
                    mudados.append((d["qid"], len(d.get("qrels") or {}), len(NOVO[d["qid"]])))
                    d["qrels"] = NOVO[d["qid"]]
                recs.append(d)

    print(f"\ngolden_qa.jsonl: {len(recs)} registros")
    for qid, a, b in mudados:
        print(f"   {qid}: campo qrels {a} → {b} trechos")

    if args.dry_run:
        print("\n--dry-run: nada foi escrito.")
        return

    for p in (QRELS, GOLD):
        bkp = p.with_suffix(p.suffix + SUFIXO_BKP)
        if bkp.exists():
            print(f"\nbackup já existia, preservado: {bkp.name}")
        else:
            shutil.copy2(p, bkp)
            print(f"\nbackup: {bkp.name}")

    with io.open(QRELS, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(saida)

    with io.open(GOLD, "w", encoding="utf-8", newline="\n") as f:
        for d in recs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # ── conferência: os dois artefatos batem? ───────────────────────────────
    q_csv: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for row in csv.DictReader(io.open(QRELS, encoding="utf-8-sig", newline="")):
        q_csv[row["QueryId"]][row["ChunkId"]] = int(row["Relevance"])
    q_json = {d["qid"]: d["qrels"] for d in recs if d.get("qrels")}

    div = [q for q in set(q_csv) | set(q_json) if q_csv.get(q) != q_json.get(q)]
    print(f"\nconferência — consultas em que qrels.csv e golden_qa.jsonl divergem: {len(div)}")
    for q in sorted(div)[:10]:
        print(f"   {q}: csv={len(q_csv.get(q, {}))} json={len(q_json.get(q, {}))}")

    tam = [len(v) for v in q_csv.values()]
    tam.sort()
    print(f"\ndensidade do gabarito: {len(tam)} consultas · "
          f"mediana {tam[len(tam)//2]} · média {sum(tam)/len(tam):.1f} · "
          f"máximo {tam[-1]} · {sum(tam)} rótulos no total")
    print("\nPróximo passo (não executado por este script): reprocessar as métricas a "
          "partir dos rankings JÁ em disco — ver eval/retrieval_eval/_common.py:finalize.")


if __name__ == "__main__":
    main()
