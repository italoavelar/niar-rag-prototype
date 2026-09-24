"""
eval/tools/citation_grounding.py
════════════════════════════════
Métrica OBJETIVA de rastreabilidade da citação — SEM depender de juiz.

Pergunta que responde: das URLs que o sistema apresenta na seção "Fontes
utilizadas", quantas apontam de fato para documentos do acervo indexado?

  • ancorada   — a URL citada consta do corpus (metadata.source_url em
                 data/processed/documents.jsonl). O usuário consegue conferir.
  • não ancorada — a URL não corresponde a nenhum documento do acervo. Isso NÃO
                 significa que o endereço seja inexistente (não fazemos
                 requisição alguma); significa que a citação não é verificável
                 contra as fontes curadas, que é a promessa da ferramenta.

O contraste relevante é com o baseline sem RAG: o modelo sozinho não tem de
onde citar, mas emite a seção de fontes assim mesmo. A configuração com RAG
cita a partir do payload dos trechos recuperados — o resultado alto ali é
propriedade de projeto, não conquista empírica. Reporte os dois lados assim.

Saída: eval/results/generation/citation_grounding.csv (+ tabela no console).

Uso:
    python eval/tools/citation_grounding.py
    python eval/tools/citation_grounding.py --gen-dir eval/results/generation
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# URL termina no primeiro delimitador de markdown/pontuação. Sem o `]` o padrão
# engole links markdown inteiros (`url](url`) e subconta as citações.
_URL = re.compile(r'https?://[^\s\)\]\}\>"\',]+')
_SECAO_FONTES = re.compile(r"fontes\s+utilizadas", re.IGNORECASE)


def normalizar(url: str) -> str:
    """Forma canônica p/ comparação: sem protocolo, sem www, sem query/âncora."""
    u = url.strip().lower()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    u = u.split("#")[0].split("?")[0]
    return u.rstrip("/")


def urls_do_corpus(corpus_path: Path) -> set[str]:
    urls: set[str] = set()
    with corpus_path.open(encoding="utf-8") as fh:
        for linha in fh:
            if not linha.strip():
                continue
            meta = json.loads(linha).get("metadata") or {}
            fonte = meta.get("source_url")
            if fonte:
                urls.add(normalizar(str(fonte)))
    return urls


def ancorada(url_norm: str, corpus: set[str]) -> bool:
    """Exata, ou uma é prefixo da outra (tolera sufixo de âncora/página)."""
    if url_norm in corpus:
        return True
    return any(url_norm.startswith(c) or c.startswith(url_norm) for c in corpus)


def analisar(caminho: Path, corpus: set[str]) -> dict:
    respostas = seccao = com_url = 0
    total = dentro = dentro_exato = 0
    for linha in caminho.open(encoding="utf-8"):
        if not linha.strip():
            continue
        resposta = json.loads(linha).get("answer") or ""
        respostas += 1
        if _SECAO_FONTES.search(resposta):
            seccao += 1
        achadas = _URL.findall(resposta)
        if achadas:
            com_url += 1
        for bruta in achadas:
            u = normalizar(bruta)
            total += 1
            dentro_exato += u in corpus
            dentro += ancorada(u, corpus)
    return {
        "config": caminho.stem.replace("judged_", ""),
        "respostas": respostas,
        "com_secao_fontes": seccao,
        "com_url": com_url,
        "urls_citadas": total,
        "ancoradas": dentro,
        "ancoradas_exatas": dentro_exato,
        "nao_ancoradas": total - dentro,
        "pct_ancoradas": round(100 * dentro / total, 1) if total else None,
    }


def main() -> None:
    raiz = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=raiz / "data/processed/documents.jsonl")
    ap.add_argument("--gen-dir", default=raiz / "eval/results/generation")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    corpus_path, gen_dir = Path(args.corpus), Path(args.gen_dir)
    corpus = urls_do_corpus(corpus_path)
    print(f"URLs distintas no acervo: {len(corpus)}  ({corpus_path})\n")

    linhas = [analisar(p, corpus) for p in sorted(gen_dir.glob("judged_*.jsonl"))]
    if not linhas:
        raise SystemExit(f"nenhum judged_*.jsonl em {gen_dir}")

    cab = f"{'config':<24}{'secao':>7}{'c/URL':>7}{'URLs':>7}{'ancor.':>8}{'fora':>7}{'%':>7}"
    print(cab)
    print("-" * len(cab))
    for r in linhas:
        print(
            f"{r['config']:<24}{r['com_secao_fontes']:>7}{r['com_url']:>7}"
            f"{r['urls_citadas']:>7}{r['ancoradas']:>8}{r['nao_ancoradas']:>7}"
            f"{(r['pct_ancoradas'] if r['pct_ancoradas'] is not None else 0):>7}"
        )

    destino = Path(args.out) if args.out else gen_dir / "citation_grounding.csv"
    with destino.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)
    print(f"\n✓ {destino}")


if __name__ == "__main__":
    main()
