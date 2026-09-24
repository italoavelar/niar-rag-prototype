# Pipeline de avaliação do LEME

Avalia o LEME de ponta a ponta. A **recuperação** é medida nos Cenários A/B/C; a
**geração**, com e sem RAG e com e sem ICL, é pontuada por 2 juízes-LLM. O
pipeline reporta métricas robustas e significância estatística. É o protocolo
descrito na Seção 3 do artigo do WFA/WebMedia 2026.

## Visão geral

```
gold set ancorado  ──►  recuperação A/B/C (02)  ──►  geração × ICL (generation_eval/)
   │  100 perguntas         nDCG, GeoRisk,            Qwen3.6-27B + 2 juízes
   │  + qrels + respostas   Recall, MRR                       │
   │    de referência                                         ▼
   └──────────────────────────────────────────►  concordância entre juízes (04)

```

| Etapa | Script | O que faz | Métricas |
|------|--------|-----------|----------|
| 01 | `tools/build_gold_manual.py` | Integra o gold set e valida cada `chunk_id` dos qrels contra o corpus | — |
| 02 | `02_retrieval_eval.py` | **A** BM25, sem e com tradução PT→EN · **B** denso (Gemini, BGE-m3 denso/multi-vetor/híbrido) · **C** fusão RRF | **nDCG@5** (principal), P@5, Recall@5/@100, MRR@10, GeoRisk, por idioma |
| 03 | `generation_eval/run_all.py` | Gera as respostas nos cenários A/B/C × {sem, com ICL} e sem RAG; 2 juízes pontuam 7 critérios | fidelidade, alucinação, relevância, correção, completude, citação, recusa |
| 04 | `04_agreement.py` | Concordância entre os juízes | κ ponderado, α de Krippendorff, Spearman, % exata e ±1 |

## Decisões do protocolo

- **Gold set:** 100 perguntas na linguagem da persona — 25 factuais, 25 multi-hop, 25 comparativas e 25 fora de escopo. Foram escritas por uma LLM independente (Claude, em sessão) a partir de trechos reais do corpus e integradas por `tools/build_gold_manual.py`. Ancoram-se em 13 dos 63 documentos do acervo; os demais continuam no índice de busca. Um especialista jurídico auditou 50 perguntas. O caminho automatizado `01_build_golden.py` (bloco `golden` do `config.yaml`) é só um fallback e **não** produziu o gold set em uso.
- **Qrels por construção:** grau 2 para o trecho que originou a pergunta, grau 1 para os vizinhos no mesmo documento, 0 para o resto. As métricas de recuperação usam só as 75 perguntas respondíveis; as 25 fora de escopo têm qrels vazios e entram apenas na geração, onde medem a recusa.
- **Sem dataset público:** os benchmarks PT-BR levantados não cobrem o domínio nem a tarefa — ver [`data/README.md`](data/README.md).
- **Cenário B:** compara o **Gemini** (`gemini-embedding-001`, proprietário, usado em produção) com o **BGE-m3** (aberto, auto-hospedável). A pergunta é se dá para trocar a API paga sem perder qualidade: não-inferioridade com margem de −0,03 em nDCG@5, definida a priori.
- **Corpus bilíngue (PT + EN):** as métricas são reportadas também por idioma do trecho-fonte. Como as perguntas são sempre em PT, a coluna EN mede recuperação cross-lingual.
- **Gerador:** `qwen/qwen3.6-27b` via Groq, temperatura 0, `reasoning_effort: none`, com os 5 primeiros trechos no contexto (`generation.context_top_k`).
- **Juízes:** Llama-3.3-70B e gpt-oss-120B, ambos diferentes do gerador, para evitar
  viés de auto-preferência. A nota reportada é a média dos dois.
- **ICL:** few-shot com 3 exemplos (`data/icl_examples.json`), só na geração.

## Instalação

```bash
# a partir da raiz do repositório, com o venv de avaliação ativo
pip install -r requirements.txt -r eval/requirements-eval.txt
```

Variáveis no `.env` da raiz: `GROQ_API_KEY`, `GOOGLE_API_KEY` (o código aceita o
apelido `GOOGLE_GENAI_API_KEY`), `QDRANT_URL` e `QDRANT_API_KEY`. `CEREBRAS_API_KEY`
é opcional, para um juiz alternativo.

## Como rodar

```bash
# 0) (uma vez) teste o núcleo de métricas — não usa rede
python eval/lib/metrics.py

# 1) gold set — JÁ VERSIONADO em data/golden_qa.jsonl (é a versão canônica).
#    Só para regenerá-lo: o builder recria os qrels ANTERIORES à correção de
#    q0015/q0016, então a correção precisa ser reaplicada logo em seguida.
python eval/tools/build_gold_manual.py
python eval/tools/corrige_qrels_q0015_q0016.py

# 2) recuperação (A/B/C)   --limit para depurar; --rebuild recalcula os caches densos
python eval/02_retrieval_eval.py

# 3) geração + juízes (custa API — comece com --limit)
python eval/generation_eval/run_all.py --limit 30
python eval/generation_eval/run_all.py
python eval/generation_eval/run_all.py --scenarios B no_rag   # só alguns cenários

# 4) concordância entre juízes
python eval/04_agreement.py


```

Análises complementares citadas no artigo:

```bash
python eval/tools/citation_grounding.py   # URLs citadas que pertencem ao acervo
python eval/tools/contraste_por_juiz.py   # com vs. sem RAG, testado por juiz
```

A cota diária gratuita da Groq não cobre uma rodada completa de geração.
[`GUIA_GERACAO.md`](GUIA_GERACAO.md) mostra como dividir a rodada entre duas contas.

### Embedding aberto (BGE-m3) em CPU

O BGE-m3 (~560M parâmetros) roda em **CPU**. Na 1ª execução do `02` ele embute o
corpus localmente e guarda os vetores em `results/indexes/dense_bge_m3_<N>.npz`;
só recalcula com `--rebuild`.

## Onde estão os resultados do artigo

| No artigo | Arquivo |
|---|---|
| Tabela 1 e diferença BGE-m3 − Gemini | `results/retrieval/metrics.csv`, `results/report.md` §1 |
| Tabela 2 e contraste com vs. sem RAG | `results/generation/judged_*.jsonl`, `results/report.md` §2 e §4 |
| Concordância entre juízes (κ) | `results/agreement/agreement.csv`, `results/report.md` §3 |
| Ancoragem das URLs citadas | `results/generation/citation_grounding.csv` |

## Notas

1. **Corpus avaliado:** 63 fontes e 5.791 trechos (`data/processed/documents.jsonl`).
   Se o corpus mudar, reindexe o Qdrant (`src/build_vectorstore.py`) e **regenere os
   qrels**: os `chunk_id` mudam com o recorte.
2. **Sem limiar de similaridade na avaliação:** A e B são buscas independentes no
   corpus inteiro, cada uma com top-`retrieval.top_k` (100), para que Recall e nDCG
   façam sentido. O `score_threshold` de `agent/utils/tools.py` não é aplicado aqui.

3. **Correção de q0015/q0016:** o gabarito dessas duas perguntas foi corrigido por
   `tools/corrige_qrels_q0015_q0016.py`. As duas tinham 91 e 100 trechos marcados
   como relevantes, efeito de fragmentos quase idênticos gerados pelo recorte. Os
   resultados versionados já incorporam a correção.

Tudo é dirigido por [`config.yaml`](config.yaml) — modelos, k, cenários, ICL, juízes,
α e semente — para garantir a reprodutibilidade.
