# LEME — assistente RAG para normas de saúde e IA Responsável

O LEME é um chatbot em português que responde perguntas sobre legislação, normas e diretrizes de IA Responsável em saúde, brasileiras e internacionais. Ele busca as respostas em um acervo curado de 63 fontes oficiais, bilíngue (PT/EN), e cita a fonte de cada resposta. Quando o acervo não sustenta uma resposta, ele recusa.

Público-alvo: profissionais de saúde, auditores de IA Responsável e desenvolvedores de sistemas de IA, sem exigir formação jurídica.

> Artigo: *LEME: assistente RAG em português para normas brasileiras e internacionais sobre saúde e IA Responsável* — WebMedia 2026, Workshop de Ferramentas e Aplicações (WFA).

---

## Arquitetura

```
Front-end ──► serviço LangGraph (agent/) ──► ferramenta de recuperação ──► Qdrant
                 │  nós: boas-vindas, chat,        (embedding Gemini)         ▲
                 │       ferramentas                                          │
                 ▼                                                  indexação offline
            Qwen3.6-27B (Groq, temperatura 0)                       (src/), acervo em
            resposta com fontes ou recusa                           docs/raw + data/
```

| Componente | Onde está | Tecnologia |
|---|---|---|
| Agente (grafo de estados) | `agent/agent.py` | LangGraph |
| Prompt de sistema do gerador | `agent/utils/prompt.py` | — |
| Ferramenta de recuperação | `agent/utils/tools.py` | `gemini-embedding-001` + Qdrant, similaridade de cosseno |
| Gerador | `agent/agent.py` | `qwen/qwen3.6-27b` via Groq, `temperature=0` |
| Extração e recorte do acervo | `src/` | PyMuPDF (PDF), BeautifulSoup + Selenium (web) |
| Protótipo de interface | `app.py` | Streamlit |
| Avaliação | `eval/` | ver [`eval/README.md`](eval/README.md) |

---

## Instalação

Requer Python 3.10+ e contas na Groq, no Google AI (Gemini) e no Qdrant (nuvem ou local).

```bash
git clone https://github.com/italoavelar/niar-rag-prototype.git
cd niar-rag-prototype
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements-agent.txt
```

O `requirements-agent.txt` instala só o necessário para o agente. A avaliação usa
outra pilha (BGE-m3, BM25), em `requirements.txt` + `eval/requirements-eval.txt`,
de preferência num venv separado (a justificativa está no cabeçalho do
`requirements-agent.txt`).

Crie um arquivo `.env` na raiz (ele não é versionado):

```env
GROQ_API_KEY=...
GOOGLE_API_KEY=...          # embedding Gemini (agente)
GOOGLE_GENAI_API_KEY=...    # mesmo valor; usado por src/build_vectorstore.py
QDRANT_URL=...
QDRANT_API_KEY=...
CEREBRAS_API_KEY=...        # opcional: juiz alternativo na avaliação
```

---

## Uso

### 1. Construir o índice (uma vez)

O acervo processado já está versionado em `data/processed/documents.jsonl`
(5.791 trechos). Para enviar os vetores ao Qdrant:

```bash
python src/build_vectorstore.py
```

O script recria a coleção `niar_rag_documents` no Qdrant e reaproveita os vetores de
`data/processed/embeddings_backup.jsonl` (versionado com Git LFS). Ele só chama a API
do Gemini para os trechos que vêm depois do último salvo no backup. Rode
`git lfs pull` antes, se o arquivo vier como ponteiro.

> Se o acervo for regenerado, apague o backup antes: ele é retomado por posição,
> não por conteúdo, e vetores antigos seriam associados aos trechos novos.

Para refazer o acervo a partir dos documentos originais:

```bash
python src/extract_to_jsonl.py        # PDFs de docs/raw → data/processed/pdf_chunks.jsonl
python src/extract_html_to_jsonl.py   # páginas coletadas → data/processed/html_chunks.jsonl
python src/merge_corpus_jsonl.py      # une os dois → data/processed/documents.jsonl
python src/validate_jsonl.py
python src/build_vectorstore.py
```

A coleta das páginas web está em `notebooks/web_scraper_estatico_profundo_v6.ipynb`.
As fontes e seus metadados estão em `corpus_manifest.csv` (arquivos do acervo) e em
`corpus_documentos.csv` (lista das 63 fontes, com URL de origem).

### 2. Subir o agente

```bash
langgraph dev
```

O servidor do LangGraph expõe o grafo `agent` (definido em `langgraph.json`), e o
front-end conversa só com esse serviço. Para testar sem front-end, use o LangGraph
Studio, que abre junto com o `langgraph dev`, ou o protótipo em Streamlit:

```bash
streamlit run app.py
```

---

## Reproduzir a avaliação

O protocolo, o *gold set* (100 perguntas: 25 factuais, 25 multi-hop, 25 comparativas
e 25 fora de escopo), os julgamentos de relevância e todos os resultados do artigo
estão em `eval/`. Passo a passo em [`eval/README.md`](eval/README.md).

| Artefato do artigo | Arquivo |
|---|---|
| Gold set e qrels | `eval/data/golden_qa.jsonl`, `eval/data/qrels.csv` |
| Tabela 1 (recuperação) | `eval/results/retrieval/metrics.csv`, `eval/results/report.md` |
| Tabela 2 (geração) e juízes | `eval/results/generation/judged_*.jsonl`, `eval/results/report.md` |
| Concordância entre juízes | `eval/results/agreement/agreement.csv` |
| Contraste com vs. sem RAG por juiz | `eval/tools/contraste_por_juiz.py` |
| Ancoragem das citações | `eval/results/generation/citation_grounding.csv` |
| Prompt dos juízes | `eval/lib/judges.py` |
| Configuração completa (modelos, k, sementes) | `eval/config.yaml` |

---

## Licenças

- **Código:** MIT — ver [`LICENSE`](LICENSE).
- **Gold set, qrels e resultados de avaliação:** CC BY 4.0 — ver
  [`eval/data/LICENSE-DATA`](eval/data/LICENSE-DATA).
- **Acervo:** documentos normativos de acesso público. Os direitos sobre os textos
  originais permanecem com seus emissores; cada trecho guarda a URL de origem em
  `metadata.source_url`. As licenças acima cobrem a curadoria, a segmentação e os
  metadados produzidos neste trabalho.

O LEME é uma ferramenta de apoio à consulta. Não substitui aconselhamento jurídico
nem orientação clínica.

## Como citar

```bibtex
[referência do artigo — preencher após a publicação nos Anais Estendidos do WebMedia 2026]
```
