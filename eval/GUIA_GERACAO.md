# Guia — rodar a avaliação de geração (cenário B, denso)

Duas pessoas dividem o trabalho: **uma roda com ICL, outra sem ICL.** Cada uma usa
a própria conta da Groq, porque a cota diária não dá para os dois conjuntos.

---

## O que o código faz, em uma página

O pipeline tem duas etapas. A **recuperação já foi rodada** e está pronta no
repositório: para cada uma das 100 perguntas do *gold set*, ela já registrou quais
100 trechos do acervo o buscador considerou mais relevantes. Esse resultado está
salvo em `eval/results/retrieval/rankings.json` — **ninguém precisa rodar busca de
novo, e nada disso gasta API.**

O que vocês vão rodar é a **geração**. Para cada pergunta, o script:

1. pega os **5 primeiros trechos** que a recuperação já havia selecionado;
2. monta um *prompt* com esses trechos + a pergunta e envia ao **Qwen** (Groq),
   que escreve a resposta citando as fontes — ou recusa, se o contexto não
   sustentar;
3. envia pergunta + contexto + resposta a **dois juízes-LLM independentes**
   (Llama-3.3-70B e gpt-oss-120B), que dão notas de 1 a 5 em sete critérios;
4. grava o resultado e passa para a próxima pergunta.

São 100 perguntas × 3 chamadas de API cada (1 geração + 2 juízes).

**A diferença entre as duas variantes é só uma:** com ICL (*in-context learning*),
o *prompt* do gerador inclui 3 exemplos de respostas bem-feitas antes da pergunta.
Sem ICL, não inclui. Tudo o mais é idêntico — mesmas perguntas, mesmos trechos,
mesmos juízes. É isso que permite atribuir qualquer diferença ao ICL.

### Salvamento e retomada — o mais importante

A cota diária **vai acabar antes das 100 perguntas.** Isso é esperado.

O script grava cada pergunta assim que ela termina, num arquivo de checkpoint. Se
a cota estourar (ou o computador desligar), **basta rodar o mesmo comando de
novo**: ele pula o que já foi feito e continua de onde parou, mostrando algo como
`↻ retomando: 58/100 perguntas já concluídas`.

Um erro `RuntimeError` no fim da execução, depois de umas retentativas, **é o
sinal normal de cota esgotada** — não é bug, e o progresso já está salvo.

---

## Preparação (uma vez só)

```bash
git pull
```

```bash
pip install -r eval/requirements-eval.txt
```

Crie um arquivo `.env` na raiz do projeto com as suas chaves:

```
GROQ_API_KEY=sua-chave-aqui
GOOGLE_GENAI_API_KEY=sua-chave-aqui
QDRANT_URL=...
QDRANT_API_KEY=...
```

> **Não edite o `eval/config.yaml`.** As duas pessoas precisam usar exatamente a
> mesma configuração, senão os resultados não são comparáveis e o trabalho é
> perdido.

---

## Teste antes de gastar cota

Roda 2 perguntas, sem juízes. Serve para confirmar que as chaves funcionam:

```bash
cd eval/generation_eval && python scenario_B_dense.py --limit 2 --no-judge --icl 0
```

Confira se as respostas saíram bem formadas:

```bash
cd eval && python tools/check_answers.py
```

Precisa dizer `✓ tudo limpo`. Se aparecer `com <think> aberto`, **pare e avise** —
é um defeito conhecido do gerador e os dados não prestam.

Apague os arquivos de teste antes de valer:

```bash
rm eval/results/generation/judged_B_dense_gemini_icl0.jsonl eval/results/generation/scenario_B_dense_gemini_icl0.csv
```

---

## O comando de cada pessoa

**Pessoa 1 — SEM ICL:**

```bash
cd eval/generation_eval && python scenario_B_dense.py --icl 0
```

**Pessoa 2 — COM ICL:**

```bash
cd eval/generation_eval && python scenario_B_dense.py --icl 1
```

Rode o mesmo comando todo dia até chegar a 100/100. São cerca de 2 a 3 dias.

### Acompanhar o progresso

O CSV é reescrito a cada pergunta — o número de linhas é o progresso. Ele fica em
`eval/results/generation/scenario_B_dense_gemini_icl0.csv` (ou `_icl1`).

Pode abrir no Excel durante a execução; se o arquivo estiver aberto na hora de
gravar, o script ignora aquela gravação e segue (não quebra).

---

## Ao terminar

1. Rode a verificação final:

```bash
cd eval && python tools/check_answers.py
```

2. Envie os **dois arquivos** da sua variante:
   - `eval/results/generation/judged_B_dense_gemini_icl{0,1}.jsonl`
   - `eval/results/generation/scenario_B_dense_gemini_icl{0,1}.csv`

Os arquivos das duas pessoas precisam ser reunidos na mesma pasta
`eval/results/generation/` antes de rodar as etapas de concordância e relatório.

---

## Se algo der errado

| sintoma | o que é |
|---|---|
| `RuntimeError` após ~45 s de retentativas | cota diária acabou — normal, rode de novo amanhã |
| `ModuleNotFoundError` | falta `pip install -r eval/requirements-eval.txt` |
| `! ranking B_dense_gemini ausente` | faltou `git pull` |
| `check_answers` acusa `<think>` | avise antes de continuar; os dados saem inutilizáveis |
| respostas com ~8.000 caracteres | mesmo problema acima |
