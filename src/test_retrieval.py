import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient

load_dotenv()
# Configurações e constantes
COLLECTION_NAME = "niar_rag_documents"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")

# Normaliza um vetor para ter norma 1 (unitário)
def normalize(vec):
    v = np.array(vec)
    norm = np.linalg.norm(v)

    if norm == 0:
        return v.tolist()

    return (v / norm).tolist()

# Gera o embedding para a query usando a API Gemini
def embed_query(query):
    client = genai.Client(api_key=GOOGLE_GENAI_API_KEY)

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        ),
    )

    return normalize(response.embeddings[0].values)

# Testa a função de recuperação consultando o Qdrant com a query e exibindo os resultados
def test_retrieval(query, limit=4):
    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    query_vector = embed_query(query)

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
    )

    print(f"\nPergunta: {query}")
    print("=" * 80)

    if not results.points:
        print("Nenhum resultado encontrado.")
        return

    for idx, point in enumerate(results.points, start=1):
        payload = point.payload

        print(f"\nResultado {idx}")
        print(f"Score: {point.score}")
        print(f"Título: {payload.get('title')}")
        print(f"Fonte: {payload.get('fonte')}")
        print(f"Página: {payload.get('page')}")
        print(f"Tipo: {payload.get('document_type')}")
        print(f"Tema: {payload.get('theme')}")
        print("\nTrecho:")
        print(payload.get("texto", "")[:1000])
        print("-" * 80)


if __name__ == "__main__":
    test_retrieval("O que é telemedicina segundo a resolução do CFM?")
    test_retrieval("Quais são os princípios de IA responsável?")
    test_retrieval("O PL 2338 fala sobre sistemas de alto risco?")