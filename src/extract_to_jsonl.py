import csv
import json
from pathlib import Path

import fitz
from tqdm import tqdm


INPUT_DIR = Path("docs/raw")
MANIFEST_FILE = Path("corpus_manifest.csv")
OUTPUT_FILE = Path("data/processed/documents.jsonl")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def load_manifest():
    metadata_by_file = {}

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            metadata_by_file[row["filename"]] = row

    return metadata_by_file


def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = " ".join(text.split())
    return text


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def process_pdfs():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    pdf_files = list(INPUT_DIR.glob("*.pdf"))

    total_chunks = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pdf_file in tqdm(pdf_files, desc="Processando PDFs"):
            doc_metadata = manifest.get(pdf_file.name, {})

            with fitz.open(pdf_file) as doc:
                for page_index, page in enumerate(doc, start=1):
                    text = clean_text(page.get_text())

                    if not text:
                        continue

                    chunks = chunk_text(text)

                    for chunk_index, chunk in enumerate(chunks):
                        data = {
                            "id": f"{pdf_file.stem}_p{page_index}_c{chunk_index}",
                            "text": chunk,
                            "metadata": {
                                "source": pdf_file.name,
                                "title": doc_metadata.get("title", pdf_file.stem),
                                "page": page_index,
                                "chunk": chunk_index,
                                "document_type": doc_metadata.get("document_type", ""),
                                "year": doc_metadata.get("year", ""),
                                "theme": doc_metadata.get("theme", ""),
                                "source_url": doc_metadata.get("source_url", "")
                            }
                        }

                        f.write(json.dumps(data, ensure_ascii=False) + "\n")
                        total_chunks += 1

    print("PROCESSAMENTO FINALIZADO")
    print(f"Total de chunks: {total_chunks}")
    print(f"Arquivo salvo em: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_pdfs()