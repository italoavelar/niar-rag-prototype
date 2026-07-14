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


def parse_ria_dimensions(value: str) -> list[str]:
    """
    Converte o campo ria_dimension do CSV em uma lista.

    Exemplo:
    "Privacidade e Segurança; Transparência; Governança"

    Retorno:
    [
        "Privacidade e Segurança",
        "Transparência",
        "Governança"
    ]
    """
    if not value:
        return []

    return [
        dimension.strip()
        for dimension in value.split(";")
        if dimension.strip()
    ]


def load_manifest() -> dict[str, dict]:
    """
    Carrega os metadados do corpus_manifest.csv.

    O dicionário retornado utiliza o nome do arquivo PDF como chave.
    """
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifesto não encontrado: {MANIFEST_FILE}"
        )

    metadata_by_file = {}

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "filename",
            "title",
            "document_type",
            "year",
            "theme",
            "source_url",
        }

        available_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - available_columns

        if missing_columns:
            raise ValueError(
                "Colunas obrigatórias ausentes no manifesto: "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(reader, start=2):
            filename = row.get("filename", "").strip()

            if not filename:
                print(
                    f"[AVISO] Linha {row_number} ignorada: "
                    "filename vazio."
                )
                continue

            if filename in metadata_by_file:
                raise ValueError(
                    f"Filename duplicado no manifesto: {filename}"
                )

            row["filename"] = filename
            row["ria_dimensions"] = parse_ria_dimensions(
                row.get("ria_dimension", "")
            )

            metadata_by_file[filename] = row

    return metadata_by_file


def clean_text(text: str) -> str:
    """
    Realiza apenas a limpeza básica do texto.

    A limpeza avançada de cabeçalhos, rodapés, sumários,
    agradecimentos e referências será adicionada posteriormente.
    """
    text = text.replace("\t", " ")
    text = text.replace("\n", " ")
    text = " ".join(text.split())

    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Divide o texto em chunks com sobreposição.

    Esta é a estratégia atual. O chunking por parágrafos
    será implementado em uma próxima etapa.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser maior que zero.")

    if overlap < 0:
        raise ValueError("overlap não pode ser negativo.")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap deve ser menor que chunk_size."
        )

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def build_chunk_record(
    pdf_file: Path,
    doc_metadata: dict,
    page_index: int,
    chunk_index: int,
    chunk: str,
) -> dict:
    """
    Cria o registro JSONL de um chunk com seus metadados.
    """
    return {
        "id": (
            f"{pdf_file.stem}_"
            f"p{page_index}_"
            f"c{chunk_index}"
        ),
        "text": chunk,
        "metadata": {
            "document_id": pdf_file.stem,
            "source": pdf_file.name,
            "source_type": doc_metadata.get(
                "source_type",
                "PDF",
            ),
            "title": doc_metadata.get(
                "title",
                pdf_file.stem,
            ),
            "page": page_index,
            "chunk": chunk_index,
            "document_type": doc_metadata.get(
                "document_type",
                "",
            ),
            "author": doc_metadata.get(
                "author",
                "",
            ),
            "year": doc_metadata.get(
                "year",
                "",
            ),
            "theme": doc_metadata.get(
                "theme",
                "",
            ),
            "ria_dimensions": doc_metadata.get(
                "ria_dimensions",
                [],
            ),
            "source_url": doc_metadata.get(
                "source_url",
                "",
            ),
        },
    }


def process_pdfs() -> None:
    """
    Processa os PDFs do corpus, extrai o texto página por página,
    cria os chunks e salva o resultado em JSONL.
    """
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Pasta de PDFs não encontrada: {INPUT_DIR}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_manifest()

    pdf_files = sorted(
        INPUT_DIR.glob("*.pdf"),
        key=lambda path: path.name.lower(),
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"Nenhum PDF encontrado em: {INPUT_DIR}"
        )

    total_documents = len(pdf_files)
    processed_documents = 0
    failed_documents = 0
    total_pages = 0
    empty_pages = 0
    total_chunks = 0

    documents_without_metadata = []
    documents_without_text = []
    processing_errors = []

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as output_file:
        for pdf_file in tqdm(
            pdf_files,
            desc="Processando PDFs",
        ):
            doc_metadata = manifest.get(pdf_file.name)

            if doc_metadata is None:
                documents_without_metadata.append(
                    pdf_file.name
                )

                print(
                    f"\n[ERRO] PDF sem metadados: "
                    f"{pdf_file.name}"
                )

                failed_documents += 1
                continue

            document_text_found = False

            try:
                with fitz.open(pdf_file) as document:
                    for page_index, page in enumerate(
                        document,
                        start=1,
                    ):
                        total_pages += 1

                        raw_text = page.get_text("text")
                        text = clean_text(raw_text)

                        if not text:
                            empty_pages += 1
                            continue

                        document_text_found = True

                        chunks = chunk_text(text)

                        for chunk_index, chunk in enumerate(
                            chunks
                        ):
                            data = build_chunk_record(
                                pdf_file=pdf_file,
                                doc_metadata=doc_metadata,
                                page_index=page_index,
                                chunk_index=chunk_index,
                                chunk=chunk,
                            )

                            output_file.write(
                                json.dumps(
                                    data,
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )

                            total_chunks += 1

                if document_text_found:
                    processed_documents += 1
                else:
                    documents_without_text.append(
                        pdf_file.name
                    )
                    failed_documents += 1

            except Exception as error:
                failed_documents += 1

                processing_errors.append(
                    {
                        "filename": pdf_file.name,
                        "error": str(error),
                    }
                )

                print(
                    f"\n[ERRO] Falha ao processar "
                    f"{pdf_file.name}: {error}"
                )

    print("\n" + "=" * 72)
    print("PROCESSAMENTO FINALIZADO")
    print("=" * 72)
    print(f"PDFs encontrados: {total_documents}")
    print(f"PDFs processados: {processed_documents}")
    print(f"PDFs com falha: {failed_documents}")
    print(f"Páginas analisadas: {total_pages}")
    print(f"Páginas sem texto: {empty_pages}")
    print(f"Chunks gerados: {total_chunks}")
    print(f"Arquivo salvo em: {OUTPUT_FILE}")

    if documents_without_metadata:
        print("\nPDFs sem metadados:")
        for filename in documents_without_metadata:
            print(f"- {filename}")

    if documents_without_text:
        print("\nPDFs sem texto extraível:")
        for filename in documents_without_text:
            print(f"- {filename}")
        print(
            "Esses documentos podem ser escaneados "
            "e precisar de OCR."
        )

    if processing_errors:
        print("\nErros encontrados:")
        for item in processing_errors:
            print(
                f"- {item['filename']}: "
                f"{item['error']}"
            )


if __name__ == "__main__":
    process_pdfs()