import csv
from collections import Counter
from pathlib import Path


PDF_DIR = Path("docs/raw")
MANIFEST_FILE = Path("corpus_manifest.csv")


def load_manifest_filenames() -> list[str]:
    with open(MANIFEST_FILE, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if "filename" not in (reader.fieldnames or []):
            raise ValueError(
                "A coluna 'filename' não foi encontrada no corpus_manifest.csv."
            )

        return [
            row["filename"].strip()
            for row in reader
            if row.get("filename", "").strip()
        ]


def validate_corpus() -> None:
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {PDF_DIR}")

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Manifesto não encontrado: {MANIFEST_FILE}")

    pdf_filenames = sorted(
        path.name
        for path in PDF_DIR.glob("*.pdf")
        if path.is_file()
    )

    manifest_filenames = load_manifest_filenames()

    pdf_set = set(pdf_filenames)
    manifest_set = set(manifest_filenames)

    pdfs_without_metadata = sorted(pdf_set - manifest_set)
    metadata_without_pdf = sorted(manifest_set - pdf_set)

    duplicate_filenames = sorted(
        filename
        for filename, count in Counter(manifest_filenames).items()
        if count > 1
    )

    print("=" * 72)
    print("VALIDAÇÃO DO CORPUS")
    print("=" * 72)
    print(f"PDFs encontrados: {len(pdf_filenames)}")
    print(f"Registros no manifesto: {len(manifest_filenames)}")
    print(f"PDFs sem metadados: {len(pdfs_without_metadata)}")
    print(f"Metadados sem PDF: {len(metadata_without_pdf)}")
    print(f"Filenames duplicados: {len(duplicate_filenames)}")

    if pdfs_without_metadata:
        print("\nPDFs sem registro no manifesto:")
        for filename in pdfs_without_metadata:
            print(f"- {filename}")

    if metadata_without_pdf:
        print("\nRegistros do manifesto sem PDF correspondente:")
        for filename in metadata_without_pdf:
            print(f"- {filename}")

    if duplicate_filenames:
        print("\nFilenames duplicados no manifesto:")
        for filename in duplicate_filenames:
            print(f"- {filename}")

    if (
        not pdfs_without_metadata
        and not metadata_without_pdf
        and not duplicate_filenames
    ):
        print("\nCORPUS VALIDADO COM SUCESSO")
    else:
        raise ValueError(
            "Foram encontrados problemas entre os PDFs e o manifesto."
        )


if __name__ == "__main__":
    validate_corpus()