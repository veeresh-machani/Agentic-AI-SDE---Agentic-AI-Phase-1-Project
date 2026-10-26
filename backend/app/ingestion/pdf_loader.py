from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(pdf_path: Path) -> list[Document]:
    """
    Load a PDF page-by-page using PyPDFLoader.

    Each returned Document contains:
        - page_content: extracted text
        - metadata["source"]: PDF path
        - metadata["page"]: zero-based page number
    """

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path.suffix}")

    loader = PyPDFLoader(str(pdf_path))

    documents = loader.load()

    return documents


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def load_all_pdfs(documents_dir: str | Path | None = None) -> list[Document]:
    """
    Load every PDF from the documents directory.
    Defaults to the project root's data/documents directory.
    """

    if documents_dir is None:
        documents_path = DEFAULT_DOCUMENTS_DIR
    else:
        documents_path = Path(documents_dir)
        if not documents_path.is_absolute() and not documents_path.exists():
            documents_path = PROJECT_ROOT / documents_dir

    if not documents_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_path}")

    pdf_files = sorted(documents_path.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {documents_path}")

    all_documents = []

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        documents = load_pdf(pdf_file)

        all_documents.extend(documents)

        print(f"  Pages loaded: {len(documents)}")

    return all_documents


if __name__ == "__main__":
    documents = load_all_pdfs()

    print("\n" + "=" * 60)
    print("PDF LOADING COMPLETE")
    print("=" * 60)

    print(f"Total pages loaded: {len(documents)}")

    print("\nDocuments:")

    sources = {}

    for document in documents:
        source = Path(document.metadata["source"]).name
        sources[source] = sources.get(source, 0) + 1

    for source, page_count in sources.items():
        print(f"  {source}: {page_count} pages")

    print("\nFirst page preview:")

    if documents:
        document = documents[0]

        print(f"Source: {Path(document.metadata['source']).name}")
        print(f"Page: {document.metadata['page'] + 1}")
        print("-" * 60)
        print(document.page_content[:1000])
