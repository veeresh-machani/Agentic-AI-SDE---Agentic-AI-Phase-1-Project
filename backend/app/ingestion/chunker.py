from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.app.ingestion.pdf_loader import load_all_pdfs


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    """
    Split page-level Documents into smaller chunks.

    Metadata such as source and page number is preserved
    automatically by RecursiveCharacterTextSplitter.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    # Add a unique chunk ID while preserving existing metadata.
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = index

    return chunks


if __name__ == "__main__":
    # Load all PDFs from the knowledge base
    documents = load_all_pdfs()

    # Create chunks
    chunks = chunk_documents(documents)

    print("\n" + "=" * 60)
    print("CHUNKING COMPLETE")
    print("=" * 60)

    print(f"Pages loaded : {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    print("\nFirst 5 chunks:")

    for chunk in chunks[:5]:
        print("\n" + "-" * 60)

        print(f"Chunk ID : {chunk.metadata['chunk_id']}")
        print(f"Source   : {chunk.metadata['source']}")
        print(f"Page     : {chunk.metadata['page'] + 1}")
        print(f"Characters: {len(chunk.page_content)}")

        print("-" * 60)

        print(chunk.page_content[:500])
