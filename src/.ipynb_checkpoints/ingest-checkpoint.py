import os
from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

DOCS_DIR = Path("data/docs")
CHROMA_PATH = "chroma_store"
COLLECTION_NAME = "week20_domain_qa"

def load_pdf_as_documents(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    docs = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": i + 1
                    }
                )
            )
    return docs

def chunk_documents(docs, chunk_size=350, overlap=80):
    chunked = []

    for d in docs:
        words = d.page_content.split()
        start = 0
        chunk_id = 0

        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])

            chunked.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        **d.metadata,
                        "chunk_id": chunk_id
                    }
                )
            )

            chunk_id += 1
            start = end - overlap

    return chunked

def build_vector_db():
    embeddings = OpenAIEmbeddings()

    all_docs = []
    for file in DOCS_DIR.glob("*.pdf"):
        all_docs.extend(load_pdf_as_documents(file))

    chunked_docs = chunk_documents(all_docs)

    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    db.add_documents(chunked_docs)

    print(f"Loaded {len(all_docs)} pages and stored {len(chunked_docs)} chunks.")

if __name__ == "__main__":
    build_vector_db()
