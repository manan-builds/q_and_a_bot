from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

CHROMA_PATH = "chroma_store"
COLLECTION_NAME = "week20_domain_qa"

def format_docs(docs):
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "NA")
        chunk_id = d.metadata.get("chunk_id", "NA")

        blocks.append(
            f"[Source: {src} | Page: {page} | Chunk: {chunk_id}]\n{d.page_content}"
        )

    return "\n\n".join(blocks)

def simple_confidence(question: str, docs):
    q_words = [w.lower() for w in question.split() if len(w) > 3]
    joined = " ".join([d.page_content.lower() for d in docs])
    hits = sum(1 for w in q_words if w in joined)

    score = hits / max(1, len(q_words))
    return score

def get_rag_answer(question: str, top_k=5):
    embeddings = OpenAIEmbeddings()

    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

    retriever = db.as_retriever(search_kwargs={"k": top_k})

    docs = retriever.invoke(question)
    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer ONLY using the context. If not found, say 'I don't know.' "
         "List sources at the end."),
        ("user",
         "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer, docs
