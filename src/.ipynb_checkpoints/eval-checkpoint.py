from langchain_openai import ChatOpenAI
from rag_chain import get_rag_answer

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

EVAL_QUESTIONS = [
    "What is attendance requirement?",
    "How many marks is final exam?",
]

def answer_without_rag(q):
    return llm.invoke(q).content

def run_eval():
    for q in EVAL_QUESTIONS:
        no_rag = answer_without_rag(q)
        rag, _ = get_rag_answer(q)

        print("\nQ:", q)
        print("\nNo RAG:", no_rag)
        print("\nWith RAG:", rag)
        print("="*40)

if __name__ == "__main__":
    run_eval()
