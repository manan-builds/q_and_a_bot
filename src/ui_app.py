import gradio as gr
from rag_chain import get_rag_answer, simple_confidence

def chat_fn(user_question):
    answer, docs = get_rag_answer(user_question)
    conf = simple_confidence(user_question, docs)

    if conf < 0.15:
        return "Low confidence. Try rephrasing."

    return f"{answer}\n\nConfidence: {conf:.2f}"

demo = gr.Interface(
    fn=chat_fn,
    inputs=gr.Textbox(lines=2),
    outputs="text",
    title="Domain Q&A Bot"
)

if __name__ == "__main__":
    demo.launch()
