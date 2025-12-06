import gradio as gr
import requests
from rag import RAGEngine
from config import GROQ_API_KEY

rag = RAGEngine()

def ask_bot(question):
    retrieved = rag.retrieve(question)
    context = "\n\n".join(retrieved)

    prompt = f"""
You are a statistics teaching assistant.
Answer the student's question using ONLY the information below.

Context:
{context}

Question: {question}
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": "llama-3.1-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
        },
    )

    content = response.json()["choices"][0]["message"]["content"]
    return content + "\n\nSources:\n" + "\n".join(retrieved)

demo = gr.ChatInterface(
    ask_bot,
    title="StatBot",
    description="Ask questions about the lecture material.",
)

if __name__ == "__main__":
    demo.launch()
