"""
Milestone 5 interface: a Gradio web UI over the grounded RAG pipeline.

Run:
    python app.py
then open http://localhost:7860
"""

from __future__ import annotations

import gradio as gr

from generate import ask

EXAMPLES = [
    "What order should I list my work experience in on a resume?",
    "How should I describe my accomplishments in resume bullet points?",
    "Should I include 'References available upon request' on my resume?",
    "What are the main parts of a cover letter?",
    "How long should a resume be for a student or recent graduate?",
]


def handle_query(question: str):
    if not question or not question.strip():
        return "Enter a question above.", ""
    result = ask(question)
    sources = "\n".join(f"• {s}" for s in result["sources"]) or "(no sources cited)"
    return result["answer"], sources


with gr.Blocks(title="The Unofficial Guide — Resume & Cover Letter RAG") as demo:
    gr.Markdown(
        "# The Unofficial Guide\n"
        "Ask about resumes and cover letters. Answers are grounded in 10 "
        "university career guides; the system declines when the guides don't "
        "cover your question."
    )
    inp = gr.Textbox(label="Your question", placeholder="e.g. How long should a student resume be?")
    btn = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)
    gr.Examples(examples=EXAMPLES, inputs=inp)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
