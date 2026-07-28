import os
import gradio as gr
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="philschmid/MiniLM-L6-H384-uncased-sst2"
)

def analyze_sentiment(text):
    # Handle empty input
    if not text.strip():
        return "Please enter some text.", ""

    result = classifier(text)

    label = result[0]["label"]
    score = f"{result[0]['score']:.2f}"

    return label, score

app = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(
        label="Enter a Sentence",
        lines=3,
        placeholder="Type a sentence here..."
    ),
    outputs=[
        gr.Textbox(label="Sentiment"),
        gr.Textbox(label="Score")
    ],
    title="Sentiment Analyzer by Surya :D",
    description="Enter a sentence to analyze its sentiment."
)

app.launch(server_name="0.0.0.0", server_port=int(OS.environ.get("PORT",7860)))
