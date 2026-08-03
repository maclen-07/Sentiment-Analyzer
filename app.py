import os
import gradio as gr
from transformers import pipeline

# Model will be loaded only when needed
classifier = None

def analyze_sentiment(text):
    global classifier

    # Load model on first request
    if classifier is None:
        classifier = pipeline(
            "sentiment-analysis",
            model="philschmid/MiniLM-L6-H384-uncased-sst2",
            device=-1
        )

    # Handle empty input
    if not text.strip():
        return "Please enter some text.", ""

    result = classifier(text)

    # Convert LABEL_0/LABEL_1 to readable labels
    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Positive"
    }

    label = label_map.get(result[0]["label"], result[0]["label"])
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

app.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
