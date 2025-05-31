from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

app = Flask(__name__)

# Load model and tokenizer
model_name = "blaze999/Medical-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

@app.route("/ner", methods=["POST"])
def ner():
    text = request.form.get("text", "")
    if not text:
        return "<h3>Missing 'text' form parameter</h3>", 400

    results = ner_pipeline(text)
    simplified_results = [
        {
            "entity_group": entity["entity_group"],
            "word": entity["word"]
        }
        for entity in results
    ]

    # Build HTML table
    table_rows = "".join(
        f"<tr><td>{item['entity_group']}</td><td>{item['word']}</td></tr>"
        for item in simplified_results
    )
    html = f"""
    <table>
        <tr><th>Entity Group</th><th>Word</th></tr>
        {table_rows}
    </table>
    """
    return html

@app.route("/", methods=["GET"])
def serve_main_page():
    return render_template("main_page.html")
