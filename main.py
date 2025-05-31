from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

app = Flask(__name__)

# Load model and tokenizer
model_name = "blaze999/Medical-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

@app.route("/ner", methods=["GET"])
def ner():
    text = request.args.get("text", "")
    if not text:
        return "<h3>Missing 'text' query parameter</h3>", 400

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
    <html>
    <head>
        <title>NER Results</title>
        <style>
            table {{
                border-collapse: collapse;
                width: 50%;
                margin: 20px auto;
 }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <h2 style="text-align:center;">NER Output</h2>
        <table>
            <tr><th>Entity Group</th><th>Word</th></tr>
            {table_rows}
        </table>
    </body>
    </html>
    """
    return html
