from flask import Flask, request, render_template_string
from shared_functions import (
    load_food_data,
    create_similarity_search_collection,
    populate_similarity_collection,
    perform_similarity_search
)
from enhanced_rag_chatbot import generate_llm_rag_response

app = Flask(__name__)

# Setup DB on launch
print("Loading database...")
data = load_food_data('./FoodDataSet.json')
collection = create_similarity_search_collection("flask_food_chatbot")
populate_similarity_collection(collection, data)
print("Ready!")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Food RAG Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }
        input[type="text"] { width: 75%; padding: 10px; font-size: 16px; }
        button { padding: 10px 16px; font-size: 16px; cursor: pointer; }
        .box { background: #f4f4f4; padding: 15px; border-radius: 8px; margin-top: 20px; }
    </style>
</head>
<body>
    <h2>🍲 Food RAG Chatbot</h2>
    <form method="POST">
        <input type="text" name="query" placeholder="Ask for food recommendations..." value="{{ query }}" required>
        <button type="submit">Send</button>
    </form>

    {% if response %}
    <div class="box">
        <h3>Recommendation:</h3>
        <p style="white-space: pre-line;">{{ response }}</p>
    </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def chat():
    query = ""
    response = ""
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            results = perform_similarity_search(collection, query, n_results=3)
            if results:
                response = generate_llm_rag_response(query, results)
            else:
                response = "No matching food items found."
    return render_template_string(HTML_TEMPLATE, query=query, response=response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)