# Food RAG Chatbot

A food recommendation system using vector similarity search (ChromaDB) with RAG (Retrieval-Augmented Generation) powered by IBM WatsonX Granite model.

## Features

- **Vector Search**: Semantic food search using SentenceTransformers embeddings
- **RAG Pipeline**: Retrieves relevant food items and generates natural language recommendations
- **Multiple Interfaces**: CLI, Flask web app, Streamlit chatbot, and interactive terminal
- **Filters**: Search by cuisine type, calorie limits, ingredients
- **185 food items** across various cuisines

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Options

**1. Streamlit Chatbot (Recommended)**
```bash
streamlit run streamlit_app.py
```
Opens at http://localhost:8501

**2. Flask Web App**
```bash
python flask_app.py
```
Opens at http://localhost:5000

**3. CLI Query**
```bash
python ask.py "healthy low calorie dinner"
```

**4. Interactive Terminal Chatbot**
```bash
python interactive_food_chatbot.py
```

**5. Batch Tests**
```bash
python batch_test.py
```

## Configuration

### IBM WatsonX (Optional)
For AI-generated responses, set credentials in `enhanced_rag_chatbot_watsonx.py`:

```python
my_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "YOUR_API_KEY"
}
project_id = "YOUR_PROJECT_ID"
```

Without credentials, falls back to template-based responses.

## Project Structure

```
├── shared_functions.py          # Core vector DB & search functions
├── enhanced_rag_chatbot.py      # RAG response generation (fallback)
├── enhanced_rag_chatbot_watsonx.py  # WatsonX integration
├── streamlit_app.py             # Streamlit chatbot UI
├── flask_app.py                 # Flask web app
├── ask.py                       # CLI single-query tool
├── interactive_food_chatbot.py  # Interactive terminal chatbot
├── batch_test.py                # Automated test suite
├── food_recommendation.py       # Core recommendation demo
├── FoodDataSet.json             # Food dataset (185 items)
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── .streamlit/config.toml       # Streamlit config
```

## Food Dataset Fields

- `food_id`, `food_name`, `food_description`
- `food_calories_per_serving`, `food_ingredients`
- `food_nutritional_factors`, `food_health_benefits`
- `cooking_method`, `cuisine_type`, `food_features`

## Deployment

### Streamlit Cloud
1. Push to GitHub
2. Connect repo at share.streamlit.io
3. Set main file: `streamlit_app.py`

### Docker
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Heroku / Render / Railway
- Add `Procfile`: `web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
- Ensure `FoodDataSet.json` is in repo (or download at runtime)

## License

MIT