import sys
from shared_functions import (
    load_food_data,
    create_similarity_search_collection,
    populate_similarity_collection,
    perform_similarity_search
)
from enhanced_rag_chatbot import generate_llm_rag_response

if len(sys.argv) < 2:
    print("Usage: python ask.py \"<your query here>\"")
    sys.exit(1)

query = sys.argv[1]

# Setup DB & retrieve
food_items = load_food_data('./FoodDataSet.json')
collection = create_similarity_search_collection("cli_query_food")
populate_similarity_collection(collection, food_items)

print(f"\nQuery: {query}")
results = perform_similarity_search(collection, query, n_results=3)
print("\nGenerating AI recommendation...")
reply = generate_llm_rag_response(query, results)
print(f"\nBot:\n{reply}\n")