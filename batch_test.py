from shared_functions import (
    load_food_data,
    create_similarity_search_collection,
    populate_similarity_collection,
    perform_similarity_search,
    perform_filtered_similarity_search
)
from enhanced_rag_chatbot import generate_llm_rag_response

# 1. Initialize data & ChromaDB collection
food_items = load_food_data('./FoodDataSet.json')
collection = create_similarity_search_collection("test_batch_food_search")
populate_similarity_collection(collection, food_items)

# 2. Automated test queries
test_cases = [
    {"query": "chocolate dessert", "filter_cuisine": None, "max_cal": None},
    {"query": "Italian food", "filter_cuisine": "Italian", "max_cal": 500},
    {"query": "healthy chicken", "filter_cuisine": None, "max_cal": 400},
]

print("\n" + "=" * 60)
print("RUNNING AUTOMATED BATCH TESTS")
print("=" * 60)

for i, test in enumerate(test_cases, 1):
    q = test["query"]
    cuisine = test["filter_cuisine"]
    cal = test["max_cal"]
    
    print(f"\n[Test {i}] Query: '{q}' | Cuisine Filter: {cuisine} | Max Cal: {cal}")
    
    # Retrieve using filtered search if criteria exist, else standard search
    if cuisine or cal:
        results = perform_filtered_similarity_search(
            collection, q, cuisine_filter=cuisine, max_calories=cal, n_results=3
        )
    else:
        results = perform_similarity_search(collection, q, n_results=3)
    
    print(f"-> Retrieved {len(results)} items:")
    for r in results:
        pct = r['similarity_score'] * 100
        print(f"   • {r['food_name']} ({r['cuisine_type']}) - {r['food_calories_per_serving']} cal [{pct:.1f}% match]")
    
    # RAG Response Generation Test
    if results:
        print("-> Generating RAG response via IBM Granite...")
        response = generate_llm_rag_response(q, results)
        print(f"-> Bot Response:\n{response}\n")
    print("-" * 60)