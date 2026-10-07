from shared_functions import *
from typing import List, Dict
from ibm_watsonx_ai.foundation_models import ModelInference

# watsonx.ai setup - REPLACE WITH YOUR CREDENTIALS
# Get API key from: https://cloud.ibm.com/iam/apikeys
my_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "YOUR_IBM_CLOUD_API_KEY_HERE"  # Replace with your API key
}
model_id = 'ibm/granite-4-h-small'
gen_parms = {"max_new_tokens": 400}
project_id = "skills-network"  # Replace with your watsonx.ai project ID

# Initialize model (will fail gracefully if credentials not set)
try:
    model = ModelInference(
        model_id=model_id,
        credentials=my_credentials,
        params=gen_parms,
        project_id=project_id,
        verify=False
    )
except Exception as e:
    print(f"Model initialization failed: {e}")
    model = None

def prepare_context_for_llm(query: str, search_results: List[Dict]) -> str:
    """Format retrieved vector results into a structured prompt context."""
    if not search_results:
        return "No relevant food items found in the database."
    
    context_parts = ["Based on your query, here are the most relevant food options:"]
    for i, result in enumerate(search_results[:3], 1):
        food_context = [
            f"Option {i}: {result['food_name']}",
            f" - Description: {result['food_description']}",
            f" - Cuisine: {result['cuisine_type']}",
            f" - Calories: {result['food_calories_per_serving']} per serving"
        ]
        if result.get('food_ingredients'):
            food_context.append(f" - Key ingredients: {result['food_ingredients']}")
        if result.get('food_health_benefits'):
            food_context.append(f" - Health benefits: {result['food_health_benefits']}")
        context_parts.extend(food_context)
    return "\n".join(context_parts)

def generate_fallback_response(query: str, search_results: List[Dict]) -> str:
    if not search_results:
        return "I couldn't find any food items matching your request."
    top = search_results[0]
    return f"Based on '{query}', I recommend {top['food_name']} ({top['cuisine_type']}, {top['food_calories_per_serving']} cal)."

def generate_llm_rag_response(query: str, search_results: List[Dict]) -> str:
    try:
        context = prepare_context_for_llm(query, search_results)
        prompt = f"""You are a helpful food recommendation assistant.
User Query: "{query}"

Retrieved Food Options:
{context}

Please provide a helpful, short response that:
1. Acknowledges the user's request
2. Recommends 2-3 specific food items from the retrieved options
3. Explains why these match their request
4. Includes relevant details like calories and cuisine
Response:"""
        generated_response = model.generate(prompt=prompt, params=None)
        if generated_response and "results" in generated_response:
            return generated_response["results"][0]["generated_text"].strip()
        return generate_fallback_response(query, search_results)
    except Exception as e:
        print(f"LLM Error: {e}")
        return generate_fallback_response(query, search_results)

def handle_enhanced_rag_query(collection, query: str):
    search_results = perform_similarity_search(collection, query, 3)
    if not search_results:
        print("Bot: No food items matched your request.")
        return
    
    print(f"\nRetrieved {len(search_results)} relevant matches. Generating AI response...")
    ai_response = generate_llm_rag_response(query, search_results)
    print(f"\nBot: {ai_response}")
    
    print("\nUnderlying Retrieval Details:")
    for i, res in enumerate(search_results, 1):
        print(f" {i}. {res['food_name']} | {res['cuisine_type']} | {res['food_calories_per_serving']} cal")

def main():
    food_items = load_food_data('./FoodDataSet.json')
    collection = create_similarity_search_collection("enhanced_rag_food_chatbot")
    populate_similarity_collection(collection, food_items)
    
    if model is None:
        print("\nLLM not available - using fallback responses.")
        print("Set your IBM Cloud API key and project_id in the script to enable WatsonX.")
    else:
        print("\nTesting LLM connection...")
        test_response = model.generate(prompt="Hello", params=None)
        if test_response and "results" in test_response:
            print("LLM connection established.\n" + "=" * 55)
        else:
            print("LLM connection failed - using fallback responses.")
            print("=" * 55)

    print("Type your questions (or 'quit' to exit):")
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            handle_enhanced_rag_query(collection, user_input)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()