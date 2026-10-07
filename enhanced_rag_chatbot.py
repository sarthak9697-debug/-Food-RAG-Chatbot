from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from typing import List, Dict
import os

def generate_llm_rag_response(query: str, retrieved_items: List[Dict], 
                              credentials: dict = None, project_id: str = None) -> str:
    """
    Generate a RAG response using IBM Granite model via watsonx.ai.
    
    Args:
        query: User's search query
        retrieved_items: List of food items from similarity search
        credentials: IBM Cloud credentials (url, api_key)
        project_id: watsonx.ai project ID
    
    Returns:
        Generated response string
    """
    
    if not credentials or not project_id:
        return generate_fallback_response(query, retrieved_items)
    
    try:
        client = APIClient(credentials=credentials, project_id=project_id)
        
        model = ModelInference(
            model_id=ModelTypes.GRANITE_13B_CHAT_V2,
            credentials=credentials,
            project_id=project_id,
            params={
                GenParams.DECODING_METHOD: "greedy",
                GenParams.MAX_NEW_TOKENS: 300,
                GenParams.TEMPERATURE: 0.1,
                GenParams.REPETITION_PENALTY: 1.1
            }
        )
        
        context = build_rag_context(retrieved_items)
        prompt = build_rag_prompt(query, context)
        
        response = model.generate(prompt=prompt)
        return response['results'][0]['generated_text'].strip()
        
    except Exception as e:
        print(f"Error generating LLM response: {e}")
        return generate_fallback_response(query, retrieved_items)


def build_rag_context(retrieved_items: List[Dict]) -> str:
    """Build context string from retrieved food items."""
    context_parts = []
    for i, item in enumerate(retrieved_items, 1):
        context_parts.append(
            f"{i}. {item['food_name']} ({item['cuisine_type']}) - "
            f"{item['food_calories_per_serving']} cal - "
            f"{item['food_description'][:200]}"
        )
    return "\n".join(context_parts)


def build_rag_prompt(query: str, context: str) -> str:
    """Build the RAG prompt for the LLM."""
    return f"""<|system|>
You are a helpful food recommendation assistant. Use the provided food data to answer the user's query.
Provide personalized recommendations based on the search results. Be concise and helpful.
<|user|>
User Query: {query}

Retrieved Food Items:
{context}

Please provide a helpful food recommendation response based on these search results.
<|assistant|>"""


def generate_fallback_response(query: str, retrieved_items: List[Dict]) -> str:
    """Generate a fallback response when LLM is not available."""
    if not retrieved_items:
        return f"I couldn't find any matches for '{query}'. Try different keywords!"
    
    response = f"Based on your search for '{query}', I found {len(retrieved_items)} great options:\n\n"
    
    for i, item in enumerate(retrieved_items, 1):
        score = item['similarity_score'] * 100
        response += (
            f"{i}. **{item['food_name']}** ({item['cuisine_type']}) - "
            f"{item['food_calories_per_serving']} calories - "
            f"{score:.0f}% match\n"
            f"   {item['food_description'][:150]}...\n\n"
        )
    
    response += "Would you like more details about any of these dishes?"
    return response


def test_rag_response():
    """Test function for RAG response generation."""
    sample_items = [
        {
            'food_name': 'Chocolate Lava Cake',
            'cuisine_type': 'French',
            'food_calories_per_serving': 450,
            'food_description': 'Rich chocolate cake with molten center',
            'similarity_score': 0.85
        },
        {
            'food_name': 'Tiramisu',
            'cuisine_type': 'Italian',
            'food_calories_per_serving': 320,
            'food_description': 'Coffee-flavored Italian dessert',
            'similarity_score': 0.78
        }
    ]
    
    print(generate_fallback_response("chocolate dessert", sample_items))


if __name__ == "__main__":
    test_rag_response()