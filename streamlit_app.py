import streamlit as st
from shared_functions import (
    load_food_data,
    create_similarity_search_collection,
    populate_similarity_collection,
    perform_similarity_search,
)

st.set_page_config(page_title="Food RAG Chatbot", page_icon="🍲")
st.title("🍲 Food Recommendation Chatbot")

@st.cache_resource
def setup_vector_db():
    food_data = load_food_data("./FoodDataSet.json")
    collection = create_similarity_search_collection("streamlit_rag_food")
    populate_similarity_collection(collection, food_data)
    return collection

with st.spinner("Loading food dataset and vector database..."):
    collection = setup_vector_db()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask for food recommendations (e.g., 'healthy dinner under 400 cal')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            results = perform_similarity_search(collection, prompt, n_results=3)
            
            if results:
                response = "Based on your search, here are the top matches:\n\n"
                for i, item in enumerate(results, 1):
                    response += (
                        f"{i}. **{item['food_name']}** ({item['cuisine_type']})\n"
                        f"   Calories: {item['food_calories_per_serving']} per serving\n"
                        f"   Match: {item['similarity_score']*100:.1f}%\n"
                        f"   {item['food_description'][:200]}...\n\n"
                    )
            else:
                response = "No matching food items found. Try different keywords."
            
            st.markdown(response)

            if results:
                with st.expander("Retrieved Matches"):
                    for item in results:
                        st.markdown(
                            f"- **{item['food_name']}** ({item['cuisine_type']}) | "
                            f"{item['food_calories_per_serving']} cal | "
                            f"{item['similarity_score']*100:.1f}% match"
                        )

    st.session_state.messages.append({"role": "assistant", "content": response})