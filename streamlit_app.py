import streamlit as st
from shared_functions import (
    load_food_data,
    create_similarity_search_collection,
    populate_similarity_collection,
    perform_similarity_search,
)
from enhanced_rag_chatbot import generate_llm_rag_response

st.set_page_config(page_title="Food RAG Chatbot", page_icon="🍲")
st.title("🍲 Interactive Food & RAG Chatbot")

# 1. Initialize and cache vector database across user interactions
@st.cache_resource
def setup_vector_db():
    food_data = load_food_data("./FoodDataSet.json")
    collection = create_similarity_search_collection("streamlit_rag_food")
    populate_similarity_collection(collection, food_data)
    return collection

with st.spinner("Loading food dataset and vector database..."):
    collection = setup_vector_db()

# 2. Maintain conversational chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. User query input
if prompt := st.chat_input("Ask for food recommendations (e.g., 'healthy dinner under 400 cal')..."):
    # Append & show user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching database & generating AI recommendation..."):
            results = perform_similarity_search(collection, prompt, n_results=3)
            
            if results:
                response = generate_llm_rag_response(prompt, results)
            else:
                response = "No matching food items found in the database. Please try different keywords."
            
            st.markdown(response)

            # Show retrieved matches for full transparency
            if results:
                with st.expander("Retrieved Vector Matches (ChromaDB)"):
                    for item in results:
                        match_pct = item['similarity_score'] * 100
                        st.markdown(
                            f"- **{item['food_name']}** ({item['cuisine_type']}) | "
                            f"{item['food_calories_per_serving']} cal | *{match_pct:.1f}% match*"
                        )

    # Save assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": response})