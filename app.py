import streamlit as st
import openai
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd

@st.cache_resource
def load_data():
    df = pd.read_csv("recipes.csv")
    texts = df.apply(lambda row: f"{row['ingredients']} → {row['dish']} ({row['region']}): {row['steps']} – takes {row['time']}", axis=1).tolist()
    return df, texts

df, recipe_texts = load_data()

@st.cache_resource
def build_index(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    return model, index, embeddings

model, index, embeddings = build_index(recipe_texts)
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🍲 Vidhya's smart Cooking Assistant")
query = st.text_input("Enter ingredients you have (e.g. lemon, rice, turmeric):")

if query:
    with st.spinner("Finding recipe ideas..."):
        query_embedding = model.encode([query])
        _, indices = index.search(query_embedding, k=5)

        matches = [recipe_texts[i] for i in indices[0]]
        context = "\n\n".join(matches)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a cooking assistant. Given some ingredients, suggest a dish based on this context:"},
                {"role": "user", "content": f"Context:\n{context}\n\nIngredients: {query}"}
            ],
            temperature=0.7
        )

        st.markdown("**🍽️ Suggested Dish:**")
        st.markdown(response.choices[0].message.content.strip())
