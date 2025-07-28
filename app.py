import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Vidhya's Smart AI Cook 🍽️")
st.write("Ask me what you want to cook based on ingredients, region, or mood!")

# Load the recipe data
@st.cache_data
def load_data():
    df = pd.read_csv("recipes_diverse.csv")
    df["combined"] = df["ingredients"] + " " + df["recipe_steps"] + " " + df["region"] + " " + df["dish"]
    return df

df = load_data()

# Vectorize all recipe descriptions
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["combined"])

# User query input
query = st.text_input("What do you feel like cooking today?")

if query:
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_index = similarity.argmax()
    top_score = similarity[top_index]

    if top_score > 0.2:
        result = df.iloc[top_index]
        st.subheader(result['dish'])

        if pd.notna(result['image_url']) and result['image_url'].startswith("http"):
            st.image(result['image_url'], caption=result['dish'], use_container_width=True)

        st.write("**Region:**", result['region'])
        st.write("**Cooking Time:**", result['cooking_time'])
        st.write("**Ingredients:**", result['ingredients'])
        st.write("**Steps:**", result['recipe_steps'])
    else:
        st.warning("Hmm... I couldn’t find a good match. Try something else?")
