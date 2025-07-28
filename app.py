
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("recipes_with_images.csv")

# Combine ingredients and region for matching
df["combined"] = df["ingredients"] + " " + df["region"]

# Input from user
st.title("Smart AI Cook 🍳")
query = st.text_input("What ingredients or type of cuisine do you want to cook with?")

if query:
    # Vectorize
    vect = TfidfVectorizer()
    tfidf_matrix = vect.fit_transform(df["combined"])
    query_vec = vect.transform([query])

    # Cosine similarity
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_index = similarity.argmax()

    # Get top result
    result = df.iloc[top_index]

    st.subheader(f"🍽️ {result['dish']}")
    st.image(result["image_url"], caption=result["dish"])
    st.write("**Ingredients:**", result["ingredients"])
    st.write("**Steps:**", result["recipe_steps"])
    st.write("**Region:**", result["region"])
    st.write("**Cooking Time:**", result["cooking_time"])
