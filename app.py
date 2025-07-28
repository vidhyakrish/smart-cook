import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("recipes_with_images.csv")

# Combine more fields for better context
df["combined"] = df["ingredients"] + " " + df["recipe_steps"] + " " + df["region"]

# UI
st.title("Smart AI Cook 🍳")
query = st.text_input("What ingredients or dish do you want to cook?")

if query:
    # Vectorize the combined column
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["combined"])
    query_vec = vectorizer.transform([query])

    # Calculate similarity
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_index = similarity.argmax()

    result = df.iloc[top_index]

    # Show result
    st.subheader(result['dish'])
    if pd.notna(result['image_url']) and result['image_url'].startswith("http"):
        st.image(result['image_url'], caption=result['dish'], use_column_width=True)
    st.write("**Region:**", result['region'])
    st.write("**Cooking Time:**", result['cooking_time'])
    st.write("**Ingredients:**", result['ingredients'])
    st.write("**Steps:**", result['recipe_steps'])
