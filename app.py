import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the CSV
df = pd.read_csv("recipes_with_images_better.csv")

# Combine fields to give the model more context
df["combined"] = df["ingredients"] + " " + df["recipe_steps"] + " " + df["region"] + " " + df["dish"]

# UI
st.title("Smart AI Cook 🍳")
query = st.text_input("What ingredients or dish do you want to cook?")

if query:
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["combined"])
    query_vec = vectorizer.transform([query])

    # Similarity
    similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_index = similarity.argmax()
    top_score = similarity[top_index]

    if top_score < 0.2:
        st.warning("Hmm... I couldn’t find a good match. Try something else?")
    else:
        result = df.iloc[top_index]
        st.subheader(result['dish'])

        if pd.notna(result['image_url']) and result['image_url'].startswith("http"):
            st.image(result['image_url'], caption=result['dish'], use_container_width=True)

        st.write("**Region:**", result['region'])
        st.write("**Cooking Time:**", result['cooking_time'])
        st.write("**Ingredients:**", result['ingredients'])
        st.write("**Steps:**", result['recipe_steps'])
