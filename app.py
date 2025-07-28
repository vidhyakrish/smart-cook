import streamlit as st
import pandas as pd
import os
from openai import OpenAI

# Load your OpenAI API key securely
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load recipe data
df = pd.read_csv("recipes_diverse.csv")

st.title("🍳 Smart AI Cook")
st.write("Ask me what you want to cook with — ingredients, region, cuisine...")

query = st.text_input("What would you like to cook today?")

if query:
    # Turn CSV into a context string
    context = ""
    for _, row in df.iterrows():
        context += f"""
        Dish: {row['dish']}
        Region: {row['region']}
        Cooking Time: {row['cooking_time']}
        Ingredients: {row['ingredients']}
        Steps: {row['recipe_steps']}
        ---
        """

    # Create the prompt
    prompt = f"""You are a recipe assistant. Based on the user query below, search the context for a suitable dish:
    
    Context:
    {context}

    User Query: {query}
    
    Give a helpful response with dish name, ingredients, steps, and a friendly tone."""

    # Make OpenAI call
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        answer = response.choices[0].message.content
        st.markdown("### 🍽️ Suggested Recipe")
        st.write(answer)

        # Try to show the matching image from CSV
        match = df[df['dish'].str.lower().str.contains(query.lower())]
        if not match.empty:
            image_url = match.iloc[0]['image_url']
            if pd.notna(image_url):
                st.image(image_url, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
