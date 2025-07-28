import streamlit as st
import pandas as pd
import openai
import os

st.set_page_config(page_title="Smart AI Cook", layout="centered")
st.title("Smart AI Cook 🍽️")
st.write("Type what you feel like cooking. Mention ingredients, region, or dish type!")

# Load recipes
@st.cache_data
def load_data():
    df = pd.read_csv("recipes_diverse_final.csv")
    return df

df = load_data()

# Get OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Handle query
query = st.text_input("What would you like to cook today?")

if query:
    context = ""
    for _, row in df.iterrows():
        context += (
            f"Dish: {row['dish']}\n"
            f"Ingredients: {row['ingredients']}\n"
            f"Steps: {row['recipe_steps']}\n"
            f"Region: {row['region']}\n"
            f"Cooking Time: {row['cooking_time']}\n\n"
        )

    prompt = f"""You are a cooking assistant. Based on the following dataset, respond to the user query with the most relevant dish.

    Dataset:
    {context}

    User query: {query}

    Answer only with the most relevant matching recipe from the dataset. Keep it brief and factual.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        answer = response.choices[0].message.content.strip()
        st.markdown("### Suggested Recipe")
        st.markdown(answer)

        # Attempt to show image for top matching dish
        for _, row in df.iterrows():
            if row["dish"].lower() in answer.lower() and str(row["image_url"]).startswith("http"):
                st.image(row["image_url"], caption=row["dish"], use_container_width=True)
                break
    except Exception as e:
        st.error("Something went wrong. Please check your API key or try again later.")
