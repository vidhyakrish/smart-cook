import streamlit as st
import openai
from openai import OpenAI

st.set_page_config(page_title="Smart Cooking Assistant", page_icon="🍳")

st.title("🍳 Vidhya's Smart Cooking Assistant")
st.write("Tell me the ingredients you have, and I'll suggest a dish or (2 or more )!")

# Load system context
with open("LLM.txt", "r") as f:
    context = f.read()

# Input field
ingredients = st.text_input("🧂 What ingredients do you have?")

# Use OpenAI v1 client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if ingredients:
    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": f"I have: {ingredients}"}
            ],
            temperature=0.8
        )
        st.markdown("**🍽️ Suggested Dish:** " + response.choices[0].message.content.strip())
