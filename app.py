import streamlit as st
import openai
from openai import OpenAI

st.set_page_config(page_title="Campfire Design Assistant", page_icon="🔥")

st.title("🔥 Campfire Design System Assistant")
st.write("Ask anything about components, UI patterns, or usage guidance.")

# Load system context
with open("LLM.txt", "r") as f:
    context = f.read()

# Input field
question = st.text_input("💬 Your Question:")

# Use OpenAI v1 client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if question:
    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": question}
            ],
            temperature=0.4
        )
        st.markdown("**💡 Answer:** " + response.choices[0].message.content.strip())
