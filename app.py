import streamlit as st
import pandas as pd
import pyttsx3
import speech_recognition as sr
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv("recipes_diverse.csv")
    # Add simple diet column if not present
    if 'diet' not in df.columns:
        df['diet'] = ['Vegetarian' if 'tofu' in ingr.lower() or 'vegetable' in ingr.lower() else 'Non-Vegetarian' for ingr in df['ingredients']]
    return df

df = load_data()

# Sidebar filters
regions = ['All'] + sorted(df['region'].unique().tolist())
selected_region = st.sidebar.selectbox("Filter by Region", regions)

time_ranges = ['All', '< 20 mins', '20-40 mins', '> 40 mins']
selected_time = st.sidebar.selectbox("Filter by Cooking Time", time_ranges)

diets = ['All', 'Vegetarian', 'Non-Vegetarian']
selected_diet = st.sidebar.selectbox("Filter by Diet Preference", diets)

# Filter dataframe
filtered_df = df.copy()
if selected_region != 'All':
    filtered_df = filtered_df[filtered_df['region'] == selected_region]

if selected_diet != 'All':
    filtered_df = filtered_df[filtered_df['diet'] == selected_diet]

def time_to_minutes(time_str):
    try:
        return int(time_str.split()[0])
    except:
        return 0

if selected_time != 'All':
    filtered_df['time_mins'] = filtered_df['cooking_time'].apply(time_to_minutes)
    if selected_time == '< 20 mins':
        filtered_df = filtered_df[filtered_df['time_mins'] < 20]
    elif selected_time == '20-40 mins':
        filtered_df = filtered_df[(filtered_df['time_mins'] >= 20) & (filtered_df['time_mins'] <= 40)]
    else:
        filtered_df = filtered_df[filtered_df['time_mins'] > 40]

# Main app UI
st.title("Smart AI Cook 🍽️")

# Voice input function
def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak now.")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        return text
    except Exception:
        return ""

voice_input = st.button("🎤 Use Voice Input")
spoken_text = ""
if voice_input:
    spoken_text = get_audio()
    if spoken_text:
        st.success(f"You said: {spoken_text}")
    else:
        st.error("Sorry, I couldn't understand. Please try again.")

# Text input: default to voice input if available
query = st.text_input("What would you like to cook today?", value=spoken_text)

if query:
    # Build context from filtered_df
    context = ""
    for _, row in filtered_df.iterrows():
        context += f"""
Dish: {row['dish']}
Region: {row['region']}
Cooking Time: {row['cooking_time']}
Ingredients: {row['ingredients']}
Steps: {row['recipe_steps']}
"""

    prompt = f"""
You are a helpful cooking assistant. Based on the following recipes, answer the user query briefly and helpfully:

{context}

User query: {query}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        answer = response.choices[0].message.content
        st.markdown("### 🍽️ Suggested Recipe")
        st.write(answer)

        # Find best matching dish in filtered_df to show image
        matched = filtered_df[filtered_df['dish'].str.lower().apply(lambda x: x in answer.lower())]
        if not matched.empty:
            img_url = matched.iloc[0]['image_url']
            if pd.notna(img_url):
                st.image(img_url, use_container_width=True)

        # Text to Speech button
        if st.button("🔊 Listen to recipe"):
            engine = pyttsx3.init()
            engine.say(answer)
            engine.runAndWait()

    except Exception as e:
        st.error(f"Error: {e}")
