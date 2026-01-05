import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config("🌍 Travel Recommendation Chatbot", layout="wide")

HF_TOKEN = st.secrets["HF_API_TOKEN"]

BLIP_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
LLM_API = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

if "landmark" not in st.session_state:
    st.session_state.landmark = ""

if "chat" not in st.session_state:
    st.session_state.chat = []

def image_to_landmark(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    response = requests.post(BLIP_API, headers=HEADERS, data=img_bytes.getvalue())
    return response.json()[0]["generated_text"]

def ask_llm(prompt):
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300}}
    response = requests.post(LLM_API, headers=HEADERS, json=payload)
    return response.json()[0]["generated_text"]

st.title("🌍 Travel Recommendation Chatbot")
st.caption("Multimodal AI using VLM + LLM")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📸 Upload Landmark Image")
    image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if image_file:
        image = Image.open(image_file)
        st.image(image, width=280)
        if st.button("🔍 Identify Landmark"):
            with st.spinner("Analyzing image..."):
                st.session_state.landmark = image_to_landmark(image)
                st.success("Landmark identified!")

with col2:
    st.subheader("🧭 Travel Assistant")
    if st.session_state.landmark:
        st.markdown(f"**Identified Landmark:** {st.session_state.landmark}")

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

    user_input = st.chat_input("Ask about travel, budget, best time...")
    if user_input:
        st.session_state.chat.append(("user", user_input))
        context = f"Landmark: {st.session_state.landmark}\n"
        for r, m in st.session_state.chat:
            context += f"{r}: {m}\n"
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_llm(context)
                st.write(answer)
        st.session_state.chat.append(("assistant", answer))
