import streamlit as st
import requests
from PIL import Image
import io
import time

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Travel Recommendation Chatbot",
    layout="wide"
)

HF_TOKEN = st.secrets["HF_API_TOKEN"]

BLIP_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
LLM_API = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ---------------- SESSION STATE ----------------
if "landmark" not in st.session_state:
    st.session_state.landmark = ""

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- SAFE API CALLS ----------------
def image_to_landmark(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")

    for _ in range(3):  # retry 3 times
        response = requests.post(
            BLIP_API,
            headers=HEADERS,
            data=img_bytes.getvalue(),
            timeout=60
        )

        if response.status_code == 200:
            try:
                return response.json()[0]["generated_text"]
            except:
                return "Landmark detected but description unavailable."

        time.sleep(5)

    return "Image model is busy. Please try again after some time."

def ask_llm(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 300}
    }

    for _ in range(3):  # retry 3 times
        response = requests.post(
            LLM_API,
            headers=HEADERS,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            try:
                return response.json()[0]["generated_text"]
            except:
                return "Response generated but could not be parsed."

        time.sleep(5)

    return "Language model is currently busy. Please try again later."

# ---------------- UI ----------------
st.title("Travel Recommendation Chatbot")
st.caption("Transformer-based Multimodal AI (VLM + LLM)")

col1, col2 = st.columns([1, 2])

# ---------------- LEFT PANEL ----------------
with col1:
    st.subheader("Upload Landmark Image")
    image_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"]
    )

    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", width=280)

        if st.button("Identify Landmark"):
            with st.spinner("Analyzing image..."):
                st.session_state.landmark = image_to_landmark(image)
                st.success("Process completed")

# ---------------- RIGHT PANEL ----------------
with col2:
    st.subheader("Travel Chatbot")

    if st.session_state.landmark:
        st.markdown(f"**Identified Landmark:** {st.session_state.landmark}")

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

    user_input = st.chat_input("Ask about travel, budget, best time, attractions")

    if user_input:
        st.session_state.chat.append(("user", user_input))

        context = f"""
You are a professional travel guide.

Landmark:
{st.session_state.landmark}

Conversation:
"""

        for r, m in st.session_state.chat:
            context += f"{r}: {m}\n"

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                answer = ask_llm(context)
                st.write(answer)

        st.session_state.chat.append(("assistant", answer))

st.divider()
st.caption("MACS AIML • Multimodal Transformer Project")
