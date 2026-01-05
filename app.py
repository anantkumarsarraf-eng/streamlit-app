import streamlit as st
import requests
from PIL import Image
import io

# ---------------- CONFIG ----------------
st.set_page_config("Travel Recommendation Chatbot", layout="wide")

HF_TOKEN = st.secrets["HF_API_TOKEN"]

BLIP_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
LLM_API = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# ---------------- SESSION STATE ----------------
if "landmark" not in st.session_state:
    st.session_state.landmark = ""

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- FUNCTIONS ----------------
def image_to_landmark(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")

    try:
        response = requests.post(
            BLIP_API,
            headers=HEADERS,
            data=img_bytes.getvalue(),
            timeout=60
        )

        if response.status_code != 200:
            return "Image model is currently unavailable. Please try again."

        data = response.json()
        return data[0]["generated_text"]

    except Exception:
        return "Failed to identify landmark. Please try another image."


def ask_llm(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 300}
    }

    try:
        response = requests.post(
            LLM_API,
            headers=HEADERS,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            return "Language model is busy. Please try again."

        data = response.json()
        return data[0]["generated_text"]

    except Exception:
        return "Failed to generate response. Please retry."

# ---------------- UI ----------------
st.title("Travel Recommendation Chatbot")
st.caption("Transformer-based Multimodal AI (VLM + LLM)")

col1, col2 = st.columns([1, 2])

# ---------------- LEFT PANEL ----------------
with col1:
    st.subheader("Upload Landmark Image")
    image_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", width=280)

        if st.button("Identify Landmark"):
            with st.spinner("Analyzing image..."):
                landmark = image_to_landmark(image)
                st.session_state.landmark = landmark
                st.success("Processing complete")

# ---------------- RIGHT PANEL ----------------
with col2:
    st.subheader("Travel Chatbot")

    if st.session_state.landmark:
        st.markdown(f"**Identified Landmark:** {st.session_state.landmark}")

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

    user_input = st.chat_input(
        "Ask about travel, budget, best time, attractions..."
    )

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
st.caption("MACS AIML - Multimodal Transformer Project")
