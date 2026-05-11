import streamlit as st
import requests
import json
import os

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="SHL Agent Tester", layout="wide")
st.title("SHL Assessment Agent – Tester")

# Sidebar – health check
with st.sidebar:
    st.header("🔍 Health Check")
    api_base = st.text_input("API Base URL", value=API_BASE).rstrip("/")
    if st.button("Check /health"):
        try:
            resp = requests.get(f"{api_base}/health")
            st.json(resp.json())
        except Exception as e:
            st.error(f"Error: {e}")


# Initialise conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history (uses st.chat_message – works like a real chat UI)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If assistant response includes recommendations, show them
        if msg["role"] == "assistant" and "recommendations" in msg:
            recs = msg["recommendations"]
            if recs:
                st.markdown("**📋 Recommendations:**")
                for r in recs:
                    st.markdown(f"- [{r['name']}]({r['url']}) – {r.get('test_type', 'N/A')}")

# Chat input
if prompt := st.chat_input("Ask about SHL assessments..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI /chat
    payload = {"messages": [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]}
    try:
        resp = requests.post(f"{api_base}/chat", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        assistant_reply = data["reply"]
        recommendations = data.get("recommendations", [])

        # Store assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_reply,
            "recommendations": recommendations
        })

        with st.chat_message("assistant"):
            st.markdown(assistant_reply)
            if recommendations:
                st.markdown("**📋 Recommendations:**")
                for r in recommendations:
                    st.markdown(f"- [{r['name']}]({r['url']}) – {r.get('test_type', 'N/A')}")

    except Exception as e:
        st.error(f"API call failed: {e}")