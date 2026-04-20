import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from src.agent import repondre

st.set_page_config(
    page_title="NosAI",
    page_icon="🌸",
    layout="centered"
)

st.title("NosAI")
st.caption("Ton expert en parfumerie de niche — objectif, sourcé, honnête.")

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pose ta question sur la parfumerie de niche..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("NosAI réfléchit..."):
            reponse = repondre(prompt)
        st.markdown(reponse)

    st.session_state.messages.append({"role": "assistant", "content": reponse})