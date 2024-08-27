import streamlit as st
from llmpipeline import process_answer

st.markdown("##  LogiAssist: AI-powered logistics insights")
if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role":"assistant","content":"In your service ..."})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask here"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    answer = process_answer(prompt)
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})





