import streamlit as st
from IPython.core.debugger import prompt
from chatdoc import get_rag_response

st.set_page_config(page_title="Chatdoc", layout="wide")
st.title("Chatdoc")
if "messages" not in st.session_state:
    st.session_state["messages"] = []
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("请输入..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
        # ... (这里是 AI 思考和生成答案的过程) ...
            #response = "AI 的回答"
            response = get_rag_response(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})