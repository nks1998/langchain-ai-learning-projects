from dotenv import load_dotenv
import streamlit as st
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
out = StrOutputParser()
chain = llm | out

st.title("AskBuddy - AI QnA Bot!")
st.markdown("My QnA Bot with Langchain")

if "messages" not in st.session_state:
    st.session_state.messages = []

for messages in st.session_state.messages:
    role = messages["role"]
    content = messages["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask Anything!")
if query:
    st.session_state.messages.append({"role":"user","content": query})
    st.chat_message("user").markdown(query)
    ans = chain.invoke(query)
    st.session_state.messages.append({"role":"ai","content": ans})
    st.chat_message("ai").markdown(ans)
    