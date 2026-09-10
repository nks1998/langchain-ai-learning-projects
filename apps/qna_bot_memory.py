import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="You are a General Knowledge Expert!",
    checkpointer=st.session_state.memory,
)

st.title("AskBuddy - AI QnA Bot!")
st.markdown("My QnA Bot with Langchain")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask Anything!")
if query:
    st.session_state.history.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": "1"}},
    )

    response = result["messages"][-1].content[0]["text"]
    st.session_state.history.append(
        {"role": "ai", "content": response}
    )
    st.chat_message("ai").markdown(response)
