from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from typing import Annotated

class ChatState(BaseModel):
    messages: Annotated[list, add_messages]

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

def ChatBotNode(state: ChatState) -> ChatState:
    res = llm.invoke(state.messages)
    state.messages = [res]
    return state

memory = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chatbot", ChatBotNode)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

graph = graph.compile(checkpointer=memory)

config = {"configurable":{"thread_id": "my-bot-1"}}

while True:
    query = input("User: ")
    if query.lower() == "quit":
        break


    res = graph.invoke({"messages":[{"role":"user", "content": query}]}, config)
    ans = res["messages"][-1].content[0]["text"]
    print("AI: ", ans)