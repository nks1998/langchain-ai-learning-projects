import os

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def process_doc(path):
    loader = PyPDFDirectoryLoader(path)
    doc = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitter_data = splitter.split_documents(doc)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vector_store =  InMemoryVectorStore.from_documents(
        documents=splitter_data,
        embedding=embeddings
    )

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

    @tool
    def retriever_tool(query:str):
        """
        This toole can help you to retrive the relevant data of the PDF 
        documents and these document have details about medical reports
        """
        docs = vector_store.similarity_search(query=query, k=4)
        context = ""

        for doc in docs:
            context += doc.page_content+"\n\n"

        return context

    System_Prompt = """
    You are a helpful assistant that answers questions using retrieved context.
    Always use the 'retriever_tool' tool fo the questions requiring external knowledge.
    """

    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[retriever_tool],
        system_prompt=System_Prompt,
        checkpointer=memory
    )
    st.session_state.agent = agent
    st.session_state.document_uploaded = True

### Upload UI
if not st.session_state.document_uploaded:
    uploaded =  st.file_uploader(label="Select PDF Files", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing...."):
            path = "doc_files/"
            os.makedirs(path, exist_ok=True)

            for file in uploaded:
                with open(path+file.name,"wb") as f:
                    f.write(file.getvalue())

            process_doc(path)
            st.rerun()

### CHAT UI

if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)
        
    query = st.chat_input("Ask anything related to uploaded documents....")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": 1}}
        )
        ans = response["messages"][-1].content[0]["text"]
        st.session_state.messages.append({"role": "ai", "content": ans})
        st.chat_message("ai").markdown(ans)


# while True:
#     query = input("User: ")
#     if query.lower() == "quit":
#         break

#     response = agent.invoke(
#         {"messages": [{"role": "user", "content": query}]},
#         {"configurable":{"thread_id":1}}
#     )
#     result = response["messages"][-1].content[0]["text"]

#     print("AI: ", result)