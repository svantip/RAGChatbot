from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Redis
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver

redis_url = "redis://localhost:6379"
index_name = "my-index"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = Redis.from_existing_index(
    embedding=embeddings,
    redis_url=redis_url,
    index_name=index_name,
    schema=None
)

model = ChatOllama(model="deepseek-r1:8b", temperature=0.3)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
    Ti si AI asistent koji pomaže s upisom na fakultet pomoću vektoriziranih podataka parsiranih datoteka natječaja za upise.
    U slučaju da korisnik pozdravi pozdravi ga nazad i reci čemu služiš.
    Uvijek odgovaraj jezgrovito i jasno.
    Ako korisnik piše na srpskom jeziku, odgovaraj na hrvatskom jeziku.
    Ako korisnik piše na nekom drugom jeziku, odgovaraj na istom jeziku.
    Koristi samo informacije iz konteksta.
    Odgovori samo ako nađeš odgovor ne izmišljaj informacije reci ne znam.
    """),
    MessagesPlaceholder(variable_name="messages"),
])


def call_model(state: MessagesState):
    user_question = state["messages"][-1].content
    retrieved_docs = vectorstore.similarity_search_with_score(
        user_question, k=3)
    context = "\n\n".join([doc.page_content for doc, _ in retrieved_docs])
    final_prompt = f"Context:\n{context}\n\nQuestion:\n{user_question}"

    prompt = prompt_template.invoke({
        "messages": [{"role": "user", "content": final_prompt}]
    })

    response = model.invoke(prompt)
    return {"messages": response}


workflow = StateGraph(state_schema=MessagesState)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

st.set_page_config(page_title="Chat with Uni AI Assistant", page_icon="🦜")
st.title("AI Asistent za upise na faks")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Postavi pitanje o upisima..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        response = app.invoke({
            "messages": st.session_state.messages
        }, config={"thread_id": "default"})

    answer = response["messages"][-1].content
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
