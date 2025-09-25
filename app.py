import streamlit as st
import pymongo
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()
client = pymongo.MongoClient(os.getenv('MONGODB_URI'))
db = client['rbi_updates']
collection = db['notifications']

# RAG Setup (free Hugging Face)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = HuggingFaceEndpoint(
    repo_id="gpt2",
    huggingfacehub_api_token=os.getenv('HF_TOKEN'),
    temperature=0.1
)

@st.cache_data(ttl=7200)  # Single cache decorator
def load_docs():
    ist = ZoneInfo("Asia/Kolkata")
    two_weeks_ago = datetime.now(ist).date() - timedelta(days=14)  # Past 2 weeks
    docs = list(collection.find().sort('timestamp', -1).limit(100))
    recent_docs = []
    for d in docs:
        try:
            # Parse RSS date format (e.g., 'Wed, 06 Aug 2025 16:30:00')
            pub_date_str = d['pub_date']
            pub_date = datetime.strptime(pub_date_str, '%a, %d %b %Y %H:%M:%S')
            # Assume UTC for RSS and convert to IST
            pub_date = pub_date.replace(tzinfo=timezone.utc).astimezone(ist)
            if two_weeks_ago <= pub_date.date() <= datetime.now(ist).date():
                recent_docs.append(d)
        except (ValueError, KeyError):
            continue  # Skip invalid dates or missing fields
    texts = [f"Title: {d['title']}\nCategory: {d['category']}\nDate: {d['pub_date']}\nDescription: {d['description']}\nLink: {d['link']}" for d in recent_docs]
    vectorstore = FAISS.from_texts(texts, embeddings) if texts else None
    return vectorstore, recent_docs

def rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) if vectorstore else None
    template = """Based only on the following RBI context, list all notifications from the past 2 weeks with their title, category, date, link, and a concise synopsis (2-3 sentences summarizing the description). Format each notification clearly with these details. If no notifications are found, say so explicitly.

Context: {context}

Question: {question}

Answer:
"""
    prompt = PromptTemplate.from_template(template)
    if not vectorstore:
        # Return a simple Runnable for no context case
        def no_context():
            return "No notifications found in the past 2 weeks."
        chain = RunnablePassthrough.assign(context=no_context) | prompt | llm | StrOutputParser()
    else:
        chain = (
            {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    return chain

st.title("RBI Updates Dashboard & AI Assistant")

tab1, tab2 = st.tabs(["Latest Updates", "AI Chat"])

with tab1:
    docs = list(collection.find().sort('pub_date', -1).limit(20))
    st.dataframe([{"Title": d['title'], "Category": d['category'], "Date": d['pub_date'], "Link": d['link']} for d in docs])

with tab2:
    vectorstore, recent_docs = load_docs()  # Updated to match return variable
    chain = rag_chain(vectorstore)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input("Ask about RBI updates (e.g., 'What are the latest notifications about digital lending?')"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            response = chain.invoke(prompt)
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    pass