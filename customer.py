import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os
import re
import torch

# Set environment flags
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load and chunk FAQ data
with open("faq_data.txt", "r", encoding="utf-8") as f:
    raw_data = f.read()

faq_chunks = re.split(r"\n\s*\n", raw_data.strip())
docs = [Document(page_content=chunk.strip()) for chunk in faq_chunks if chunk.strip()]

# Embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embedding=embeddings)

# Load local model and tokenizer (force CPU to avoid MPS issues)
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(torch.device("cpu"))
generation_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

# Memory and QA chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = HuggingFacePipeline(pipeline=generation_pipeline)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(),
    memory=memory
)

# Streamlit interface
st.title("📘 FAQ Chatbot")
st.write("Ask a question based on our FAQ data:")

user_input = st.text_input("You:", key="user_input")

if user_input:
    response = qa_chain.invoke({"question": user_input})
    st.markdown("**Bot:** " + response.get("answer", str(response)))
