from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Redis
import redis


def clean_text(text):
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.replace(" ", "").isdigit():
            continue
        if line.count("0") > 10:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


pdf_loader = DirectoryLoader(
    path="data", glob="**/*.pdf", loader_cls=PyPDFLoader)
pdf_docs = pdf_loader.load()

docx_loader = DirectoryLoader(
    path="data", glob="**/*.docx", loader_cls=Docx2txtLoader)
word_docs = docx_loader.load()

all_docs = pdf_docs + word_docs
print(f"Loaded {len(pdf_docs)} PDF pages.")
print(f"Loaded {len(word_docs)} Word documents.")
print(f"Total: {len(all_docs)} documents.")

for doc in all_docs:
    doc.page_content = clean_text(doc.page_content)
    doc.metadata["source"] = doc.metadata.get("source", "unknown")
    if "doktorski" in doc.page_content.lower():
        doc.metadata["level"] = "doktorski"
    elif "diplomski" in doc.page_content.lower():
        doc.metadata["level"] = "diplomski"
    elif "preddiplomski" in doc.page_content.lower():
        doc.metadata["level"] = "preddiplomski"
    else:
        doc.metadata["level"] = "nepoznato"


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

r = redis.Redis(host="127.0.0.1", port=6379)

r.flushall()
print("Cleared all data from Redis.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "],
)

raw_chunks = text_splitter.split_documents(all_docs)
docs_with_splits = [
    doc for doc in raw_chunks
    if len(doc.page_content.strip()) > 200 and "0 0 0" not in doc.page_content
]
print(f"Split into {len(docs_with_splits)} document chunks.")

vectorstore = Redis.from_documents(
    docs_with_splits,
    embeddings,
    redis_url="redis://localhost:6379",
    index_name="my-index"
)

lengths = [len(doc.page_content) for doc in docs_with_splits]
print(
    f"Min length: {min(lengths)}, Max length: {max(lengths)}, Avg length: {sum(lengths)//len(lengths)}")

print("Vectors stored in Redis!")
