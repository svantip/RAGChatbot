from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Redis
import redis

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

embeddings = OllamaEmbeddings(model="deepseek-r1:8b")

r = redis.Redis(host="127.0.0.1", port=6379)

r.flushall()
print("Cleared all data from Redis.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=500,
)

docs_with_splits = text_splitter.split_documents(all_docs)
print(f"Split into {len(docs_with_splits)} document chunks.")

vectorstore = Redis.from_documents(
    docs_with_splits,
    embeddings,
    redis_url="redis://localhost:6379",
    index_name="my-index"
)

print("Vectors stored in Redis!")
