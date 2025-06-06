from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Redis

from langchain.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

vectorstore = Redis.from_existing_index(
    embedding=embedding,
    redis_url="redis://localhost:6379",
    index_name="my-index",
    schema=None
)

test_queries = [
    "Kako upisati preddiplomski studij?",
    "Uvjeti za diplomski studij informatike",
    "Koje dokumente treba za doktorat?",
    "Do kada je prijava za upis na studij?",
    "Postoji li razlika u upisu za strane studente?"
]

for query in test_queries:
    print("="*80)
    print(f"🔍 {query}")
    results = vectorstore.similarity_search_with_score(query, k=3)
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n--- Result {i} (Score: {score:.4f}) ---")
        print(doc.page_content[:500])
