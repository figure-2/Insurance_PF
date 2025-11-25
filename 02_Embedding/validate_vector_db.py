
import time
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Paths
DB_PATH = "/home/pencilfoxs/0_Insurance_PF/02_Embedding/chroma_db"
MODEL_NAME = "jhgan/ko-sroberta-multitask"

def main():
    # GPU/CPU 자동 감지
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Device Status: Using {device.upper()}")
    
    print(f"Initializing embedding model: {MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print(f"Loading Vector DB from {DB_PATH}...")
    vector_store = Chroma(
        collection_name="insurance_policies",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )
    
    # 1. Count check
    # Note: Chroma's langchain wrapper doesn't always expose a direct count method easily without accessing the client
    # We can try accessing the underlying collection
    try:
        count = vector_store._collection.count()
        print(f"✅ Total documents in Vector DB: {count}")
    except Exception as e:
        print(f"Could not get count directly: {e}")

    # 2. Retrieval Test
    query = "음주 운전하면 면책인가요?"
    print(f"\nTest Query: {query}")
    print("-" * 30)
    
    results = vector_store.similarity_search_with_score(query, k=3)
    
    for doc, score in results:
        print(f"Score: {score:.4f}") # Note: Chroma returns distance by default (lower is better), but Langchain might convert.
        # Usually cosine distance: 0 is identical, 1 is opposite.
        # If normalize_embeddings=True, inner product is cosine similarity. 
        # Chroma default is L2 distance if not specified. Let's see the output.
        print(f"Company: {doc.metadata.get('company')}")
        print(f"Breadcrumbs: {doc.metadata.get('breadcrumbs')}")
        print(f"Text: {doc.page_content[:100]}...")
        print("-" * 20)

    # 3. Filtering Test
    print("\nTest Filtering (Company: 롯데손해보험주식회사)")
    print("-" * 30)
    results_filter = vector_store.similarity_search_with_score(
        query, 
        k=3,
        filter={"company": "롯데손해보험주식회사"}
    )
    
    for doc, score in results_filter:
        print(f"Score: {score:.4f}")
        print(f"Company: {doc.metadata.get('company')}")
        print(f"Text: {doc.page_content[:50]}...")
        print("-" * 20)

if __name__ == "__main__":
    main()

