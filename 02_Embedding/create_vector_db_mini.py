
import json
import os
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Paths
DATA_PATH = "/home/pencilfoxs/0_Insurance_PF/01_Preprocessing/chunked_data.jsonl"
DB_PATH = "/home/pencilfoxs/0_Insurance_PF/02_Embedding/chroma_db_mini" # Use a separate DB for testing
MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 10
LIMIT = 50 # Only process 50 chunks

def load_chunks(file_path, limit=None):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            item = json.loads(line)
            text = item['text']
            meta = item['metadata']
            chunk_id = item['chunk_id']
            
            clean_meta = {
                "source": meta.get("source", ""),
                "company": meta.get("company", ""),
                "breadcrumbs": meta.get("breadcrumbs", ""),
                "token_count": meta.get("token_count", 0),
                "chunk_id": chunk_id,
                "policy_type": meta.get("policy_type", "unknown")
            }
            
            if "page_range" in meta and isinstance(meta["page_range"], list) and len(meta["page_range"]) == 2:
                clean_meta["page_start"] = meta["page_range"][0]
                clean_meta["page_end"] = meta["page_range"][1]
            
            doc = Document(page_content=text, metadata=clean_meta)
            documents.append(doc)
    return documents

def main():
    print(f"Initializing embedding model: {MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print(f"Loading first {LIMIT} chunks from {DATA_PATH}...")
    docs = load_chunks(DATA_PATH, limit=LIMIT)
    
    print(f"Creating Mini Vector DB at {DB_PATH}...")
    
    if os.path.exists(DB_PATH):
        import shutil
        shutil.rmtree(DB_PATH) # Clean start for mini test
        
    vector_store = Chroma(
        collection_name="insurance_policies_mini",
        embedding_function=embeddings,
        persist_directory=DB_PATH
    )
    
    vector_store.add_documents(docs)
    print("Mini Vector DB creation complete!")

if __name__ == "__main__":
    main()

