
import argparse
import json
import os
import time
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

DEFAULT_DATA_PATH = "/home/pencilfoxs/0_Insurance_PF/01_Preprocessing/chunked_data.jsonl"
DEFAULT_DB_PATH = "/home/pencilfoxs/0_Insurance_PF/02_Embedding/chroma_db"
DEFAULT_MODEL = "jhgan/ko-sroberta-multitask"
DEFAULT_BATCH = 100

def load_chunks(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            text = item['text']
            meta = item['metadata']
            chunk_id = item['chunk_id']
            
            # Flatten metadata for ChromaDB
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
    parser = argparse.ArgumentParser(description="Create Chroma vector DB from chunked data.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace embedding model name")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH, help="Path to chunked_data.jsonl")
    parser.add_argument("--db_path", default=DEFAULT_DB_PATH, help="Output directory for Chroma DB")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH, help="Batch size for ingestion")
    parser.add_argument("--skip_cleanup", action="store_true", help="Do not delete existing DB before ingest")
    args = parser.parse_args()

    model_name = args.model
    data_path = args.data_path
    db_path = args.db_path
    batch_size = args.batch_size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Device Status: Using {device.upper()}")
    if device == 'cuda':
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"Initializing embedding model: {model_name}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print(f"Loading data from {data_path}...")
    docs = load_chunks(data_path)
    total_docs = len(docs)
    print(f"Loaded {total_docs} documents.")
    
    if os.path.exists(db_path) and not args.skip_cleanup:
        import shutil
        print(f"Cleaning existing DB at {db_path} for fresh ingestion...")
        shutil.rmtree(db_path)

    print(f"Creating Vector DB at {db_path}...")
    
    vector_store = Chroma(
        collection_name="insurance_policies",
        embedding_function=embeddings,
        persist_directory=db_path
    )
    
    start_time = time.time()
    
    for i in range(0, total_docs, batch_size):
        batch = docs[i : i + batch_size]
        vector_store.add_documents(batch)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + len(batch))
        remaining = (total_docs - (i + len(batch))) * avg_time
        
        print(f"Processed {i + len(batch)}/{total_docs} chunks... (Avg: {avg_time:.3f}s/doc, Est. remaining: {remaining:.0f}s)")
        
    print(f"Vector DB creation complete! Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
