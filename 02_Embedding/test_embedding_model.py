
import time
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def test_model(model_name, texts):
    print(f"Loading model: {model_name}...")
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}, # Use CPU for now as I'm not sure about GPU availability
        encode_kwargs={'normalize_embeddings': True}
    )
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")

    print("Embedding texts...")
    start_time = time.time()
    embedded_texts = embeddings.embed_documents(texts)
    embed_time = time.time() - start_time
    print(f"Embedded {len(texts)} texts in {embed_time:.2f} seconds")
    
    return embedded_texts, embeddings

def main():
    # Sample texts relevant to insurance
    texts = [
        "자동차 사고 시 보상 범위에 대해 알고 싶습니다.",
        "음주운전 사고는 보상하지 않습니다.",
        "대인배상과 대물배상의 차이점은 무엇인가요?",
        "보험 가입 시 필요한 서류는 무엇입니까?",
        "교통사고 발생 시 처리 절차 안내"
    ]
    
    # Query text
    query = "음주 운전하면 보험 처리 되나요?"
    
    # Compare models
    models = ["jhgan/ko-sroberta-multitask", "BAAI/bge-m3"]
    
    for model_name in models:
        print(f"\nTesting {model_name}...")
        doc_embeddings, model = test_model(model_name, texts)
        query_embedding = model.embed_query(query)
        
        # Calculate similarity
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        print(f"Query: {query}")
        print("-" * 30)
        for text, score in zip(texts, similarities):
            print(f"Score: {score:.4f} | Text: {text}")

if __name__ == "__main__":
    main()

