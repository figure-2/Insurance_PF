
import json
import time
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
DATASET_PATH = "/home/pencilfoxs/0_Insurance_PF/02_Embedding/evaluation_dataset.json"
MODELS_TO_TEST = [
    "jhgan/ko-sroberta-multitask",
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large"
]

def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_model_embeddings(model_name, dataset):
    """Generates embeddings for all queries, positives, and negatives in the dataset."""
    print(f"\n>> Testing Model: {model_name}...")
    
    # 1. Load Model
    start_load = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    load_time = time.time() - start_load
    print(f"   Model Loaded in {load_time:.2f}s")
    
    results = {
        "model_name": model_name,
        "load_time": load_time,
        "details": [],
        "avg_pos_score": 0,
        "avg_separation": 0
    }
    
    # 2. Process Dataset items
    queries = []
    passages = [] # Will contain [pos, neg] for each item
    
    for item in dataset:
        q = item['query']
        p = item['positive']
        n = item['negative']
        
        # Add prefixes for e5 models if needed
        if "e5" in model_name:
            q = f"query: {q}"
            p = f"passage: {p}"
            n = f"passage: {n}"
            
        queries.append(q)
        passages.extend([p, n])
        
    # 3. Batch Embedding
    start_embed = time.time()
    q_vecs = embeddings.embed_documents(queries)
    p_vecs = embeddings.embed_documents(passages)
    embed_time = time.time() - start_embed
    
    results['embed_time_per_item'] = embed_time / len(dataset)
    
    # 4. Calculate Scores
    total_pos_score = 0
    total_separation = 0
    
    for i, item in enumerate(dataset):
        q_vec = np.array([q_vecs[i]])
        p_vec = np.array([p_vecs[2*i]])     # Positive
        n_vec = np.array([p_vecs[2*i + 1]]) # Negative
        
        pos_sim = cosine_similarity(q_vec, p_vec)[0][0]
        neg_sim = cosine_similarity(q_vec, n_vec)[0][0]
        separation = pos_sim - neg_sim
        
        total_pos_score += pos_sim
        total_separation += separation
        
        results['details'].append({
            "category": item['category'],
            "query": item['query'],
            "pos_score": float(pos_sim),
            "neg_score": float(neg_sim),
            "separation": float(separation)
        })

    results['avg_pos_score'] = total_pos_score / len(dataset)
    results['avg_separation'] = total_separation / len(dataset)
    
    return results

def main():
    print("=== Extended Embedding Model Evaluation (30 FAQ Dataset) ===")
    dataset = load_dataset(DATASET_PATH)
    print(f"Loaded {len(dataset)} test cases.")
    
    final_report = []
    
    for model_name in MODELS_TO_TEST:
        try:
            res = get_model_embeddings(model_name, dataset)
            final_report.append(res)
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            
    # Print Summary Table
    print("\n" + "="*80)
    print(f"{'Model Name':<30} | {'Avg Pos Score':<13} | {'Avg Separation':<14} | {'Time/Item':<10}")
    print("-" * 80)
    
    for res in final_report:
        print(f"{res['model_name'].split('/')[-1]:<30} | {res['avg_pos_score']:.4f}        | {res['avg_separation']:.4f}         | {res['embed_time_per_item']:.3f}s")
    print("="*80)
    
    # Identify Best Model
    best_model = max(final_report, key=lambda x: x['avg_separation'])
    print(f"\n🏆 Best Model by Separation: {best_model['model_name']}")
    
    # Detailed failure cases for best model (where separation < 0)
    print(f"\n>> Hard Cases for {best_model['model_name']} (Separation < 0.1):")
    for detail in best_model['details']:
        if detail['separation'] < 0.1:
            print(f"[{detail['category']}] Q: {detail['query']}")
            print(f"   Pos: {detail['pos_score']:.4f} | Neg: {detail['neg_score']:.4f} | Diff: {detail['separation']:.4f}")

if __name__ == "__main__":
    main()
