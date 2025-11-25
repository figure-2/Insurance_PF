"""
리트리버 문맥(Context) 품질 검증 스크립트
LLM에게 전달될 검색 결과가 충분한 정보를 담고 있는지 육안으로 확인
"""

import sys
import os

# 현재 디렉토리를 경로에 추가하여 모듈 import 가능하게 함
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bm25_retriever import MecabBM25Retriever

def print_separator(char="=", length=80):
    print(char * length)

def test_queries():
    # 테스트할 질문 리스트 (다양한 유형 포함)
    queries = [
        # 1. 명확한 면책/보상 질문
        "음주운전하면 보상 받을 수 있나요?",
        
        # 2. 구체적인 수치/조건 질문
        "자기부담금은 얼마인가요?",
        
        # 3. 절차/방법 질문
        "사고 났을 때 보험금 청구는 어떻게 하나요?",
        
        # 4. 정의/용어 질문
        "무보험자동차란 무엇인가요?",
        
        # 5. 복합 상황 질문
        "다른 사람 차를 운전하다가 사고가 나면 제 보험으로 처리 되나요?"
    ]
    
    # 리트리버 로드
    index_path = "/home/pencilfoxs/0_Insurance_PF/03_Retrieval/bm25_index.pkl"
    print(f"Loading BM25 index from {index_path}...")
    
    try:
        retriever = MecabBM25Retriever.load_index(index_path)
        # LLM에게 보통 3~5개의 문서를 주므로 Top-3 확인
        retriever.k = 3
    except Exception as e:
        print(f"Error loading index: {e}")
        return

    print_separator()
    print("🕵️  리트리버 문맥 품질 검증 (Context Quality Check)")
    print("    - LLM이 답변하기에 충분한 정보가 검색되는지 확인")
    print_separator()

    for i, query in enumerate(queries, 1):
        print(f"\n❓ [질문 {i}] {query}")
        print_separator("-")
        
        # 검색 수행
        results = retriever.invoke(query)
        
        if not results:
            print("❌ 검색 결과 없음")
            continue
            
        for rank, doc in enumerate(results, 1):
            company = doc.metadata.get('company', 'Unknown')
            breadcrumbs = doc.metadata.get('breadcrumbs', 'N/A')
            source = os.path.basename(doc.metadata.get('source', 'Unknown File'))
            
            print(f"📄 [Rank {rank}] {company} | {breadcrumbs}")
            print(f"   (Source: {source})")
            print("-" * 40)
            
            # 본문 출력 (너무 길면 앞부분만 표시하되, 핵심은 보이도록)
            content = doc.page_content.strip()
            # 가독성을 위해 줄바꿈 정리
            print(content)
            print("-" * 40)
            print()
        
        print_separator()
        # 다음 질문으로 넘어가기 전 잠시 대기 (가독성)
        # time.sleep(1) 

if __name__ == "__main__":
    test_queries()

