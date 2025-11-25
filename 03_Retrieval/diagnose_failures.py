"""
리트리버 정밀 진단 및 실패 분석 스크립트 (50개 질문)
오답 노트 자동 생성 기능 포함
"""

import json
import os
import sys
import time
from typing import List, Dict, Any
from mecab import MeCab as MeCabKo

# 현재 디렉토리를 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bm25_retriever import MecabBM25Retriever

# -----------------------------------------------------------------------------
# 추가 테스트 데이터 (기존 30개 + 추가 20개 = 총 50개)
# -----------------------------------------------------------------------------
ADDITIONAL_QUERIES = [
    # 동의어/유사어 테스트
    {"query": "뺑소니 사고 보상되나요?", "positive": "보유불명자동차에 의한 사고", "category": "동의어"},
    {"query": "자차부담금 얼마예요?", "positive": "자기차량손해 자기부담금", "category": "동의어"},
    {"query": "대리기사 사고 보상", "positive": "대리운전자가 운전 중 사고", "category": "동의어"},
    {"query": "견인 거리 얼마나 되나요?", "positive": "긴급견인서비스", "category": "동의어"},
    {"query": "배터리 방전됐어요", "positive": "배터리충전서비스", "category": "동의어"},
    
    # 복합 상황 테스트
    {"query": "여행 중 렌터카 빌렸는데 내 보험 되나요?", "positive": "다른자동차운전담보특약", "category": "복합"},
    {"query": "가족이 내 차 몰다가 사고나면?", "positive": "운전자 범위 및 연령 한정", "category": "복합"},
    {"query": "차유리 돌 맞아서 깨졌는데 보상되나요?", "positive": "자기차량손해", "category": "복합"},
    {"query": "태풍으로 침수되면 보상되나요?", "positive": "자기차량손해 보상하는 손해", "category": "복합"},
    {"query": "문콕 당했는데 상대방을 못 찾으면?", "positive": "물적사고 할증기준", "category": "복합"},
    
    # 구체적 수치/조건
    {"query": "음주운전 부담금 얼마?", "positive": "음주운전 사고부담금", "category": "수치"},
    {"query": "무면허운전 부담금", "positive": "무면허운전 사고부담금", "category": "수치"},
    {"query": "할증 기준 금액이 얼마인가요?", "positive": "물적사고 할증기준금액", "category": "수치"},
    {"query": "긴급출동 몇 번 부를 수 있나요?", "positive": "연간 이용한도", "category": "수치"},
    {"query": "대물배상 최소 가입금액", "positive": "대물배상 의무보험 가입금액", "category": "수치"},
    
    # 절차/서류
    {"query": "보험금 청구 서류 뭐 필요해?", "positive": "보험금 청구시 구비서류", "category": "절차"},
    {"query": "가지급금 받을 수 있나요?", "positive": "가지급금", "category": "절차"},
    {"query": "보험료 분할 납부 되나요?", "positive": "보험료의 분할납입", "category": "절차"},
    {"query": "계약 취소하고 싶은데요", "positive": "계약의 취소", "category": "절차"},
    {"query": "주소 변경하려면 어떻게?", "positive": "알릴 의무", "category": "절차"},
]

def load_dataset():
    """기존 데이터셋 + 추가 데이터셋 병합"""
    base_path = "/home/pencilfoxs/0_Insurance_PF/02_Embedding/evaluation_dataset.json"
    with open(base_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 추가 데이터셋 병합
    for item in ADDITIONAL_QUERIES:
        dataset.append({
            "query": item["query"],
            "positive": item["positive"],
            "category": item.get("category", "기타"),
            "type": "additional"
        })
    
    return dataset

def find_rank(retriever, query, positive_text, tokenizer):
    """정답 문서의 순위 찾기"""
    results = retriever.invoke(query)
    positive_tokens = set(tokenizer.morphs(positive_text))
    
    for rank, doc in enumerate(results, 1):
        # 1. 단순 텍스트 포함 여부 확인
        if positive_text in doc.page_content:
            return rank, doc
            
        # 2. 토큰 겹침 비율 확인 (30% 이상)
        doc_tokens = set(tokenizer.morphs(doc.page_content))
        overlap = len(positive_tokens & doc_tokens)
        if len(positive_tokens) > 0 and (overlap / len(positive_tokens)) >= 0.3:
            return rank, doc
            
    return 999, None

def analyze_failure(query, positive, top_doc):
    """실패 원인 자동 분석 (간이)"""
    if not top_doc:
        return "검색 결과 없음"
        
    return f"Top-1 문서: {top_doc.metadata.get('breadcrumbs', 'N/A')} (관련성 낮음)"

def main():
    print("🚀 리트리버 정밀 진단 시작 (총 50개 질문)")
    print("="*60)
    
    # 리트리버 로드
    index_path = "/home/pencilfoxs/0_Insurance_PF/03_Retrieval/bm25_index.pkl"
    try:
        retriever = MecabBM25Retriever.load_index(index_path)
        retriever.k = 5  # Top-5 검사
    except Exception as e:
        print(f"Error loading index: {e}")
        return

    tokenizer = MeCabKo()
    dataset = load_dataset()
    
    results = []
    failures = []
    
    for i, item in enumerate(dataset, 1):
        query = item['query']
        positive = item['positive']
        category = item.get('category', 'General')
        
        rank, found_doc = find_rank(retriever, query, positive, tokenizer)
        
        result = {
            "id": i,
            "query": query,
            "positive": positive,
            "rank": rank,
            "category": category
        }
        results.append(result)
        
        if rank > 5:  # Top-5 진입 실패
            # 실패 시 Top-1 문서 정보 가져오기 (분석용)
            top_results = retriever.invoke(query)
            top_doc = top_results[0] if top_results else None
            analysis = analyze_failure(query, positive, top_doc)
            
            failures.append({
                **result,
                "analysis": analysis,
                "top_1_content": top_doc.page_content[:100] if top_doc else ""
            })
            print(f"❌ [Fail] Q{i}: {query} (Rank: {rank})")
        else:
            print(f"✅ [Pass] Q{i}: {query} (Rank: {rank})")

    # 통계 계산
    success_count = len(dataset) - len(failures)
    success_rate = (success_count / len(dataset)) * 100
    
    print("\n" + "="*60)
    print(f"📊 진단 결과 요약 (총 {len(dataset)}개)")
    print(f"   - 성공: {success_count}개 ({success_rate:.1f}%)")
    print(f"   - 실패: {len(failures)}개")
    print("="*60)
    
    # 리포트 작성
    report_path = "/home/pencilfoxs/0_Insurance_PF/03_Retrieval/failure_analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 🩺 리트리버 실패 분석 리포트 (Failure Analysis)\n\n")
        f.write(f"**작성 일시:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**테스트 데이터:** 총 {len(dataset)}개 (기존 30 + 추가 20)\n")
        f.write(f"**성공률:** {success_rate:.1f}% ({success_count}/{len(dataset)})\n\n")
        
        f.write("## 1. 실패 케이스 목록 (Top-5 진입 실패)\n")
        for fail in failures:
            f.write(f"### ❌ Q{fail['id']}. {fail['query']}\n")
            f.write(f"- **카테고리:** {fail['category']}\n")
            f.write(f"- **정답 키워드:** `{fail['positive']}`\n")
            f.write(f"- **실패 원인 분석:** {fail['analysis']}\n")
            f.write(f"- **Top-1 문서 내용:** {fail['top_1_content']}...\n\n")
            
        f.write("## 2. 카테고리별 실패율\n")
        cats = set(r['category'] for r in results)
        for cat in cats:
            total = sum(1 for r in results if r['category'] == cat)
            failed = sum(1 for f in failures if f['category'] == cat)
            rate = (failed / total) * 100 if total > 0 else 0
            f.write(f"- **{cat}:** {total}개 중 {failed}개 실패 ({rate:.1f}%)\n")

    print(f"\n📝 상세 리포트가 생성되었습니다: {report_path}")

if __name__ == "__main__":
    main()

