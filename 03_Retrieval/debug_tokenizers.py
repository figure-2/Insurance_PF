"""
형태소 분석기 정성 평가: 실제 보험 약관 텍스트 샘플을 각 토크나이저로 분석하여 결과 비교
"""

from kiwipiepy import Kiwi
from mecab import MeCab as MeCabKo
from konlpy.tag import Okt
import json

# 샘플 문장 (보험 약관에서 실제로 나올 법한 복잡한 문장들) - 총 20개
sample_sentences = [
    # 1. 법률/전문 용어 (5개)
    "자동차손해배상보장법 제3조에 의한 배상책임을 집니다.",
    "피보험자의 고의로 인한 손해는 면책사항에 해당합니다.",
    "기명피보험자와 친족피보험자의 범위는 다릅니다.",
    "대물배상에서 지급하는 소송비용 및 변호사보수의 한도는?",
    "자기차량손해 담보의 차량가액 산정 기준은 무엇인가요?",
    
    # 2. 복합 명사 및 띄어쓰기 (5개)
    "무보험자동차상해담보는 뺑소니사고도 보상합니다.",  # 붙여씀
    "음주 운전 면책 조항에 의거하여 지급을 거절합니다.",  # 띄어씀
    "다른자동차운전담보특약에 가입되어 있어야 합니다.",
    "교통사고처리지원금은 형사합의금 성격의 비용입니다.",
    "긴급출동서비스 중 비상급유서비스는 연간 5회로 제한됩니다.",
    
    # 3. 조건 및 숫자 (5개)
    "제15조 제2항 제3호의 규정에 따릅니다.",
    "보험기간 중 발생한 사고에 한하여 보상합니다.",
    "만 26세 이상 한정운전 특약 위반 시 보상하지 않습니다.",
    "혈중알코올농도 0.03퍼센트 이상인 경우 면책됩니다.",
    "자기부담금은 손해액의 20%이며, 최저 20만원 최고 50만원입니다.",
    
    # 4. 애매모호한 표현/구어체 (5개)
    "차 문 열다가 옆 차 문콕했는데 보상되나요?",  # 문콕
    "대리 기사가 운전하다 사고 냈는데 제 보험으로 되나요?",  # 대리
    "가족들이랑 교대로 운전하다 사고 나면 어떡해요?",  # 교대
    "새 차 샀는데 기존 보험 승계 가능한가요?",  # 승계
    "블랙박스 달면 보험료 할인해주나요?",  # 특약 할인
]

def tokenize_with_all(text: str):
    """모든 토크나이저로 동일한 텍스트 분석"""
    results = {}
    
    # Kiwi
    try:
        kiwi = Kiwi()
        result = kiwi.analyze(text)
        if result and len(result) > 0 and len(result[0]) > 0:
            kiwi_tokens = [morph for morph, pos, _, _ in result[0][0]]
        else:
            kiwi_tokens = []
        results['Kiwi'] = kiwi_tokens
    except Exception as e:
        results['Kiwi'] = f"Error: {e}"
    
    # Mecab
    try:
        mecab = MeCabKo()
        results['Mecab'] = mecab.morphs(text)
    except Exception as e:
        results['Mecab'] = f"Error: {e}"
    
    # Okt
    try:
        okt = Okt()
        results['Okt'] = okt.morphs(text)
    except Exception as e:
        results['Okt'] = f"Error: {e}"
    
    return results

def main():
    print("="*80)
    print("형태소 분석기 정성 평가: 보험 약관 텍스트 샘플 분석 (20개)")
    print("="*80)
    
    categories = [
        ("법률/전문 용어", 0, 5),
        ("복합 명사 및 띄어쓰기", 5, 10),
        ("조건 및 숫자", 10, 15),
        ("애매모호한 표현/구어체", 15, 20),
    ]
    
    for cat_name, start, end in categories:
        print(f"\n{'='*80}")
        print(f"카테고리: {cat_name} ({end-start}개)")
        print(f"{'='*80}")
        
        for i in range(start, end):
            sentence = sample_sentences[i]
            print(f"\n[샘플 {i+1}] 원문: {sentence}")
            print("-" * 80)
            
            results = tokenize_with_all(sentence)
            
            for tokenizer_name, tokens in results.items():
                if isinstance(tokens, list):
                    print(f"{tokenizer_name:>10}: {' | '.join(tokens)}")
                else:
                    print(f"{tokenizer_name:>10}: {tokens}")
            
            print()
    
    # 실제 벡터 DB에서 샘플 가져오기
    print("\n" + "="*80)
    print("실제 벡터 DB 샘플 분석")
    print("="*80)
    
    try:
        with open("/home/pencilfoxs/0_Insurance_PF/01_Preprocessing/chunked_data.jsonl", 'r', encoding='utf-8') as f:
            samples = []
            for i, line in enumerate(f):
                if i >= 5:  # 처음 5개만
                    break
                item = json.loads(line)
                text = item['text']
                # 첫 200자만 추출
                sample_text = text[:200] if len(text) > 200 else text
                samples.append(sample_text)
        
        for i, sample_text in enumerate(samples, 1):
            print(f"\n[DB 샘플 {i}] (첫 200자)")
            print(f"원문: {sample_text[:100]}...")
            print("-" * 80)
            
            results = tokenize_with_all(sample_text)
            
            for tokenizer_name, tokens in results.items():
                if isinstance(tokens, list):
                    # 토큰이 너무 많으면 일부만 표시
                    display_tokens = tokens[:30] if len(tokens) > 30 else tokens
                    token_str = ' | '.join(display_tokens)
                    if len(tokens) > 30:
                        token_str += f" ... (총 {len(tokens)}개 토큰)"
                    print(f"{tokenizer_name:>10}: {token_str}")
                else:
                    print(f"{tokenizer_name:>10}: {tokens}")
            print()
    
    except Exception as e:
        print(f"Error loading DB samples: {e}")

if __name__ == "__main__":
    main()

