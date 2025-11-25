import json
import os
import re
import glob
from typing import List, Dict, Any, Optional
import markdownify
# import tiktoken (제거)
# import pandas as pd (제거)
from tqdm import tqdm

# --- 설정 ---
# 글자 수 기준 (대략 1토큰 ≈ 1.5~2글자, 넉넉잡아 계산)
TARGET_TOKEN_SIZE = 1000  # 목표 청크 크기 (글자 수)
MAX_TOKEN_SIZE = 2000    # 최대 허용 청크 크기
MIN_TOKEN_SIZE = 400     # 최소 청크 크기
OVERLAP_SIZE = 100       # 오버랩 크기

# Tiktoken 대신 글자 수 기반 계산
# enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """텍스트의 길이(글자 수)를 계산합니다."""
    return len(text)

def load_json(file_path: str) -> Dict[str, Any]:
    """JSON 파일을 로드합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_text(text: str) -> str:
    """기본적인 텍스트 정제를 수행합니다."""
    if not text:
        return ""
    return text.strip()

def is_garbage(element: Dict[str, Any]) -> bool:
    """
    요소가 제거해야 할 쓰레기(Header, Footer, Page Number)인지 판단합니다.
    """
    category = element.get('category', '')
    text = element.get('text', '').strip()
    
    # 1. 명시적인 Header/Footer 카테고리 제거
    if category in ['header', 'footer']:
        return True
        
    # 2. 페이지 번호 패턴 제거 (숫자만 있거나, - 숫자 - 형태 등)
    # 예: "3", "- 3 -", "Page 3"
    if re.match(r'^[\d\s\-\.]+$', text):
        # 아주 짧은 숫자만 있는 경우 페이지 번호로 간주
        if len(text) < 10: 
            return True
            
    return False

def convert_table_to_markdown(element: Dict[str, Any]) -> str:
    """
    HTML 테이블을 Markdown 형식으로 변환합니다.
    """
    html_content = element.get('html', '')
    if not html_content:
        return element.get('text', '') # HTML 없으면 텍스트 반환
    
    try:
        # markdownify를 사용하여 변환
        md = markdownify.markdownify(html_content, heading_style="ATX")
        # 연속된 공백 정리
        md = re.sub(r'\n\s*\n', '\n\n', md).strip()
        return md
    except Exception as e:
        # 변환 실패 시 텍스트 반환 및 로그 (실제로는 로깅 처리)
        return element.get('text', '')

def recursive_split(text: str, breadcrumbs: str, max_tokens: int, overlap: int) -> List[str]:
    """
    텍스트가 너무 길 경우 재귀적으로 분할합니다.
    분할된 각 조각의 앞에 breadcrumbs를 붙여 문맥을 유지합니다.
    """
    # tokens = enc.encode(text) # 사용 안 함
    if len(text) <= max_tokens:
        return [breadcrumbs + "\n\n" + text]
    
    chunks = []
    # 단순하게 반으로 나누는 것보다, 문단 단위로 나누는 것이 좋음
    # 여기서는 간단히 문자열 길이 기반으로 나누고, 추후 고도화 가능
    # 텍스트 분할 라이브러리(LangChain 등)를 안 쓰고 직접 구현한다면:
    
    # 1. "\n\n" (문단)으로 분할 시도
    split_chars = ["\n\n", "\n", ". ", " "]
    
    for char in split_chars:
        parts = text.split(char)
        current_chunk = ""
        current_tokens = 0
        
        temp_chunks = []
        success = True
        
        for part in parts:
            # 구분자 복원
            part_with_sep = part + char if char != " " else part + " " 
            part_tokens = count_tokens(part_with_sep)
            
            if part_tokens > max_tokens:
                # 하나의 문단이 이미 max_tokens보다 크다면 더 작은 단위로 쪼개야 함
                # 현재 구분자로는 실패
                success = False
                break
            
            if current_tokens + part_tokens > max_tokens:
                # 현재 청크 마무리
                temp_chunks.append(current_chunk)
                current_chunk = part_with_sep
                current_tokens = part_tokens
            else:
                current_chunk += part_with_sep
                current_tokens += part_tokens
        
        if current_chunk:
            temp_chunks.append(current_chunk)
            
        if success:
            # 분할 성공 시 breadcrumbs 붙여서 반환
            return [breadcrumbs + "\n\n" + c.strip() for c in temp_chunks]
            
    # 모든 구분자로도 실패하면 강제로 토큰 단위 자르기 (여기서는 생략)
    # 실제로는 LangChain의 RecursiveCharacterTextSplitter 사용 권장
    return [breadcrumbs + "\n\n" + text] # Fallback


def process_document(json_path: str, company_name: str) -> List[Dict[str, Any]]:
    """
    하나의 문서를 처리하여 청크 리스트를 반환합니다.
    """
    data = load_json(json_path)
    elements = data.get('elements', [])
    
    # 1. 정렬 (Page -> ID)
    elements.sort(key=lambda x: (x.get('page', 0), x.get('id', 0)))
    
    chunks = []
    
    # 상태 변수
    breadcrumbs = [] # 제목 경로 스택
    current_group_content = [] # 현재 그룹에 모으고 있는 텍스트 조각들
    current_group_tokens = 0
    current_group_page_start = -1
    current_group_page_end = -1
    current_heading_level = 0 # 현재 헤딩 레벨 (1, 2, 3...)
    
    source_filename = os.path.basename(json_path)
    
    # 약관 유형 파악 (간단한 로직)
    policy_type = "알수없음"
    if "보통약관" in json_path: policy_type = "보통약관"
    elif "특별약관" in json_path: policy_type = "특별약관"
    
    
    def flush_group():
        nonlocal current_group_content, current_group_tokens, current_group_page_start, current_group_page_end
        
        if not current_group_content:
            return

        full_text = "\n\n".join(current_group_content)
        full_breadcrumbs = " > ".join(breadcrumbs)
        
        # 토큰 확인
        total_tokens = count_tokens(full_text)
        
        # 청크 생성 (너무 길면 분할)
        if total_tokens > MAX_TOKEN_SIZE:
            split_texts = recursive_split(full_text, full_breadcrumbs, MAX_TOKEN_SIZE, OVERLAP_SIZE)
            for i, split_text in enumerate(split_texts):
                chunks.append({
                    "chunk_id": f"{company_name}_{source_filename}_{len(chunks)}",
                    "text": split_text,
                    "metadata": {
                        "source": json_path,
                        "company": company_name,
                        "policy_type": policy_type,
                        "breadcrumbs": full_breadcrumbs,
                        "page_range": [current_group_page_start, current_group_page_end],
                        "token_count": count_tokens(split_text),
                        "is_split": True
                    }
                })
        else:
            # 메타데이터에 breadcrumbs 포함
            final_text = f"[{full_breadcrumbs}]\n\n{full_text}"
            chunks.append({
                "chunk_id": f"{company_name}_{source_filename}_{len(chunks)}",
                "text": final_text,
                "metadata": {
                    "source": json_path,
                    "company": company_name,
                    "policy_type": policy_type,
                    "breadcrumbs": full_breadcrumbs,
                    "page_range": [current_group_page_start, current_group_page_end],
                    "token_count": count_tokens(final_text),
                    "is_split": False
                }
            })
            
        # 초기화
        current_group_content = []
        current_group_tokens = 0
        current_group_page_start = -1

    
    for element in elements:
        # Garbage 제거
        if is_garbage(element):
            continue
            
        category = element.get('category', 'paragraph')
        text = clean_text(element.get('text', ''))
        page = element.get('page', 0)
        
        if not text:
            continue
            
        # 페이지 범위 업데이트
        if current_group_page_start == -1:
            current_group_page_start = page
        current_group_page_end = page
        
        # [Context Continuity Logic] 
        # 파일 시작 부분이고, 아직 Breadcrumbs가 없는데, 첫 텍스트가 제목이 아닌 경우
        # (이전 파일에서 이어지는 내용일 가능성이 큼)
        if not breadcrumbs and not category.startswith('heading') and not current_group_content:
             breadcrumbs = ["(이전 내용에서 계속)"]

        # [Small Heading Pattern Logic]
        # category는 paragraph지만 텍스트가 <...>, (예시), [참고] 등으로 시작하고 짧은 경우 -> 준-헤딩으로 취급
        is_sub_heading = False
        if category == 'paragraph' and len(text) < 40:
             if text.startswith('<') or text.startswith('(') or text.startswith('[') or text.startswith('※'):
                 # 준-헤딩은 독립 청크로 만들지 않고, 다음 본문 앞에 붙여주기 위해 
                 # 현재 그룹에 추가하되, 줄바꿈을 좀 더 명확히 함
                 current_group_content.append(f"\n### {text}")
                 is_sub_heading = True
        
        if is_sub_heading:
            continue

        # Heading 처리
        if category.startswith('heading'):
            # 헤딩 레벨 파악 (heading1 -> 1)
            try:
                level = int(category.replace('heading', ''))
            except:
                level = 1
            
            # 새로운 헤딩이 나오면, 
            # 1. 이전 그룹 Flush 시도
            # 정책: 이전 그룹에 '본문 내용'이 있으면 Flush하고, 없으면(제목만 있으면) Flush하지 않고 병합한다.
            
            # 본문 내용이 있는지 확인 (현재 그룹의 텍스트가 제목(## ...) 외에 더 있는지)
            has_content = False
            for content in current_group_content:
                if not content.strip().startswith("##"):
                    has_content = True
                    break
            
            if has_content:
                flush_group()
                # 새 그룹 시작
                current_group_content.append(f"## {text}")
            else:
                # 본문이 없으면 Flush 안 함 -> 현재 제목을 버리고 새 제목으로 대체하거나, 
                # 혹은 그냥 이어 붙일 수도 있음. 
                # 하지만 '빈 제목'만 있는 청크를 막기 위해, 이전 제목 줄을 제거하고 새 제목으로 덮어쓰거나
                # Breadcrumbs만 업데이트하고 내용은 비워두는 게 좋음.
                
                # 여기서는 간단히: 이전 그룹 리셋하고 새 제목으로 시작 (어차피 내용은 없었으므로)
                current_group_content = [f"## {text}"]
            
            # 2. Breadcrumbs 업데이트
            # 현재 레벨보다 깊거나 같은 레벨이 나오면, 그 레벨 이상의 기존 breadcrumbs 제거
            # 예: [H1, H2] 상태에서 H2가 나오면 -> [H1] -> [H1, NewH2]
            # 예: [H1, H2, H3] 상태에서 H1이 나오면 -> [] -> [NewH1]
            
            # (간단한 스택 관리: level이 리스트 인덱스와 매핑되지는 않음. Upstage parser는 heading1, heading2 정도만 줌)
            # 여기서는 heading1을 최상위, heading2를 그 다음으로 가정.
            # 하지만 Upstage Parser는 heading1만 주는 경우가 많음.
            # 따라서, 텍스트 패턴(제1조, 1., 가. 등)을 보고 레벨을 판단하는 게 더 정확할 수 있음.
            # 일단 category 기반으로 하되, heading1이 연속되면 형제 노드로 취급
            
            # Breadcrumbs 관리 전략 수정:
            # 단순히 리스트에 추가하되, 'heading1'이 나오면 무조건 초기화하고 다시 시작하는게 안전할 수 있음 (약관 구조상)
            # 혹은 텍스트 내용을 보고 판단.
            
            # 전략: heading1은 대단원, heading2는 소단원이라고 가정.
            if level == 1:
                # 1레벨이면 기존 싹 비우고 시작
                breadcrumbs = [text]
            else:
                # 2레벨 이상이면, 1레벨은 유지하고 그 뒤에 붙임
                # 만약 이미 2레벨이 있었다면 교체
                if len(breadcrumbs) >= level:
                    breadcrumbs = breadcrumbs[:level-1]
                breadcrumbs.append(text)
                
            # 헤딩 자체는 텍스트에 포함하지 않고 Breadcrumbs로만 쓸 것인가? 
            # -> 헤딩도 텍스트에 포함시키는 게 좋음 (검색 걸리게)
            # 하지만 Breadcrumbs에 있으므로 본문에는 생략? -> 아니오, 본문 흐름상 있는게 자연스러움.
            # current_group_content.append(f"# {text}") 
            # -> 아님. 헤딩이 나왔으니 Flush했으므로, 이 헤딩은 '다음 그룹'의 시작이 되어야 함.
            # 다만 Breadcrumbs에 들어가므로 굳이 텍스트에 중복해서 넣을 필요는 없을 수도 있지만,
            # Markdown 스타일로 제목을 달아주는게 좋음.
            
            # 새 그룹 시작
            # current_group_content.append(f"## {text}") 
            
        elif category == 'table':
            # 표 처리
            md_table = convert_table_to_markdown(element)
            current_group_content.append(md_table)
            
        else:
            # 일반 텍스트 (paragraph, list 등)
            current_group_content.append(text)
            
    # 마지막 그룹 처리
    flush_group()
    
    return chunks

def main():
    base_dir = "/home/pencilfoxs/0_Insurance_PF/data"
    output_dir = "/home/pencilfoxs/0_Insurance_PF/01_Preprocessing"
    output_file = os.path.join(output_dir, "chunked_data.jsonl")
    
    # 모든 보험사 디렉토리 검색
    # base_dir 바로 아래에 있는 디렉토리들을 보험사 이름으로 간주
    companies = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    all_chunks = []
    
    print(f"Found {len(companies)} companies: {companies}")
    
    for target_company in companies:
        # 파일 검색
        search_pattern = os.path.join(base_dir, target_company, "data", "*.json")
        json_files = glob.glob(search_pattern)
        
        print(f"Processing {len(json_files)} files for {target_company}...")
        
        for json_file in tqdm(json_files, desc=target_company):
            try:
                chunks = process_document(json_file, target_company)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
            
    # 결과 저장
    print(f"Saving {len(all_chunks)} chunks to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            
    print("Done.")
    
    # 샘플 출력
    if all_chunks:
        print("\n--- Sample Chunk ---")
        print(json.dumps(all_chunks[10] if len(all_chunks) > 10 else all_chunks[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

