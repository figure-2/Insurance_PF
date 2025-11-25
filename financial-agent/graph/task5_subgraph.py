# -*- coding: utf-8 -*-
"""
개인별 Risk Tolerance를 고려한 매매 패턴 위험 알림 기능
Personalized Trading Pattern Risk Alert (PTPRA)
"""

# --- 1. 필수 라이브러리 임포트 ---
import os
import re
import json
import requests
import time
import traceback
import difflib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from io import BytesIO
from zoneinfo import ZoneInfo

from langgraph.graph import StateGraph, END
from .state import AgentState

from bs4 import BeautifulSoup, NavigableString
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from PIL import Image

from .utils import load_config, create_shareable_url

KST = ZoneInfo("Asia/Seoul")
config = load_config()

# prompt_load
with open(config["task5"]["prompt_paths"]["extract_mydata"], 'r', encoding='utf-8') as f:
    prompt_extract_mydata = f.read()

with open(config["task5"]["prompt_paths"]["extract_paragraph"], 'r', encoding='utf-8') as f:
    prompt_extract_paragraph = f.read()

with open(config["task5"]["prompt_paths"]["summarize_reason"], 'r', encoding='utf-8') as f:
    prompt_summarize_reason = f.read()

def _find_best_matching_chunk(llm_output: str, original_text: str, threshold: float = 0.7) -> str | None:
    """
    LLM이 생성한 텍스트와 가장 유사한 부분을 원본 텍스트에서 찾습니다.
    문장 단위 및 연속된 두 문장 단위로 비교하여 정확도를 높입니다.
    """
    # 1. 원본 텍스트를 문장 단위로 분리하고, 각 문장에 마침표를 다시 붙여줌
    #    (split으로 인해 사라진 마침표를 복원해야 정확한 비교 가능)
    sentences = [s.strip() for s in original_text.split('.') if s.strip()]
    
    # 2. 비교할 텍스트 덩어리(chunk) 리스트 생성
    chunks_to_check = []
    # 2-1. 개별 문장 추가
    chunks_to_check.extend(sentences)
    # 2-2. 연속된 두 문장을 합쳐서 추가 (더 긴 매칭을 위해)
    if len(sentences) > 1:
        for i in range(len(sentences) - 1):
            chunks_to_check.append(f"{sentences[i]}. {sentences[i+1]}".strip())

    if not chunks_to_check:
        return None

    best_match = ""
    max_similarity_ratio = 0.0
    matcher = difflib.SequenceMatcher(None, llm_output.strip())

    # 3. 생성된 모든 chunk와 유사도 비교
    for chunk in chunks_to_check:
        # 마침표가 없는 원본 문장도 있을 수 있으므로 chunk에도 마침표를 붙여 최종 비교
        # 예: " ...라고 했다" 와 " ...라고 했다." 비교
        matcher.set_seq2(f"{chunk}.")
        # 마침표를 제외하고도 비교
        ratio_with_period = matcher.ratio()

        matcher.set_seq2(chunk)
        ratio_without_period = matcher.ratio()

        # 두 경우 중 더 높은 유사도 점수를 사용
        similarity_ratio = max(ratio_with_period, ratio_without_period)

        if similarity_ratio > max_similarity_ratio:
            max_similarity_ratio = similarity_ratio
            best_match = chunk

    # 4. 임계값을 넘는 경우에만 최종 결과로 인정하고, 마침표를 붙여서 반환
    if max_similarity_ratio >= threshold:
        return f"{best_match}."
    else:
        return None
    
def _extract_json_from_llm_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    LLM 응답에서 첫 번째로 발견되는 유효한 JSON 객체를 안정적으로 추출합니다.
    - ```json ... ``` 코드 블록을 우선적으로 파싱합니다.
    - 문자열 값 내부에 이스케이프되지 않은 큰따옴표(")가 포함된 경우에도 복구를 시도합니다.
    """
    # 1. ```json ... ``` 코드 블록 우선 파싱
    match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
        # 마크다운 블록 안의 JSON도 깨졌을 수 있으므로 아래의 복구 로직을 통과시킴
        response_text = json_str

    try:
        # 2. 괄호 쌍을 이용해 첫 번째 JSON 객체로 추정되는 부분 추출
        start_index = response_text.find('{')
        if start_index == -1:
            return None

        brace_count = 1
        i = start_index + 1
        while i < len(response_text) and brace_count > 0:
            char = response_text[i]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            i += 1

        if brace_count != 0:
            return None # 짝이 맞는 괄호를 찾지 못함

        json_str_slice = response_text[start_index:i]

        # 3. 표준 파서로 먼저 시도 (가장 좋은 케이스)
        try:
            return json.loads(json_str_slice)
        except json.JSONDecodeError:
            # 4. 파싱 실패 시, 문자열 값 내부의 따옴표만 복구
            repaired_json = ""
            in_string_value = False
            i = 0
            while i < len(json_str_slice):
                char = json_str_slice[i]
                repaired_json += char

                # 콜론 뒤에 따옴표가 오면, 문자열 "값"이 시작된 것으로 간주
                # (키 부분이나 다른 구조는 건드리지 않기 위함)
                if not in_string_value and char == ':' and i + 1 < len(json_str_slice):
                    next_char_index = i + 1
                    # 콜론 뒤의 공백 무시
                    while json_str_slice[next_char_index].isspace():
                        repaired_json += json_str_slice[next_char_index]
                        next_char_index += 1

                    if json_str_slice[next_char_index] == '"':
                        repaired_json += json_str_slice[next_char_index]
                        in_string_value = True
                        i = next_char_index

                # 문자열 값 내부를 순회하며 이스케이프되지 않은 따옴표를 찾아 복구
                elif in_string_value and char == '"':
                    if json_str_slice[i-1] != '\\': # 이미 이스케이프된 경우는 제외
                        # 이 따옴표가 문자열을 닫는 것인지 확인
                        # (뒤에 쉼표나 닫는 괄호가 오면 닫는 따옴표로 간주)
                        peek_index = i + 1
                        while peek_index < len(json_str_slice) and json_str_slice[peek_index].isspace():
                            peek_index += 1

                        if peek_index < len(json_str_slice) and json_str_slice[peek_index] in [',', '}', ']']:
                             in_string_value = False # 문자열 값 종료
                        else:
                            # 닫는 따옴표가 아니라면, 내용물 따옴표이므로 이스케이프
                            repaired_json = repaired_json[:-1] + '\\"'

                i += 1

            # 복구된 문자열로 최종 파싱
            return json.loads(repaired_json)

    except (json.JSONDecodeError, IndexError):
        return None
    except Exception:
        return None

def highlight_and_capture_cropped(url: str, search_text: str, padding: int = 400) -> Optional[Image.Image]:
    """
    [최종 완성형] 파이썬(BeautifulSoup)이 HTML을 직접 분석하고 수정한 뒤,
    그 결과를 브라우저에 덮어씌우는 가장 확실한 하이라이팅 방식입니다.
    """

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument("window-size=1920,1080")
    chrome_options.add_argument("lang=ko_KR")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")

    driver = None
    try:
        # ------------------------------------------------------------------
        # 1. 브라우저에서 기사 본문의 원본 HTML 코드를 가져옵니다.
        # ------------------------------------------------------------------
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        article_selector = '#dic_area, #newsct_article, div.article_body_content'
        try:
            article_element = driver.find_element(By.CSS_SELECTOR, article_selector)
            original_html = article_element.get_attribute('innerHTML')
        except:
            print("  - 오류: 기사 본문 영역을 찾지 못했습니다.")
            return None

        # ------------------------------------------------------------------
        # 2. 파이썬(BeautifulSoup)이 HTML을 정밀 분석하고 <mark> 태그를 삽입합니다.
        # ------------------------------------------------------------------
        soup = BeautifulSoup(original_html, 'html.parser')

        # 정규화 함수: 비교를 위해 공백과 일부 특수문자를 제거
        def normalize_text(text):
            return re.sub(r'[\s"“”‘’]', '', text)

        clean_search_text = normalize_text(search_text)

        # 모든 텍스트 노드를 찾아서, 그 내용을 합친 전체 텍스트를 만듭니다.
        all_text_nodes = soup.find_all(string=True)
        full_text = "".join(all_text_nodes)
        clean_full_text = normalize_text(full_text)

        # 전체 텍스트에서 찾으려는 문장의 시작 위치를 찾습니다.
        start_index = clean_full_text.find(clean_search_text)

        if start_index == -1:
            print(f"  - 경고: 파이썬 분석 단계에서 텍스트 '{search_text}'를 찾지 못했습니다.")
            # 하이라이팅 없이 스크린샷 캡처
            png = driver.get_screenshot_as_png()
            return Image.open(BytesIO(png))

        end_index = start_index + len(clean_search_text)

        # 시작/종료 위치를 이용해 실제 하이라이팅할 텍스트 노드들을 찾습니다.
        char_count = 0
        nodes_to_highlight = []
        for node in all_text_nodes:
            node_text_clean = normalize_text(node.string)
            node_len_clean = len(node_text_clean)

            # 현재 노드가 하이라이트 범위에 포함되는지 확인
            if char_count < end_index and char_count + node_len_clean > start_index:
                nodes_to_highlight.append(node)

            char_count += node_len_clean
            if char_count >= end_index:
                break

        # 찾은 노드들을 <mark> 태그로 감싸줍니다.
        for node in nodes_to_highlight:
            if isinstance(node, NavigableString) and node.parent.name != 'script':
                mark_tag = soup.new_tag("mark", style="background-color: yellow; color: black;")
                mark_tag.string = node.string
                node.replace_with(mark_tag)

        # 수정된 HTML 코드를 문자열로 다시 가져옵니다.
        highlighted_html = str(soup)

        # ------------------------------------------------------------------
        # 3. 브라우저에 수정된 HTML 코드를 덮어씌웁니다.
        # ------------------------------------------------------------------
        # Lazy Loading 처리 및 폰트 주입
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)
        driver.execute_script("window.scrollTo(0, 0);")
        driver.execute_script("""
            var style = document.createElement('style');
            style.innerHTML = `@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');
                               * { font-family: 'Noto Sans KR', '맑은 고딕', sans-serif !important; }`;
            document.head.appendChild(style);
        """)

        WebDriverWait(driver, 10).until(
            # 바깥쪽은 ()로 감싸고, 자바스크립트 문자열은 ""와 ''를 섞어 사용
            lambda d: d.execute_script("return document.fonts.check('1em \"Noto Sans KR\"')")
        )

        # 자바스크립트는 이제 계산 없이, 수정된 HTML을 '덮어쓰는' 역할만 합니다.
        driver.execute_script(f"""
            const articleBody = document.querySelector('{article_selector}');
            if (articleBody) {{
                articleBody.innerHTML = arguments[0];
                const mark = articleBody.querySelector('mark');
                if (mark) {{
                    mark.scrollIntoView({{ block: 'center', behavior: 'instant' }});
                }}
            }}
        """, highlighted_html)

        # ------------------------------------------------------------------
        # 4. 스크린샷을 찍고 이미지를 잘라냅니다.
        # ------------------------------------------------------------------
        time.sleep(2) # 렌더링 대기

        # mark 태그의 위치를 브라우저에서 직접 계산
        rect_script = """
        const mark = document.querySelector('mark');
        if (!mark) return null;
        const rect = mark.getBoundingClientRect();
        return { top: rect.top + window.scrollY, bottom: rect.bottom + window.scrollY };
        """
        rect = driver.execute_script(rect_script)

        if not rect:
             raise Exception("하이라이팅은 성공했으나, 화면 위치를 계산할 수 없습니다.")

        total_height = driver.execute_script("return document.body.parentNode.scrollHeight")
        driver.set_window_size(1920, total_height)
        time.sleep(1)
        png = driver.get_screenshot_as_png()
        full_screenshot = Image.open(BytesIO(png))

        top = max(0, int(rect['top'] - padding))
        bottom = min(full_screenshot.height, int(rect['bottom'] + padding))

        cropped_image = full_screenshot.crop((0, top, full_screenshot.width, bottom))
        print("  - 선정된 뉴스 기사 중 매매의 근본 원인에 해당하는 핵심 문장을 하이라이팅하였습니다.")
        return cropped_image

    except Exception as e:
        print(f"\nERROR: highlight_and_capture_cropped 처리 중 심각한 오류 발생: {e}")
        traceback.print_exc()
        # 오류 발생 시 비상용으로 전체 페이지 캡처
        if driver:
            try:
                return Image.open(BytesIO(driver.get_screenshot_as_png()))
            except:
                return None
        return None
    finally:
        if driver:
            driver.quit()

# --- 에이전트 노드 함수들 ---
def extract_transactions(state: AgentState) -> Dict[str, Any]:
    print("\n[PTPRA Agent] 1. 사용자 입력에서 매매 기록 추출 중...")
    user_input = state["query"]
    transaction_records = []
    json_start = user_input.find('[')
    json_end = user_input.rfind(']')
    if json_start != -1 and json_end != -1 and json_end > json_start:
        json_str = user_input[json_start : json_end + 1]
        try:
            parsed_data = json.loads(json_str)
            for record in parsed_data:
                transaction_records.append({
                    "time": datetime.strptime(record["날짜"], "%Y-%m-%d"),
                    "type": record["거래유형"],
                    "stock_name": record["종목"],
                    "price": record["가격"],
                    "quantity": record["수량"]
                })
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            state["error"] = f"매매 기록 JSON 파싱 중 오류 발생: {e}"
            return state
    if not transaction_records:
        state["error"] = "유효한 매매 기록을 추출할 수 없습니다."
    else:
        state["transaction_records"] = sorted(transaction_records, key=lambda x: x['time'])
        print(f"  - {len(transaction_records)}개의 매매 기록 추출 완료.")
    return state

def extract_mydata(state: AgentState) -> Dict[str, Any]:
    print("\n[PTPRA Agent] 2. 사용자 입력에서 마이데이터 추출 중...")
    user_input = state["query"]
    llm = state["llm"]
    system_msg = "당신은 주어진 텍스트에서 사용자의 금융 관련 개인정보를 정확히 추출하여 JSON 형식으로 구조화하는 전문가입니다."
    prompt = prompt_extract_mydata.format(user_input=user_input)
    try:
        response = llm.invoke(prompt, system_message=system_msg)
        my_data = _extract_json_from_llm_response(response)
        if not my_data or my_data.get('age') is None or my_data.get('investment_profile') is None or my_data.get('total_financial_assets') is None:
            state["error"] = "개인화 분석에 필요한 마이데이터(투자 성향, 나이, 총 금융자산) 정보가 부족합니다."
            return state
        state["my_data"] = my_data
        print(f"  - 마이데이터 추출 완료: {my_data}")
    except Exception as e:
        state["error"] = f"마이데이터 추출 중 LLM 호출 오류 발생: {e}"
    return state

def analyze_risk_patterns(state: AgentState) -> Dict[str, Any]:
    print("\n[PTPRA Agent] 3. 개인화된 위험 기준 적용 및 패턴 분석 중...")
    transactions = state["transaction_records"]
    my_data = state["my_data"]
    holdings = {}
    for trade in transactions:
        stock_name = trade['stock_name']
        qty = trade['quantity']
        price = trade['price']
        trade_value = qty * price
        if trade['type'] == '매수':
            if stock_name not in holdings:
                holdings[stock_name] = {'quantity': 0, 'value': 0}
            holdings[stock_name]['quantity'] += qty
            holdings[stock_name]['value'] += trade_value
        elif trade['type'] == '매도':
            if stock_name in holdings:
                holdings[stock_name]['quantity'] -= qty
                holdings[stock_name]['value'] -= trade_value
                if holdings[stock_name]['quantity'] <= 0:
                    del holdings[stock_name]
        if stock_name in holdings:
            holdings[stock_name]['last_trade'] = trade
    final_portfolio_value = sum(h['value'] for h in holdings.values())
    if final_portfolio_value <= 0:
        state["final_alert_message"] = "모든 종목이 매도되어 분석할 현재 보유 포트폴리오가 없습니다."
        return state
    age = my_data['age']
    profile = my_data['investment_profile']
    if 20 <= age < 40: lifecycle_coeff = 0.30
    elif 40 <= age < 60: lifecycle_coeff = 0.20
    else: lifecycle_coeff = 0.10
    profile_map = {"안정형": 0.20, "안정추구형": 0.40, "위험중립형": 0.60, "적극투자형": 0.80, "공격투자형": 0.90}
    equity_limit_ratio = profile_map.get(profile, 0.60)
    personalized_threshold = equity_limit_ratio * lifecycle_coeff
    print(f"  - 개인화된 단일 종목 최대 비중 임계치: {personalized_threshold:.2%}")
    riskiest_stock = None
    max_concentration = 0
    for name, holding_info in holdings.items():
        concentration = holding_info['value'] / final_portfolio_value
        if concentration > personalized_threshold and concentration > max_concentration:
            max_concentration = concentration
            riskiest_stock = name
    state['preprocessed_data'] = {'holdings': holdings, 'final_portfolio_value': final_portfolio_value}
    if riskiest_stock:
        risk_pattern = {
            "risk_category": "집중 투자 위험",
            "stock_name": riskiest_stock,
            "concentration": max_concentration,
            "description": f"'{riskiest_stock}' 종목의 비중이 {max_concentration:.2%}로, 고객님의 프로필(나이: {age}세, 성향: {profile})에 따른 권장 한도 {personalized_threshold:.2%}를 초과한 상황입니다.",
            "recommendation": "특정 종목에 대한 과도한 투자는 해당 종목의 가격 변동에 포트폴리오 전체가 크게 흔들릴 수 있습니다. 분산 투자를 통해 안정성을 높이는 것을 고려해보세요."
        }
        trigger_trade = holdings[riskiest_stock]['last_trade']
        state["identified_risk_pattern"] = risk_pattern
        state["triggering_trade_info"] = trigger_trade
        stock_name = trigger_trade['stock_name']
        causal_keywords = ["실적", "계약", "신제품", "목표주가", "컨센서스", "전망"]
        search_queries = [f"{stock_name} {keyword}" for keyword in causal_keywords]
        search_queries.append(f"{stock_name} 주가")
        state["search_queries"] = search_queries
        print(f"  - 최종 포트폴리오 분석 결과, '{riskiest_stock}'에서 집중 투자 위험 감지.")
    else:
        print("  - 최종 포트폴리오 분석 결과, 개인화 기준을 초과하는 심각한 위험 패턴은 발견되지 않았습니다.")
        state["final_alert_message"] = "매매 기록을 분석한 결과, 고객님의 투자 성향과 나이를 고려했을 때 특별히 우려되는 위험 패턴은 발견되지 않았습니다. 안정적인 투자를 이어가고 계십니다."
    return state

def verify_analysis_results(state: AgentState) -> Dict[str, Any]:
    if not state.get("identified_risk_pattern"):
        state['is_verified'] = True
        return state
    print("\n[PTPRA Agent] 4. 분석 결과 검증 중...")
    risk_pattern = state["identified_risk_pattern"]
    holdings = state['preprocessed_data']['holdings']
    final_portfolio_value = state['preprocessed_data']['final_portfolio_value']
    stock_name_to_verify = risk_pattern['stock_name']
    reported_concentration = risk_pattern['concentration']
    actual_value = holdings.get(stock_name_to_verify, {}).get('value', 0)
    actual_concentration = actual_value / final_portfolio_value if final_portfolio_value > 0 else 0
    if round(reported_concentration, 4) == round(actual_concentration, 4):
        print(f"  - ✅ 검증 성공: 보고된 비중({reported_concentration:.2%})이 실제 비중({actual_concentration:.2%})과 일치합니다.")
        state['is_verified'] = True
    else:
        error_message = f"검증 실패: 보고된 '{stock_name_to_verify}' 비중({reported_concentration:.2%})이 실제 계산된 비중({actual_concentration:.2%})과 다릅니다."
        print(f"  - ❌ {error_message}")
        state['error'] = error_message
        state['is_verified'] = False
    return state

def search_korean_documents(state: AgentState) -> Dict[str, Any]:
    """
    네이버 뉴스 API를 사용해 원인 추정을 위한 문서를 검색하고,
    매매일 이전 뉴스를 우선적으로 필터링합니다.
    """
    if not state.get("search_queries"):
        return state

    print(f"\n[PTPRA Agent] 5. 원인 추정을 위한 뉴스 검색 및 필터링 중...")
    
    # --- API 설정 ---
    NAVER_CLIENT_ID = "xQgQRIqSx5_3k2rthBtY"
    NAVER_CLIENT_SECRET = "QqUBAzOagA"
    headers = {"X-Naver-Client-Id": NAVER_CLIENT_ID, "X-Naver-Client-Secret": NAVER_CLIENT_SECRET}
    
    all_results = []
    seen_urls = set()

    for query in state["search_queries"]:
        url = f"https://openapi.naver.com/v1/search/news.json?query={requests.utils.quote(query)}&display=10&sort=sim"
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            if data and "items" in data:
                for item in data["items"]:
                    link = item.get("link", "")
                    # Naver 뉴스 링크만 포함하고, 중복 URL 제거
                    if "n.news.naver.com" not in link or link in seen_urls:
                        continue
                    
                    pub_date_str = item.get("pubDate")
                    pub_date = None
                    if pub_date_str:
                        try:
                            # 시간대 정보를 포함하여 datetime 객체로 변환 후, 시간대 정보 제거 (naive datetime으로 통일)
                            dt_object = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %z")
                            pub_date = dt_object.replace(tzinfo=None)
                        except ValueError:
                            pass # 날짜 파싱 실패 시 None 유지
                    
                    result = {
                        "query": query,
                        "snippet": BeautifulSoup(item.get("description", ""), 'html.parser').get_text(),
                        "source_title": BeautifulSoup(item.get("title", ""), 'html.parser').get_text(),
                        "url": link,
                        "pub_date": pub_date
                    }
                    all_results.append(result)
                    seen_urls.add(result["url"])
        except Exception as e:
            # 실패하더라도 다른 쿼리 검색은 계속 진행
            print(f"  - 경고: '{query}' 검색 중 오류 발생 - {e}")

    if not all_results:
        state["error"] = "관련 뉴스를 찾을 수 없습니다."
        return state

    trade_date = state["triggering_trade_info"]['time']
    
    # 1. 매수일 기준(-7일 ~ 당일)으로 우선 필터링
    start_date = trade_date - timedelta(days=7)
    end_date = trade_date
    filtered_results = [res for res in all_results if res["pub_date"] and (start_date <= res["pub_date"] <= end_date)]

    # 2. 지정된 기간 내 뉴스가 없을 경우, 대체(Fallback) 로직 실행
    if not filtered_results and all_results:
        print("  - 경고: 지정된 기간 내 뉴스가 없습니다. 매매일과 가장 가까운 '미래' 뉴스를 선택합니다.")

        # 2-1. 매매일 '이전'의 전체 과거 기사 중 가장 가까운 것을 탐색
        past_results = [res for res in all_results if res.get("pub_date") and res['pub_date'] <= trade_date]
        if past_results:
            closest_news = min(past_results, key=lambda x: trade_date - x['pub_date'])
            filtered_results = [closest_news]
        else:
            # 2-2. 과거 기사가 전혀 없을 경우에만 '미래' 기사 중 가장 가까운 것을 탐색
            future_results = [res for res in all_results if res.get("pub_date") and res['pub_date'] > trade_date]
            if future_results:
                closest_news = min(future_results, key=lambda x: x['pub_date'] - trade_date)
                filtered_results = [closest_news]
                # 최종 리포트에 이 한계점을 명시할 수 있도록 상태에 기록
                state["analysis_limitation"] = "매매 시점 이전의 직접적인 원인 뉴스를 찾지 못해, 시점과 가장 가까운 최신 뉴스를 기반으로 분석했습니다."

    state["search_results_korean"] = filtered_results
    if not filtered_results:
        state["error"] = "매매 시점과 연관성 높은 뉴스를 최종적으로 찾을 수 없습니다."
        
    return state

# --- [신규] 뉴스 기사 원문 스크래핑 함수 ---
# [수정 후] get_full_text_from_url 함수 (Selenium 사용)
def get_full_text_from_url(url: str) -> Optional[str]:
    """
    [최종 수정안] Selenium을 사용하여 브라우저가 렌더링한 최종 텍스트를 추출합니다.
    이로써 하이라이팅 시점의 텍스트와 100% 일관성을 보장합니다.
    """
    
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")

    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        # 페이지 로딩을 충분히 기다립니다.
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # 기사 본문을 선택합니다.
        article_selector = '#dic_area, #newsct_article, div.article_body_content'

        # 자바스크립트를 실행하여 브라우저가 보는 그대로의 텍스트(.innerText)를 가져옵니다.
        # .innerText는 줄바꿈, 공백 등을 가장 정확하게 처리해줍니다.
        script = f"return document.querySelector('{article_selector}').innerText;"
        full_text = driver.execute_script(script)

        if full_text:
            # 여러 개의 공백을 하나로 합치는 최종 정리 작업
            normalized_text = re.sub(r'\s+', ' ', full_text).strip()
            print(f"  - [2단계 분석 완료]. Selenium으로 스크래핑 성공: {len(normalized_text)}자 텍스트 확보.")
            return normalized_text
        else:
            print("  - 경고: 뉴스 본문 컨텐츠 영역을 찾지 못했습니다.")
            return None

    except Exception as e:
        print(f"  - 오류: Selenium 기반 뉴스 원문 스크래핑 중 오류 발생 ({e})")
        # traceback.print_exc() # 디버깅 시 사용
        return None
    finally:
        if driver:
            driver.quit()

def analyze_and_extract_from_news(state: AgentState) -> Dict[str, Any]:
    """
    [수정된 버전]
    가장 관련성 높은 뉴스를 선정하고, 그 뉴스 원문에서 '핵심 문장을 포함하는 문단 전체'를 추출합니다.
    """
    if not state.get("search_results_korean"):
        return state

    llm = state["llm"]

    age = state["my_data"]["age"]
    investment_profile = state["my_data"]["investment_profile"]
    trade_time = state["triggering_trade_info"]['time'].strftime('%Y년 %m월 %d일')
    stock_name = state["triggering_trade_info"]['stock_name']
    type = state["triggering_trade_info"]['type']
    
    search_results = state["search_results_korean"]

    # === 1단계: 가장 관련성 높은 뉴스 기사 URL과 '근본 원인' 요약 ===
    print("\n[PTPRA Agent] 6a. 매매 원인 추정 (1/3) - 최적 뉴스 기사 선정 중...")

    search_summary = ""
    for i, res in enumerate(search_results):
        search_summary += f"{i+1}. 제목: {res['source_title']}\n   내용: {res['snippet']}\n   게시일: {res['pub_date']}\n   URL: {res['url']}\n\n"

    system_msg_step1 = "당신은 금융 애널리스트입니다. 주어진 정보를 바탕으로, 특정 투자 결정의 가장 유력한 '근본 원인'을 찾아내는 임무를 받았습니다."
    prompt_step1 = prompt_summarize_reason.format(age=age,
                                                  investment_profile=investment_profile,
                                                  trade_time=trade_time,
                                                  stock_name=stock_name,
                                                  type=type,
                                                  search_summary=search_summary,
                                                )

    try:
        response_step1 = llm.invoke(prompt_step1, system_message=system_msg_step1)
        analysis_result_step1 = _extract_json_from_llm_response(response_step1)

        if not (analysis_result_step1 and analysis_result_step1.get("selected_url") and analysis_result_step1.get("reason_summary")):
            state["error"] = f"뉴스 분석 1단계(선정) LLM 응답 파싱 실패: {response_step1[:500]}..."
            return state

        selected_url = analysis_result_step1["selected_url"]
        reason_summary = analysis_result_step1["reason_summary"]
        state["selected_document_url"] = selected_url
        state["news_analysis_result"] = analysis_result_step1
        print(f"  - [1단계 분석 완료]. 선택된 URL: {selected_url}")

    except Exception as e:
        state["error"] = f"뉴스 분석 1단계(선정) 중 LLM 호출 오류: {e}"
        return state

    # === 2단계: 선택된 URL에서 뉴스 원문 전체 스크래핑 ===
    print("\n[PTPRA Agent] 6b. 매매 원인 추정 (2/3) - 뉴스 원문 확보 중...")
    full_text = get_full_text_from_url(selected_url)
    if not full_text:
        state["error"] = f"선택된 뉴스 기사({selected_url})의 원문을 가져오는 데 실패했습니다."
        return state

    # === 3단계: 핵심 문장을 포함하는 '문단 전체' 추출 ===
    print("\n[PTPRA Agent] 6c. 매매 원인 추정 (3/3) - 핵심 문단 추출 중...")

    system_msg_step2 = "당신은 주어진 글에서 특정 정보와 가장 관련 깊은 부분을 **변경 없이 그대로** 찾아내는, 정확성이 매우 높은 AI입니다."
    prompt_step2 = prompt_extract_paragraph.format(reason_summary=reason_summary, full_text=full_text)

    try:
        # 1. LLM을 호출하여 관련 깊은 부분에 대한 '단서' 텍스트를 얻음
        response_step2 = llm.invoke(prompt_step2, system_message=system_msg_step2)
        analysis_result_step2 = _extract_json_from_llm_response(response_step2)

        if not (analysis_result_step2 and analysis_result_step2.get("containing_paragraph")):
            state["error"] = f"뉴스 분석 3단계(문단 추출) LLM 응답 파싱 실패: {response_step2[:500]}..."
            return state

        # 2. LLM이 추출한 텍스트를 원본에서 찾기 위한 '단서'로 사용
        llm_extracted_clue = analysis_result_step2["containing_paragraph"]

        # 3. '단서'와 원본 텍스트(full_text)를 비교하여 가장 유사한 '원본 문장'을 찾음
        verified_paragraph = _find_best_matching_chunk(
            llm_output=llm_extracted_clue,
            original_text=full_text
        )
        state["extracted_important_sentence"] = verified_paragraph
        state["news_analysis_result"]["extracted_sentence"] = verified_paragraph
        print(f"  - [3단계 분석 완료]. 발췌된 원문 문장: {verified_paragraph}")

    except Exception as e:
        state["error"] = f"뉴스 분석 3단계(문단 추출) 중 오류 발생: {e}"
        return state

    return state

def capture_highlighted_image(state: AgentState) -> Dict[str, Any]:
    if not state.get("selected_document_url") or not state.get("extracted_important_sentence"):
        return state
    print("\n[PTPRA Agent] 7. 뉴스 원문 하이라이팅 및 캡처 중...")
    url = state["selected_document_url"]
    sentence = state["extracted_important_sentence"]

    captured_image = highlight_and_capture_cropped(url, sentence)
    
    if captured_image:
        # 상태에서 run_id 가져오기 (없을 경우를 대비해 uuid 사용)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # 파일명으로 부적합한 문자 제거 또는 변경
        safe_filename = re.sub(r'[\\/*?:"<>|()]', '', str(run_id))
        # 'results/task5' 디렉토리가 없으면 생성
        output_dir = "results/task5"
        os.makedirs(output_dir, exist_ok=True)
        image_path = os.path.join(output_dir, f"{safe_filename}.png")

        captured_image.save(image_path)
        state["screenshot_image_path"] = image_path
    else:
        print("  - 경고: 이미지 캡처에 실패했습니다. 텍스트 응답만 생성됩니다.")
        state["screenshot_image_path"] = None
    return state

def generate_final_response(state: AgentState) -> Dict[str, Any]:
    if state.get("final_alert_message"):
        print("\n[PTPRA Agent] 8. 최종 응답 생성 중 (위험 없음)...")
        return state
    print("\n[PTPRA Agent] 8. 최종 응답 메시지 생성 중...")
    if state.get("error"):
        state["final_alert_message"] = f"죄송합니다. 요청 처리 중 오류가 발생했습니다: {state['error']}"
        return state
    trade = state["triggering_trade_info"]
    risk = state["identified_risk_pattern"]
    news_analysis = state["news_analysis_result"]
    part1_recognition = f"최근 **{trade['time'].strftime('%Y년 %m월 %d일')}**에 진행하신 **'{trade['stock_name']}' {trade['type']}** 기록을 확인했습니다."
    part2_reason_empathy = ""
    if news_analysis and news_analysis.get('reason_summary') and news_analysis.get('extracted_sentence'):
        part2_reason_empathy = f"고객님의 이러한 결정은 아마도 **'{news_analysis['reason_summary']}'** 관련 소식 때문이었을 것으로 생각됩니다. 당시 뉴스에서는 '{news_analysis['extracted_sentence']}'라며 긍정적인 전망을 내놓았죠."
    part3_risk_alert = f"하지만 고객님의 프로필(나이: {state['my_data']['age']}세, 성향: {state['my_data']['investment_profile']})을 기준으로 분석한 결과, 현재 **'{risk['risk_category']}'** 상태에 해당합니다. {risk['description']}"
    part4_future_warning = f"이러한 집중 투자 패턴이 지속될 경우, 해당 종목의 작은 변동성에도 전체 자산이 크게 영향을 받을 수 있으며, 예상치 못한 하락 시 큰 손실로 이어질 위험이 있습니다. {risk['recommendation']}"
    final_message_parts = [part1_recognition, part2_reason_empathy, part3_risk_alert, part4_future_warning]
    if state.get("screenshot_image_path"):
        final_message_parts.append(f"\n**참고 뉴스 기사의 핵심 내용:**")
        final_message_parts.append(f"![강조된 이미지]({state['screenshot_image_path']})")
        if state.get("selected_document_url"):
            final_message_parts.append(f"원본 자료: {state['selected_document_url']}")
    state["result"] = "\n\n".join(filter(None, final_message_parts))
    return state

def handle_error(state: AgentState) -> Dict[str, Any]:
    print(f"\n[PTPRA Agent] 오류 발생: {state.get('error', '알 수 없는 오류')}")
    state["final_alert_message"] = f"오류가 발생하여 요청을 완료할 수 없습니다: {state.get('error', '알 수 없는 오류')}. 다시 시도해주세요."
    return state

def generate_final_response(state: AgentState) -> Dict[str, Any]:
    if state.get("final_alert_message"):
        print("\n[PTPRA Agent] 8. 최종 응답 생성 중 (위험 없음)...")
        return state
    print("\n[PTPRA Agent] 8. 최종 응답 메시지 생성 중...")
    if state.get("error"):
        state["final_alert_message"] = f"죄송합니다. 요청 처리 중 오류가 발생했습니다: {state['error']}"
        return state
    trade = state["triggering_trade_info"]
    risk = state["identified_risk_pattern"]
    news_analysis = state["news_analysis_result"]
    part1_recognition = f"최근 **{trade['time'].strftime('%Y년 %m월 %d일')}**에 진행하신 **'{trade['stock_name']}' {trade['type']}** 기록을 확인했습니다."
    part2_reason_empathy = ""
    if news_analysis and news_analysis.get('reason_summary') and news_analysis.get('extracted_sentence'):
        part2_reason_empathy = f"고객님의 이러한 결정은 아마도 **'{news_analysis['reason_summary']}'** 관련 소식 때문이었을 것으로 생각됩니다. 당시 뉴스에서는 '{news_analysis['extracted_sentence']}'라며 긍정적인 전망을 내놓았죠."
    part3_risk_alert = f"하지만 고객님의 프로필(나이: {state['my_data']['age']}세, 성향: {state['my_data']['investment_profile']})을 기준으로 분석한 결과, 현재 **'{risk['risk_category']}'** 상태에 해당합니다. {risk['description']}"
    part4_future_warning = f"이러한 집중 투자 패턴이 지속될 경우, 해당 종목의 작은 변동성에도 전체 자산이 크게 영향을 받을 수 있으며, 예상치 못한 하락 시 큰 손실로 이어질 위험이 있습니다. {risk['recommendation']}"
    final_message_parts = [part1_recognition, part2_reason_empathy, part3_risk_alert, part4_future_warning]
    if state.get("screenshot_image_path"):
        image_url = create_shareable_url(state["screenshot_image_path"], host_ip=config['host_ip'], port=config['port'])
        if image_url:
            final_message_parts.append(f"\n하이라이팅 이미지: {image_url}")
    if state.get("selected_document_url"):
        final_message_parts.append(f"원본 뉴스: {state['selected_document_url']}")
    state["result"] = "\n".join(filter(None, final_message_parts))
    return state


# --- 4. PTPRA 서브그래프 빌더 ---

def task5_graph():
    """개인별 매매 패턴 위험 알림(PTPRA)을 위한 서브그래프를 빌드하고 컴파일합니다."""
    workflow = StateGraph(AgentState)
    

    workflow.add_node("extract_transactions", extract_transactions)
    workflow.add_node("extract_mydata", extract_mydata)
    workflow.add_node("analyze_risk_patterns", analyze_risk_patterns)
    workflow.add_node("verify_analysis_results", verify_analysis_results)
    workflow.add_node("search_korean_documents", search_korean_documents)
    workflow.add_node("analyze_and_extract_from_news", analyze_and_extract_from_news)
    workflow.add_node("capture_highlighted_image", capture_highlighted_image)
    workflow.add_node("generate_final_response", generate_final_response)
    workflow.add_node("handle_error", handle_error)
    workflow.set_entry_point("extract_transactions")
    workflow.add_edge("extract_transactions", "extract_mydata")
    workflow.add_edge("extract_mydata", "analyze_risk_patterns")
    workflow.add_edge("analyze_risk_patterns", "verify_analysis_results")

    workflow.set_entry_point("extract_transactions")

    def after_verification(state: AgentState) -> Dict[str, Any]:
        if not state.get('is_verified', False): return "handle_error"
        if state.get("search_queries"): return "search_korean_documents"
        else: return "generate_final_response"

    workflow.add_conditional_edges(
        "verify_analysis_results",
        after_verification,
        {"search_korean_documents": "search_korean_documents", "generate_final_response": "generate_final_response", "handle_error": "handle_error"}
    )
    # [수정] 엣지 연결 변경
    workflow.add_edge("search_korean_documents", "analyze_and_extract_from_news")
    workflow.add_edge("analyze_and_extract_from_news", "capture_highlighted_image")
    workflow.add_edge("capture_highlighted_image", "generate_final_response")
    workflow.add_edge("generate_final_response", END)
    workflow.add_edge("handle_error", END)
    
    return workflow.compile()