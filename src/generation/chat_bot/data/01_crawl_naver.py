"""
네이버 플레이스 크롤러 v6 - 완전판

추가 기능:
1. 별점 None → 추정 점수 생성
2. 업체 소개글 크롤링
3. 실제 리뷰 텍스트 3-5개
4. 업체 등록 사진 URL
"""

import os
import re
import json
import time
import random
import hashlib
import requests
import tempfile
import urllib.parse
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

# API 키
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
    print("❌ 환경변수 설정 필요!")
    exit(1)

# 설정
TARGET_COUNT = 1000  # 전국 확대로 1000개
DISPLAY_MAX = 5
REQUEST_SLEEP = 0.3
SELENIUM_SLEEP = 3.0
MAX_RETRIES = 2

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

# 지역/키워드 (전국 확대)
LOCATIONS = [
    # 서울 주요 지역
    "강남", "홍대", "신촌", "이태원", "성수", "연남", "망원", "한남", "청담", "압구정",
    "건대", "잠실", "신림", "노원", "강서", "마포", "용산", "종로", "명동", "을지로",
    
    # 경기 주요 지역
    "판교", "분당", "수원", "일산", "안양", "부천", "평촌", "의정부", "고양", "성남",
    "용인", "화성", "김포", "광명", "과천", "시흥",
    
    # 인천
    "송도", "구월동", "부평", "계양", "인천",
    
    # 부산
    "서면", "해운대", "광안리", "남포동", "부산대", "부산",
    
    # 대구
    "동성로", "수성구", "범어동", "대구",
    
    # 대전
    "둔산", "유성", "대전",
    
    # 광주
    "충장로", "상무지구", "광주",
    
    # 울산
    "삼산동", "울산",
    
    # 세종
    "세종",
    
    # 강원
    "춘천", "강릉", "속초", "원주",
    
    # 충청
    "천안", "청주", "아산", "충주",
    
    # 전라
    "전주", "익산", "군산", "목포", "순천", "여수",
    
    # 경상
    "포항", "경주", "구미", "창원", "진주", "김해", "거제",
    
    # 제주
    "제주시", "서귀포", "제주"
]

KEYWORDS = {
    "카페": [
        "카페", "커피", "커피숍",
        "브런치카페", "디저트카페", "베이커리카페",
        "감성카페", "뷰맛집", "루프탑카페",
        "애견카페", "북카페", "작업카페"
    ],
    "맛집": [
        "맛집", "식당", "현지인맛집",
        "한식당", "중식당", "일식당", "양식당",
        "고기집", "회집", "치킨집",
        "분식집", "국밥집", "찌개전골",
        "술집", "포차", "이자카야",
        "브런치", "뷔페"
    ],
    "베이커리": [
        "베이커리", "빵집", "제과점",
        "케이크샵", "디저트샵", "마카롱",
        "도넛", "크로와상", "스콘전문점"
    ]
}

# ============================================
# 네이버 API
# ============================================
@dataclass
class NaverPlace:
    title: str
    link: str
    category: str
    address: str
    roadAddress: str
    telephone: str
    mapx: str
    mapy: str


class NaverAPICollector:
    BASE_URL = "https://openapi.naver.com/v1/search/local.json"

    def __init__(self, client_id: str, client_secret: str):
        self.session = requests.Session()
        self.session.headers.update({
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret,
        })

    def search(self, query: str, display: int = 5) -> List[NaverPlace]:
        params = {"query": query, "display": min(display, DISPLAY_MAX), "start": 1, "sort": "random"}

        try:
            resp = self.session.get(self.BASE_URL, params=params, timeout=15)
            if resp.status_code != 200:
                return []

            data = resp.json()
            items = []
            
            for it in data.get("items", []):
                title = re.sub(r'<\/?b>', '', it.get("title", "")).strip()
                items.append(NaverPlace(
                    title=title,
                    link=it.get("link", ""),
                    category=it.get("category", ""),
                    address=it.get("address", ""),
                    roadAddress=it.get("roadAddress", ""),
                    telephone=it.get("telephone", ""),
                    mapx=str(it.get("mapx", "")),
                    mapy=str(it.get("mapy", "")),
                ))

            time.sleep(REQUEST_SLEEP)
            return items
        
        except:
            return []


# ============================================
# Selenium - 완전판
# ============================================
class CompleteCrawler:
    
    def __init__(self, headless: bool = True):
        self.driver = None
        self._init_driver(headless)
    
    def _init_driver(self, headless: bool):
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        
        options = Options()
        
        if headless:
            options.add_argument('--headless=new')
        
        # 기본 옵션
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        # 봇 탐지 우회
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # 데스크톱 User-Agent
        options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        profile_dir = tempfile.mkdtemp(prefix="chrome-")
        options.add_argument(f'--user-data-dir={profile_dir}')
        
        # Mac Chrome
        mac_chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if os.path.exists(mac_chrome):
            options.binary_location = mac_chrome
        
        # webdriver-manager
        from webdriver_manager.chrome import ChromeDriverManager
        driver_path = ChromeDriverManager().install()
        
        service = Service(driver_path)
        self.driver = webdriver.Chrome(service=service, options=options)
        
        # webdriver 속성 숨기기
        self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            '''
        })
        
        print("  ✅ Selenium 초기화 (완전판)")
    
    def scrape(self, title: str, address: str, idx: int) -> Dict:
        """
        모든 정보 수집
        """
        for attempt in range(MAX_RETRIES):
            try:
                result = self._scrape_once(title, address, idx, attempt)
                if result["success"] or attempt == MAX_RETRIES - 1:
                    return result
                
                print(f"  재시도 {attempt + 1}/{MAX_RETRIES}...", end=" ")
                time.sleep(2)
            
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"⚠️ 최종 실패: {str(e)[:50]}")
                    return {
                        "review_count": None,
                        "rating": None,
                        "estimated_rating": None,
                        "business_description": None,
                        "sample_reviews": [],
                        "business_photos": [],
                        "success": False
                    }
                time.sleep(2)
        
        return {
            "review_count": None,
            "rating": None,
            "estimated_rating": None,
            "business_description": None,
            "sample_reviews": [],
            "business_photos": [],
            "success": False
        }
    
    def _scrape_once(self, title: str, address: str, idx: int, attempt: int) -> Dict:
        """
        실제 크롤링 로직 (확장판)
        """
        result = {
            "review_count": None,
            "rating": None,
            "estimated_rating": None,
            "business_description": None,
            "sample_reviews": [],
            "business_photos": [],
            "success": False
        }
        
        query = title.strip()
        
        if attempt == 0:
            print(f"검색: {query[:30]}", end=" → ")
        
        # 데스크톱 검색 URL
        url = f"https://map.naver.com/v5/search/{urllib.parse.quote(query)}"
        
        self.driver.get(url)
        time.sleep(SELENIUM_SLEEP)
        
        # 로그인 체크
        page_text = self.driver.execute_script("return document.body.innerText;")
        if "로그인" in page_text[:500] and "비밀번호" in page_text[:500]:
            print(f"⚠️ 로그인")
            return result
        
        # iframe으로 전환
        try:
            iframe = self.driver.find_element("css selector", "iframe#searchIframe")
            self.driver.switch_to.frame(iframe)
            print(f"iframe!", end=" → ")
            time.sleep(2)
        except:
            print(f"iframe X")
            return result
        
        # 첫 번째 장소 클릭
        click_script = """
        var links = document.querySelectorAll('a');
        var clicked = false;
        
        for (var i = 0; i < links.length; i++) {
            var link = links[i];
            var text = link.textContent.trim();
            
            if (text.length > 2 && 
                !text.includes('광고') && 
                !text.includes('지도') &&
                !text.includes('길찾기') &&
                link.href && 
                link.href.includes('place')) {
                link.click();
                clicked = true;
                break;
            }
        }
        return clicked;
        """
        
        clicked = self.driver.execute_script(click_script)
        
        if not clicked:
            print(f"클릭 X")
            self.driver.switch_to.default_content()
            return result
        
        print(f"클릭!", end=" → ")
        time.sleep(4)
        
        # 상세 정보 iframe으로 전환
        try:
            self.driver.switch_to.default_content()
            detail_iframe = self.driver.find_element("css selector", "iframe#entryIframe")
            self.driver.switch_to.frame(detail_iframe)
            print(f"상세!", end=" → ")
            time.sleep(3)
        except:
            print(f"상세 X")
            self.driver.switch_to.default_content()
            return result
        
        # 🔥 리뷰 탭 클릭!
        review_tab_script = """
        var tabs = document.querySelectorAll('a, button, div, span');
        var clicked = false;
        
        for (var i = 0; i < tabs.length; i++) {
            var elem = tabs[i];
            var text = elem.innerText || elem.textContent || '';
            
            if (text.trim() === '리뷰' || text.includes('방문자리뷰') || text.includes('리뷰보기')) {
                elem.click();
                clicked = true;
                break;
            }
        }
        return clicked;
        """
        
        try:
            clicked_review_tab = self.driver.execute_script(review_tab_script)
            if clicked_review_tab:
                print(f"리뷰탭!", end=" → ")
                time.sleep(3)  # 리뷰 로딩 대기
            else:
                print(f"홈탭", end=" → ")
        except:
            print(f"탭X", end=" → ")
        
        # 🔥 모든 정보 추출! (범용 패턴)
        extract_script = """
        var data = {
            review: null,
            rating: null,
            description: null,
            reviews: [],
            photos: []
        };
        
        var allText = document.body.innerText;
        
        // 1. 리뷰 수
        var patterns = [
            /방문자\\s*리뷰\\s*([\\d,]+)/,
            /리뷰\\s*([\\d,]+)/
        ];
        for (var i = 0; i < patterns.length; i++) {
            var match = allText.match(patterns[i]);
            if (match) {
                data.review = parseInt(match[1].replace(/,/g, ''));
                break;
            }
        }
        
        // 2. 별점
        var ratingPatterns = [
            /별점\\s*([\\d\\.]+)/,
            /([\\d\\.]+)\\s*\\/\\s*5/,
            /평점\\s*([\\d\\.]+)/
        ];
        for (var i = 0; i < ratingPatterns.length; i++) {
            var match = allText.match(ratingPatterns[i]);
            if (match) {
                var num = parseFloat(match[1]);
                if (num >= 0 && num <= 5) {
                    data.rating = num;
                    break;
                }
            }
        }
        
        // 3. 업체 소개글 (범용: span, p, div 모두 탐색)
        var allElems = document.querySelectorAll('span, p, div');
        for (var i = 0; i < allElems.length; i++) {
            var elem = allElems[i];
            var text = elem.innerText ? elem.innerText.trim() : '';
            
            // 조건: 30자 이상, 500자 이하, 자식 요소 적음
            if (text.length >= 30 && text.length <= 500 && 
                elem.children.length < 3 &&
                !text.includes('리뷰') &&
                !text.includes('별점') &&
                !text.includes('영업시간') &&
                !text.includes('전화')) {
                data.description = text;
                break;
            }
        }
        
        // 4. 리뷰 텍스트 (범용: 긴 텍스트 찾기)
        var textBlocks = [];
        var allElems2 = document.querySelectorAll('span, p, div, li');
        
        for (var i = 0; i < allElems2.length; i++) {
            var elem = allElems2[i];
            var text = elem.innerText ? elem.innerText.trim() : '';
            
            // 리뷰 같은 텍스트: 15자 이상 (더 공격적으로)
            if (text.length >= 15 && text.length <= 500 &&
                elem.children.length <= 2 &&
                !text.includes('별점') &&
                !text.includes('영업') &&
                !text.includes('전화') &&
                !text.includes('주소') &&
                !text.includes('메뉴') &&
                !text.includes('가격') &&
                !text.match(/^\d+$/) &&  // 순수 숫자 제외
                !text.match(/^[0-9,]+원$/)) {  // 가격 제외
                
                // 중복 체크
                var isDuplicate = false;
                for (var j = 0; j < textBlocks.length; j++) {
                    if (textBlocks[j] === text) {
                        isDuplicate = true;
                        break;
                    }
                }
                
                if (!isDuplicate) {
                    textBlocks.push(text);
                    if (textBlocks.length >= 5) break;
                }
            }
        }
        
        data.reviews = textBlocks;
        
        // 5. 업체 등록 사진 URL (모든 img 태그)
        var allImages = document.querySelectorAll('img');
        var seenUrls = {};
        
        for (var i = 0; i < allImages.length; i++) {
            var img = allImages[i];
            var src = img.src || img.getAttribute('data-src') || img.getAttribute('data-lazy-src');
            
            if (src && src.startsWith('http') && !seenUrls[src]) {
                // 로고나 아이콘 제외 (보통 작음)
                if (!src.includes('icon') && !src.includes('logo')) {
                    seenUrls[src] = true;
                    data.photos.push(src);
                    if (data.photos.length >= 10) break;
                }
            }
        }
        
        return data;
        """
        
        data = self.driver.execute_script(extract_script)
        
        if data:
            result["review_count"] = data.get('review')
            result["rating"] = data.get('rating')
            result["business_description"] = data.get('description')
            result["sample_reviews"] = data.get('reviews', [])
            result["business_photos"] = data.get('photos', [])
            
            # 별점이 없으면 추정
            if result["review_count"] and not result["rating"]:
                result["estimated_rating"] = self._estimate_rating(result["review_count"])
            
            if result["review_count"] or result["rating"]:
                result["success"] = True
                
                rating_display = result["rating"] if result["rating"] else f"~{result['estimated_rating']}"
                print(f"✅ {result['review_count']}개, {rating_display}★")
            else:
                print(f"⏭️ 데이터 없음")
        else:
            print(f"⏭️ 추출 실패")
        
        # iframe에서 나오기
        self.driver.switch_to.default_content()
        
        return result
    
    def _estimate_rating(self, review_count: int) -> float:
        """
        리뷰 수로 별점 추정
        """
        if review_count >= 10000:
            return 4.5
        elif review_count >= 5000:
            return 4.3
        elif review_count >= 2000:
            return 4.0
        elif review_count >= 1000:
            return 3.8
        elif review_count >= 500:
            return 3.5
        else:
            return 3.0
    
    def close(self):
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass


# ============================================
# 데이터 변환
# ============================================
def map_industry(category: str) -> str:
    if '카페' in category or '커피' in category:
        return 'cafe'
    elif '베이커리' in category or '빵' in category:
        return 'bakery'
    else:
        return 'restaurant'

def extract_location(address: str) -> str:
    for loc in LOCATIONS:
        if loc in address:
            return loc
    return "기타"

def create_data(place: NaverPlace, scrape_result: Dict) -> Dict:
    industry = map_industry(place.category)
    location = extract_location(place.roadAddress or place.address)
    
    review_count = scrape_result.get("review_count") or 0
    rating = scrape_result.get("rating")
    estimated_rating = scrape_result.get("estimated_rating")
    
    # 실제 별점 또는 추정 별점
    final_rating = rating if rating else estimated_rating
    
    # 신뢰도 레벨
    if review_count >= 5000 and final_rating and final_rating >= 4.5:
        trust_level = "검증된 인기 매장"
    elif review_count >= 2000:
        trust_level = "지역 대표 매장"
    elif review_count >= 1000:
        trust_level = "입소문 난 매장"
    elif review_count >= 500:
        trust_level = "성장 중인 매장"
    else:
        trust_level = "신규 매장"
    
    return {
        "context": {
            "title": place.title,
            "industry": industry,
            "location": location,
            "category": place.category
        },
        "performance_data": {
            "naver_place": {
                "review_count": review_count,
                "rating": rating,
                "estimated_rating": estimated_rating,
                "has_rating": rating is not None,
                "address": place.address,
                "road_address": place.roadAddress,
                "telephone": place.telephone,
                "link": place.link
            }
        },
        "business_info": {
            "description": scrape_result.get("business_description"),
            "has_description": bool(scrape_result.get("business_description"))
        },
        "customer_voice": {
            "sample_reviews": scrape_result.get("sample_reviews", []),
            "review_count": len(scrape_result.get("sample_reviews", []))
        },
        "visual_assets": {
            "business_photos": scrape_result.get("business_photos", []),
            "photo_count": len(scrape_result.get("business_photos", []))
        },
        "market_insights": {
            "trust_level": trust_level,
            "quality_indicator": "high" if final_rating and final_rating >= 4.3 else "medium",
            "popularity_indicator": "viral" if review_count >= 5000 else "popular" if review_count >= 1000 else "growing"
        },
        "metadata": {
            "crawl_date": datetime.now().isoformat(),
            "data_quality": "complete" if rating else "partial",
            "mapx": place.mapx,
            "mapy": place.mapy,
            "data_source": "naver_desktop_complete"
        }
    }


# ============================================
# 메인
# ============================================
def main():
    print("=" * 80)
    print("🚀 네이버 크롤러 v6 - 완전판 (전국 확대)")
    print("=" * 80)
    print()
    
    # Step 1: API
    print("📋 Step 1: 네이버 API로 1000개 수집 (전국)")
    print("-" * 80)
    print(f"  지역: {len(LOCATIONS)}곳")
    print(f"  키워드: 카페 {len(KEYWORDS['카페'])}개, 맛집 {len(KEYWORDS['맛집'])}개, 베이커리 {len(KEYWORDS['베이커리'])}개")
    print()
    
    api = NaverAPICollector(NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)
    
    queries = []
    for loc in LOCATIONS:
        for category, keywords in KEYWORDS.items():
            for kw in keywords:
                queries.append(f"{loc} {kw}")
    
    random.shuffle(queries)
    
    all_places = []
    seen = set()
    
    for idx, query in enumerate(queries, 1):
        if len(all_places) >= TARGET_COUNT:
            break
        
        items = api.search(query, DISPLAY_MAX)
        
        added = 0
        for item in items:
            key = (item.title.lower(), item.mapx, item.mapy)
            if key in seen:
                continue
            
            seen.add(key)
            all_places.append(item)
            added += 1
        
        if idx % 20 == 0:
            print(f"[{idx}] {query} → 총 {len(all_places)}")
    
    print(f"\n✅ {len(all_places)}개 수집")
    print()
    
    # Step 2: 완전 크롤링
    print("📋 Step 2: 완전 크롤링 (소개/리뷰/사진)")
    print("-" * 80)
    
    crawler = CompleteCrawler(headless=False)
    
    results = []
    success = 0
    
    # 전체 500개 수집!
    print(f"총 {len(all_places)}개 수집 예정...")
    print()
    
    for idx, place in enumerate(all_places, 1):
        print(f"[{idx}/{len(all_places)}] {place.title[:30]} ", end="")
        
        scrape_result = crawler.scrape(place.title, place.roadAddress or place.address, idx)
        
        if scrape_result["success"]:
            success += 1
        
        final = create_data(place, scrape_result)
        results.append(final)
        
        # 중간 저장 (100개마다)
        if idx % 100 == 0:
            print(f"\n  --- 중간 저장 ({idx}개) ---")
            temp_filename = f"{OUT_DIR}/naver_temp_{idx}.json"
            with open(temp_filename, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  💾 {temp_filename}")
            print(f"  ✅ 성공률: {success}/{idx} ({success/idx*100:.1f}%)")
            print()
        
        # 중간 휴식 (50개마다)
        if idx % 50 == 0:
            print(f"\n  --- 잠시 휴식 (5초) ---")
            time.sleep(5)
            print()
    
    crawler.close()
    
    print(f"\n{'='*80}")
    print(f"✅ 전체 수집 완료: {success}/{len(results)} ({success/len(results)*100:.1f}%)")
    print(f"{'='*80}")
    print()
    
    # 통계 출력
    has_rating = sum(1 for r in results if r["performance_data"]["naver_place"]["has_rating"])
    has_description = sum(1 for r in results if r["business_info"]["has_description"])
    has_reviews = sum(1 for r in results if r["customer_voice"]["review_count"] > 0)
    has_photos = sum(1 for r in results if r["visual_assets"]["photo_count"] > 0)
    
    total = len(results)
    
    print("📊 최종 데이터 수집 통계:")
    print(f"  - 전체 수집: {total}개")
    print(f"  - 별점 있음: {has_rating}개 ({has_rating/total*100:.1f}%)")
    print(f"  - 별점 추정: {total-has_rating}개 ({(total-has_rating)/total*100:.1f}%)")
    print(f"  - 업체 소개: {has_description}개 ({has_description/total*100:.1f}%)")
    print(f"  - 리뷰 텍스트: {has_reviews}개 ({has_reviews/total*100:.1f}%)")
    print(f"  - 사진 URL: {has_photos}개 ({has_photos/total*100:.1f}%)")
    print()
    
    # 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{OUT_DIR}/naver_complete_{timestamp}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 최종 저장: {filename}")
    print()
    
    # 최종 평가
    success_rate = success / len(results) * 100
    
    if success_rate >= 85:
        print("🎉🎉🎉 대성공! 85% 이상 수집 완료!")
    elif success_rate >= 70:
        print("👍👍 성공! 70% 이상 수집 완료!")
    elif success_rate >= 50:
        print("👍 괜찮음! 50% 이상 수집 완료!")
    else:
        print("😔 추가 개선 필요...")
    
    print()
    print(f"다음 단계: RAG 시스템에 임베딩하고 GPT 연결!")


if __name__ == "__main__":
    main()
