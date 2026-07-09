import json
import random
import os
import re
from pixel_config import PixelPaths  # 기존 경로 설정 파일 활용

# 렌더링을 시각적으로 이쁘게 보기 위한 터미널 컬러/이모지 매핑
RENDER_MAP = {
    'R': '🟥', 'B': '🟦', 'G': '🟩', 'Y': '🟨', 'K': '⬛',
    'P': '🟪', 'O': '🟧', 'H': '💗', 'W': '⬜', 'A': '🪙',
    'N': '🟫', 'S': '🔷', 'L': '💚', 'M': '🧼', 'D': '👑',
    'V': '🥈', 'U': '🔵', 'E': '🌿', 'Z': '📜', 'J': '🔮'
}

def decode_bracket_string(encoded_str: str) -> list:
    """ [ ReX ] [ ... ] 구조와 RLE 토큰들을 완벽하게 풀어서 16x16 매트릭스로 복원합니다. """
    # 공백 정리
    encoded_str = re.sub(r'\s+', ' ', encoded_str).strip()
    
    # 1. 브래킷 구문 해제 ([ ReX ] [ 내용 1X ]) -> 안전한 문자열 슬라이싱 및 파싱
    # 정규식 탈출 문자(escape)를 명확히 하여 Pylance 에러 차단
    bracket_pattern = re.compile(r'\[\s*Re(\d+)\s*\]\s*\[\s*([^\]]+)\s*\]')
    
    while True:
        match = bracket_pattern.search(encoded_str)
        if not match:
            break
            
        count = int(match.group(1))
        content = match.group(2).strip()
        
        # 끝에 붙은 1X 토큰 제거 안정화
        if content.endswith(" 1X"):
            content = content[:-3].strip()
            
        # 반복 횟수만큼 분할자(|)를 붙여서 전개
        expanded = " | ".join([content] * count)
        
        # 에러 유발 지점 수정: match.end() 뒤의 오타 제거 및 안전하게 결합
        encoded_str = encoded_str[:match.start()] + expanded + encoded_str[match.end():]

    # 2. 행 단위 토큰 분리
    raw_rows = []
    tokens = encoded_str.split()
    
    current_row_tokens = []
    for token in tokens:
        if token == "|":
            if current_row_tokens:
                raw_rows.append(current_row_tokens)
                current_row_tokens = []
        elif token == "1X":
            if current_row_tokens:
                raw_rows.append(current_row_tokens)
                current_row_tokens = []
        else:
            current_row_tokens.append(token)
            
    if current_row_tokens:
        raw_rows.append(current_row_tokens)

    # 3. RLE 디코딩 구동 (예: "3R" -> ['R', 'R', 'R'])
    final_matrix = []
    for r_tokens in raw_rows:
        row_pixels = []
        for token in r_tokens:
            # 3R, 12W 같은 패턴 매칭
            match = re.match(r'(\d+)([A-Z])', token)
            if match:
                count = int(match.group(1))
                char = match.group(2)
                row_pixels.extend([char] * count)
            else:
                if len(token) == 1 and token.isalpha():
                    row_pixels.append(token)
                    
        # 16px 규격보다 모자라면 배경색(W)으로 채움
        if len(row_pixels) < 16:
            row_pixels.extend(['W'] * (16 - len(row_pixels)))
        final_matrix.append(row_pixels[:16])
        
    # 16행 규격 맞춤
    while len(final_matrix) < 16:
        final_matrix.append(['W'] * 16)
        
    return final_matrix[:16]

def render_to_terminal(matrix, prompt):
    """ 복원된 매트릭스를 터미널에 이쁘게 출력합니다. """
    print("\n" + "="*40)
    print(f"💬 Prompt: {prompt}")
    print("="*40)
    for row in matrix:
        line = "".join([RENDER_MAP.get(pixel, '❓') for pixel in row])
        print(line)
    print("="*40)

def test_preview(keyword=None, num_samples=3):
    """ JSONL 파일을 읽어 샘플을 추출하고 시각화합니다. """
    jsonl_path = PixelPaths.TOTAL_DATA
    
    if not os.path.exists(jsonl_path):
        print(f"❌ 데이터셋 파일이 없습니다: {jsonl_path}\n먼저 빌더 스크립트를 구동해 주세요.")
        return

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    entries = [json.loads(line) for line in lines]
    
    # 특정 키워드(예: 'apple', 'potion')가 주어지면 필터링
    if keyword:
        entries = [e for e in entries if keyword.lower() in e["input"].lower()]
        print(f"🔍 키워드 '{keyword}' 검색 결과: 총 {len(entries)}개 발견")
    
    if not entries:
        print("⚠️ 조건에 맞는 데이터가 데이터셋에 존재하지 않습니다.")
        return

    # 지정된 개수만큼 샘플링
    samples = random.sample(entries, min(num_samples, len(entries)))
    
    for sample in samples:
        prompt = sample["input"]
        encoded_output = sample["output"]
        
        # 디코딩 가동
        try:
            matrix = decode_bracket_string(encoded_output)
            render_to_terminal(matrix, prompt)
        except Exception as e:
            print(f"❌ 디코딩 실패 ({prompt}): {e}")
            print(f"   원본 데이터: {encoded_output}")

if __name__ == "__main__":
    # 1. 먼저 전체 데이터셋에서 무작위로 2개 뽑아서 검증
    print("[*] 🎲 무작위 샘플 렌더링 검증 테스트 시작...")
    test_preview(num_samples=2)
    
    # 2. 방금 정제한 커스텀 실제 이미지 에셋 검증 (예: 포션이나 과일)
    print("\n[*] 🧪 커스텀 추출 에셋 렌더링 검증 테스트 시작...")
    test_preview(keyword="potion", num_samples=1)
    test_preview(keyword="apple", num_samples=1)