import re
import sys

def render_protocol_with_brackets(protocol_str):
    """
    [근본적 구조 개혁 버전]
    - 20색 고유 알파벳 완벽 매핑 (L = 순수 라임색)
    - 줄바꿈 토큰을 중복 없는 고유 기호 '1X'로 전면 수정
    - <UNK> 및 규격 미달(256칸 오류) 발생 시 16x16 '?' 마스크 강제 출력
    """
    
    # ------------------------------------------------------------
    # 0. 💥 [철벽 가드] 데이터 무결성 실패 시 출력할 16x16 '?' 매트릭스
    #    (A = Gray 배경, H = Pink 물음표 본체)
    # ------------------------------------------------------------
    fallback_question_mark = [
        "1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A",
        "1A","1A","1A","1A","1A","1H","1H","1H","1H","1H","1H","1A","1A","1A","1A","1A",
        "1A","1A","1A","1A","1H","1H","1A","1A","1A","1A","1H","1H","1A","1A","1A","1A",
        "1A","1A","1A","1A","1H","1H","1A","1A","1A","1A","1H","1H","1A","1A","1A","1A",
        "1A","1A","1A","1A","1A","1A","1A","1A","1A","1H","1H","1A","1A","1A","1A","1A",
        "1A","1A","1A","1A","1A","1A","1A","1A","1H","1H","1A","1A","1A","1A","1A","1A",
        "1A","1A","1A","1A","1A","1A","1A","1H","1H","1A","1A","1A","1A","1A","1A","1A",
        "1A","1A","1A","1A","1A","1A","1H","1H","1A","1A","1A","1A","1A","1A","1A","1A",
        "1A","1A","1A","1A","1A","1A","1H","1H","1A","1A","1A","1A","1A","1A","1A","1A",
        "1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A",
        "1A","1A","1A","1A","1A","1A","1H","1H","1A","1A","1A","1A","1A","1A","1A","1A",
        "1A","1A","1A","1A","1A","1A","1H","1H","1A","1A","1A","1A","1A","1A","1A","1A",
        "1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A",
        "1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A",
        "1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A",
        "1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A","1A"
    ]

    # 1. 깔끔하게 정리된 20색 ANSI 터미널 컬러 맵
    COLOR_RESET = "\033[0m"
    COLOR_MAP = {
        'R': "\033[38;5;196m██", 'B': "\033[38;5;21m██",  'G': "\033[38;5;46m██",  'Y': "\033[38;5;226m██",
        'K': "\033[38;5;232m██", 'P': "\033[38;5;129m██", 'O': "\033[38;5;208m██", 'H': "\033[38;5;205m██",
        'W': "\033[38;5;255m██", 'A': "\033[38;5;242m██", 'N': "\033[38;5;94m██",  'S': "\033[38;5;117m██",
        'L': "\033[38;5;118m██", # L은 이제 100% 퓨어 라임색(Lime)입니다!
        'M': "\033[38;5;48m██",  'D': "\033[38;5;220m██", 'V': "\033[38;5;250m██", 'U': "\033[38;5;18m██",
        'E': "\033[38;5;58m██",  'Z': "\033[38;5;230m██", 'J': "\033[38;5;201m██"
    }

    # ------------------------------------------------------------
    # 2. 데이터 전처리 및 정규식 확장 단계
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # 2. 데이터 전처리 및 정규식 확장 단계 (공백 유연성 패치 완료)
    # ------------------------------------------------------------
    # 💡 괄호 앞뒤에 공백이 있든 없든 정규식 연산을 위해 갉아먹는 전처리 제거 및 정규식 수정
    clean_str = protocol_str.strip()
    
    # 미지 토큰 예외 필터링
    if "<UNK>" in clean_str or not clean_str.strip():
        pure_pixel_tokens = fallback_question_mark
    else:
        # 1) 숫+알파벳 토큰 분리 전, 괄호 규칙 내부의 불필요한 공백을 정형화
        # [ Re2 ] -> [Re2] / [ 4W -> [4W / 1X ] -> 1X] 로 정렬
        clean_str = re.sub(r'\[\s*Re(\d+)\s*\]', r'[Re\1]', clean_str)
        
        # 2) 토큰 분할을 위해 픽셀 데이터 양옆에 공백 주입
        clean_str = re.sub(r'(\d+[A-Z])', r' \1 ', clean_str)
        
        # 3) 💡 [핵심 교정] [Re2] [ 4W ... 1X ] 구조에서 내부 괄호 공백 유연하게 매칭하도록 \s* 주입
        repeat_pattern = re.compile(r'\[Re(\d+)\]\s*\[\s*(.*?)\s*\]')
        
        def expand_match(match):
            count = int(match.group(1))
            content = match.group(2).strip()
            if count <= 0: return ""
            return " ".join([content] * count)
        
        # 반복 패턴 확장 적용
        expanded_str = repeat_pattern.sub(expand_match, clean_str)
        expanded_str = expanded_str.replace('[', ' ').replace(']', ' ')
        raw_tokens = expanded_str.split()
        
        # ------------------------------------------------------------
        # 3. 1칸 단위 분해 배열 생성
        # ------------------------------------------------------------
        pure_pixel_tokens = []
        
        for token in raw_tokens:
            if not token or token.startswith('Re'): 
                continue
            
            if token == '1X':
                continue
                
            try:
                color = token[-1]
                length = int(token[:-1])
                
                if color not in COLOR_MAP:
                    raise ValueError
                    
                pure_pixel_tokens.extend([f"1{color}"] * length)
            except ValueError:
                pure_pixel_tokens = []
                break

    # ------------------------------------------------------------
    # 4. 💥 [최종 수량 검증 가드] 정확히 256개가 아니면 '?' 모양 마스크로 교체
    # ------------------------------------------------------------
    total_pixels_needed = 16 * 16  # 256
    
    if len(pure_pixel_tokens) == total_pixels_needed:
        final_render_tokens = pure_pixel_tokens
    else:
        final_render_tokens = fallback_question_mark

    # ------------------------------------------------------------
    # 5. 최종 시각화 출력 (정확히 256칸 출력)
    # ------------------------------------------------------------
    print("\n┌" + "─" * 32 + "┐") 
    
    for i, token in enumerate(final_render_tokens):
        color = token[-1]
        char = COLOR_MAP.get(color, "\033[37m██")
        
        if color == 'W':
            sys.stdout.write(char)
        else:
            sys.stdout.write(char + COLOR_RESET)
            
        # 가로 16칸 단위 자동 개행
        if (i + 1) % 16 == 0:
            sys.stdout.write('\n')
                    
    print("└" + "─" * 32 + "┘\n")
    sys.stdout.flush()

# ============================================================
# 테스트 실행부
# ============================================================
if __name__ == "__main__":
    # 신규 데이터셋 규격 테스트: 
    # '1X'가 줄바꿈 역할을 하고, 'L'은 온전히 라임색 픽셀(예: 14L, 2L 등)로 작동합니다.
    new_perfect_protocol = "5W 1S 10W 1X [ Re2 ] [ 4W 2S 10W 1X ] 3W 1S 1W 1S 2W 6N 2W 1X 2W 1S 2W 1S 7W 1N 2W 1X [ Re2 ] [ 5W 1S 4W 1A 2W 1N 2W 1X ] 4W 1S 5W 1A 1W 2N 2W 1X 4W 4S 2W 1A 1W 1N 3W 1X 4W 1S 4W 2A 1W 1N 3W 1X 3W 6A 3W 1N 3W 1X 3W 1A 7W 1N 4W 1X 3W 1A 5W 3N 4W 1X 3W 1A 2W 3N 2W 2N 3W 1X 7W 5N 4W 1X 16W"
    
    print("[*] 개편된 데이터셋 구조(줄바꿈=1X, L=라임색) 기반 철벽 렌더링 시작")
    render_protocol_with_brackets(new_perfect_protocol)