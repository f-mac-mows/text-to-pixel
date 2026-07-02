import re
import sys

def render_protocol_with_brackets(protocol_str):
    """
    [Re숫자][패턴]을 평탄화하고, 모든 멀티 픽셀 토큰(예: 16W -> 1W * 16)을 
    1칸 단위로 완벽히 분해하여 그리드가 절대 깨지지 않는 무결점 철벽 렌더러입니다.
    """
    # ------------------------------------------------------------
    # 1. 전처리 및 정규식 확장 (공백 밀착 및 줄바꿈 토큰 격리)
    # ------------------------------------------------------------
    # 보카 사전 분리로 인해 생긴 대괄호 사이의 공백(] [)을 완전히 밀착시킵니다.
    clean_str = protocol_str.replace("] [", "][")
    
    # 내부 공백 변형을 방지하기 위해 정규식에 공백(\s*) 유연성을 부여합니다.
    repeat_pattern = re.compile(r'\[Re(\d+)\]\s*\[(.*?)\]')
    
    def expand_match(match):
        count = int(match.group(1))
        content = match.group(2)
        return " ".join([content] * count)
    
    expanded_str = repeat_pattern.sub(expand_match, clean_str)
    
    # ✨ [치명적 버그 해결]: 패턴 전개 시 붙어버리는 '1L2W' 같은 현상을 막기 위해 1L 앞뒤로 공백 강제 주입
    expanded_str = expanded_str.replace("1L", " 1L ")
    
    # 남은 특수 대괄호 제거 후 순수 토큰화 준비
    expanded_str = expanded_str.replace('[', ' ').replace(']', ' ')
    raw_tokens = expanded_str.split()
    
    # ------------------------------------------------------------
    # 2. 원시 픽셀 단위(1칸 단위) 완전 분해 배열 생성
    # ------------------------------------------------------------
    pure_pixel_tokens = []
    
    for token in raw_tokens:
        # 무의미한 토큰, 줄바꿈 기호, 가짜 루프 토큰 제거
        if not token or token.endswith('L') or token.startswith('Re'): 
            continue
        try:
            color = token[-1]
            length = int(token[:-1])
            # "16W" -> ["1W"] * 16 형태로 낱개 쪼개기 주입 (그리드 밀림 원천 차단)
            pure_pixel_tokens.extend([f"1{color}"] * length)
        except ValueError:
            # 파싱 불가능한 가짜 토큰 예외 처리
            continue

    # ------------------------------------------------------------
    # 3. ANSI 터미널 컬러 맵 및 테두리 설정
    # ------------------------------------------------------------
    COLOR_RESET = "\033[0m"

    COLOR_MAP = {
        'W': "\033[38;5;255m██",   # 순백색 (Pure White) - 배경이 칙칙해지지 않게 강제 고정
        'R': "\033[38;5;196m██",   # 리얼 레드 (Vibrant Red)
        'Y': "\033[38;5;226m██",   # 선명한 노란색 (Bright Yellow)
        'G': "\033[38;5;46m██",    # 네온 초록색 (Vibrant Green)
        'B': "\033[38;5;21m██",    # 파란색 (True Blue)
        'K': "\033[38;5;232m██",   # 딥 블랙 (Deep Black)
        'P': "\033[38;5;129m██",   # 보라색 (Purple)
        'O': "\033[38;5;208m██",   # 주황색 (Vibrant Orange)
        'H': "\033[38;5;205m██",   # ✨ 핫핑크 (Hot Pink) - 기존 Red 오매핑 수정
        'A': "\033[38;5;244m██",   # 회색 (Medium Gray)
    }

    # 가로 16칸 맞춤형 프레임 테두리 (블록당 2자이므로 32칸 상단바)
    print("\n" + "=" * 32) 
    
    pixel_count = 0  # 가로 픽셀 카운트 (16이 되면 줄바꿈)
    row_count = 0    # 세로 줄 카운트 (16이 되면 강제 종료)
    
    # ------------------------------------------------------------
    # 4. 최종 시각화 출력 루프
    # ------------------------------------------------------------
    for token in pure_pixel_tokens:
        # [철벽 가드]: 이미 16줄을 다 그렸다면 뒤에 남은 잔여 토큰은 과감히 버림
        if row_count == 16:
            break
            
        color = token[-1]
        char = COLOR_MAP.get(color, "\033[37m██")
        
        # 흰색 배경은 리셋 토큰을 생략하여 터미널 가독성 업
        if color == 'W':
            sys.stdout.write(char)
        else:
            sys.stdout.write(char + COLOR_RESET)
            
        pixel_count += 1
        
        # 정확히 16번째 픽셀이 찍힐 때만 칼같이 줄바꿈
        if pixel_count == 16:
            sys.stdout.write('\n')
            pixel_count = 0
            row_count += 1  
                    
    print("=" * 32 + "\n")
    sys.stdout.flush()

# ============================================================
# 사용 예시 (테스트 실행부)
# ============================================================
if __name__ == "__main__":
    # 모델의 실제 추론 출력 샘플 입력 테스트
    sample_protocol = "[Re2] [16W 1L] [Re6] [2W 1Y 1W 1Y 1W 1Y 1W 1Y 1W 1Y 1W 1Y 3W 1L 3W 1Y 1W 1Y 1W 1Y 1W 1Y 1W 1Y 1W 1Y 2W 1L] [Re2] [16W]"
    
    print("[*] 고밀도 체커보드 프로토콜 렌더링을 시작합니다.")
    render_protocol_with_brackets(sample_protocol)