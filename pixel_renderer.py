import re
import sys

class HighColorRenderer:
    # 💡 256색 전용 터미널 ANSI 컬러 맵 캐시
    ANSI_COLOR_MAP = {}
    COLOR_RESET = "\033[0m"

    @classmethod
    def initialize_ansi_palette(cls):
        """데이터 생성기와 1:1 동기화되는 256색 ANSI 이스케이프 맵 빌드"""
        if cls.ANSI_COLOR_MAP:
            return
            
        idx = 0
        for r_bit in range(8):      # 3 bits
            for g_bit in range(8):  # 3 bits
                for b_bit in range(4): # 2 bits
                    # 💡 xterm-256 터미널 표준 색상 인덱스로 1:1 다이렉트 맵핑 유도
                    # 터미널 표준 256색 공식을 사용하여 RGB 공간을 화면에 복원합니다.
                    r = int(r_bit * 5 / 7)
                    g = int(g_bit * 5 / 7)
                    b = int(b_bit * 3 / 3)
                    ansi_id = 16 + (r * 36) + (g * 6) + b
                    
                    # 데이터 생성기 규격과 동일한 'c000' ~ 'c255' 키 생성
                    cls.ANSI_COLOR_MAP[f"c{idx:03d}"] = f"\033[38;5;{ansi_id}m██"
                    idx += 1
                    
        # 극단값 보정 (완전 흑색/백색 가독성 확보)
        cls.ANSI_COLOR_MAP['c000'] = "\033[38;5;232m██" # Pure Black
        cls.ANSI_COLOR_MAP['c255'] = "\033[38;5;255m██" # Pure White

    @classmethod
    def render(cls, protocol_str):
        cls.initialize_ansi_palette()

        # ------------------------------------------------------------
        # 0. 💥 [철벽 가드] 데이터 무결성 실패 시 출력할 16x16 '?' 마스크 데이터
        #    (배경은 어두운 회색 'c146', 물음표 본체는 핫핑크 'c242')
        # ------------------------------------------------------------
        fallback_question_mark = [
            "1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146",
            "1c146","1c146","1c146","1c146","1c146","1c242","1c242","1c242","1c242","1c242","1c242","1c146","1c146","1c146","1c146","1c146",
            "1c146","1c146","1c146","1c146","1c242","1c242","1c146","1c146","1c146","1c146","1c242","1c242","1c146","1c146","1c146","1c146",
            "1c146","1c146","1c146","1c146","1c242","1c242","1c146","1c146","1c146","1c146","1c242","1c242","1c146","1c146","1c146","1c146",
            "1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c242","1c242","1c146","1c146","1c146","1c146","1c146",
            "1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c242","1c242","1c146","1c146","1c146","1c146","1c146","1c146",
            "1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c242","1c242","1c146","1c146","1c146","1c146","1c146","1c146","1c146",
            "1c146","1c146","1c146","1c146","1c146","1c146","1c242","1c242","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146",
            "1c146","1c146","1c146","1c146","1c146","1c146","1c242","1c242","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146",
            "1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146",
            "1c146","1c146","1c146","1c146","1c146","1c146","1c242","1c242","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146",
            "1c146","1c146","1c146","1c146","1c146","1c146","1c242","1c242","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146",
            "1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146",
            "1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146",
            "1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146",
            "1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146","1c146"
        ]

        clean_str = protocol_str.strip()
        
        if "<UNK>" in clean_str or not clean_str:
            pure_pixel_tokens = fallback_question_mark
        else:
            # 1) 압축부 포맷팅 규칙 일치화 ([ Re2 ] -> [Re2])
            clean_str = re.sub(r'\[\s*Re(\d+)\s*\]', r'[Re\1]', clean_str)
            
            # 2) 💡 신규 고정폭 규칙 매칭용 정규식 패치: 숫자 뒤에 나오는 'c'와 숫자 3자리를 그룹화
            # 예: "12c015" 패턴의 양옆에 공백 주입
            clean_str = re.sub(r'(\d+c\d{3})', r' \1 ', clean_str)
            
            # 3) 대괄호 반복 패턴 정규식 확장 (\s* 유연성 확장 유지)
            repeat_pattern = re.compile(r'\[Re(\d+)\]\s*\[\s*(.*?)\s*\]')
            
            def expand_match(match):
                count = int(match.group(1))
                content = match.group(2).strip()
                if count <= 0: return ""
                return " ".join([content] * count)
            
            expanded_str = repeat_pattern.sub(expand_match, clean_str)
            expanded_str = expanded_str.replace('[', ' ').replace(']', ' ')
            raw_tokens = expanded_str.split()
            
            # 4) 1칸 단위 분해 배열 복원 공정
            pure_pixel_tokens = []
            
            for token in raw_tokens:
                if not token or token.startswith('Re'): 
                    continue
                if token == '1X':
                    continue
                    
                try:
                    # 💡 신규 토큰 구조 슬라이싱 분리 (끝의 4자리가 'cXXX' 형태임)
                    color_token = token[-4:] # 'c025' 형태 추출
                    length = int(token[:-4]) # 앞쪽의 반복 길이 추출
                    
                    if color_token not in cls.ANSI_COLOR_MAP:
                        raise ValueError
                        
                    pure_pixel_tokens.extend([f"1{color_token}"] * length)
                except ValueError:
                    pure_pixel_tokens = []
                    break

        # 5) 수량 검증 가드 (16x16 = 256칸 강제화)
        if len(pure_pixel_tokens) == 256:
            final_render_tokens = pure_pixel_tokens
        else:
            final_render_tokens = fallback_question_mark

        # ------------------------------------------------------------
        # 6. 고해상도 256색 최종 시각화 출력 (가로 16칸 단위 개행)
        # ------------------------------------------------------------
        print("\n┌" + "─" * 32 + "┐") 
        for i, token in enumerate(final_render_tokens):
            color_token = token[1:] # 'cXXX' 추출
            ansi_block = cls.ANSI_COLOR_MAP.get(color_token, "\033[37m██")
            
            # Pure White('c255')를 제외한 모든 블록은 출력 후 색상을 즉시 초기화
            if color_token == 'c255':
                sys.stdout.write(ansi_block)
            else:
                sys.stdout.write(ansi_block + cls.COLOR_RESET)
                
            if (i + 1) % 16 == 0:
                sys.stdout.write('\n')
        print("└" + "─" * 32 + "┘\n")
        sys.stdout.flush()

# ============================================================
# 테스트 실행부
# ============================================================
if __name__ == "__main__":
    # ============================================================
    # 💡 [정밀 보정] 3:3:2 실제 매핑 인덱스로 수정한 무결한 프로토콜 문자열
    # c000 (Black 배경), c028 (True Green 테두리), c248 (True Gold 본체), c255 (White 코어)
    # ============================================================
    new_256_protocol = "4c000 8c028 4c000 1X [ Re2 ] [ 2c000 2c028 8c248 2c028 2c000 1X ] 1c000 2c028 10c248 2c028 1c000 1X [ Re8 ] [ 4c028 8c255 4c028 1X ] 1c000 2c028 10c248 2c028 1c000 1X [ Re2 ] [ 2c000 2c028 8c248 2c028 2c000 1X ] 4c000 8c028 4c000"
    
    print("[*] 시스템 리팩토링 완료: 256색 고정폭 하이컬러 도트 그래픽 렌더링 테스트")
    HighColorRenderer.render(new_256_protocol)