import re
import sys

class HighColorRenderer:
    # 💡 256색 전용 터미널 ANSI 컬러 맵 캐시
    ANSI_COLOR_MAP = {}
    COLOR_RESET = "\033[0m"

    @classmethod
    def initialize_ansi_palette(cls):
        """데이터 생성기 및 토크나이저와 1:1 동기화되는 256색 ANSI 이스케이프 맵 빌드"""
        if cls.ANSI_COLOR_MAP:
            return
            
        idx = 0
        for r_bit in range(8):      # 3 bits
            for g_bit in range(8):  # 3 bits
                for b_bit in range(4): # 2 bits
                    # 💡 xterm-256 터미널 표준 색상 인덱스로 1:1 다이렉트 맵핑
                    r = int(r_bit * 5 / 7)
                    g = int(g_bit * 5 / 7)
                    b = int(b_bit * 3 / 3)
                    ansi_id = 16 + (r * 36) + (g * 6) + b
                    
                    # 내부 연산 편의를 위해 '000' ~ '255' 순수 숫자 키로 매핑 캐싱
                    cls.ANSI_COLOR_MAP[f"{idx:03d}"] = f"\033[38;5;{ansi_id}m██"
                    idx += 1
                    
        # 극단값 가독성 보정
        cls.ANSI_COLOR_MAP['000'] = "\033[38;5;232m██" # Pure Black
        cls.ANSI_COLOR_MAP['255'] = "\033[38;5;255m██" # Pure White

    @classmethod
    def render(cls, protocol_str):
        cls.initialize_ansi_palette()

        # ------------------------------------------------------------
        # 0. 💥 [철벽 가드] 파싱 실패 또는 미등록어 유입 시 출력할 16x16 '?' 마스크
        # ------------------------------------------------------------
        fallback_mask = ['146'] * 256
        # 물음표 기하학 본체 ('242' 핫핑크색 마킹)
        q_indices = [
            21,22,23,24,25,26, 36,37,42,43, 52,53,58,59, 73,74, 88,89, 103,104, 
            118,119, 134,135, 166,167, 182,183
        ]
        for idx in q_indices:
            fallback_mask[idx] = '242'

        clean_str = re.sub(r'\s+', ' ', protocol_str).strip()
        
        # 특수 오류 토큰 가드
        if "<UNK>" in clean_str or not clean_str:
            final_render_tokens = fallback_mask
        else:
            tokens = clean_str.split()
            pure_pixel_tokens = []
            
            for token in tokens:
                if token == "1X":
                    # 💡 행 스케일 맞춤 보정 가드 (16배수가 안 맞으면 000 채움)
                    rem = len(pure_pixel_tokens) % 16
                    if rem > 0:
                        pure_pixel_tokens.extend(['000'] * (16 - rem))
                    continue
                
                if token.startswith("p"):
                    parts = token.split("_")
                    if len(parts) < 2:
                        continue
                    
                    try:
                        count = int(parts[0][1:])
                    except ValueError:
                        continue
                        
                    colors = parts[1:]
                    
                    if len(colors) == 1:
                        # Case 1. 단일 연속 색상 복원 (p4_000)
                        pure_pixel_tokens.extend([colors[0]] * count)
                    elif len(colors) == 2:
                        # Case 2. 2색 체커보드 패턴 복원 (p2_000_255)
                        pattern = colors * count
                        pure_pixel_tokens.extend(pattern)
            
            # 최종 스케일 가드 (16x16 = 256)
            if len(pure_pixel_tokens) > 0:
                if len(pure_pixel_tokens) < 256:
                    pure_pixel_tokens.extend(['000'] * (256 - len(pure_pixel_tokens)))
                final_render_tokens = pure_pixel_tokens[:256]
            else:
                final_render_tokens = fallback_mask

        # ------------------------------------------------------------
        # 1. 고해상도 터미널 256색 렌더링 출력 (가로 16칸 개행)
        # ------------------------------------------------------------
        print("\n┌" + "─" * 32 + "┐") 
        for i, color_id in enumerate(final_render_tokens):
            ansi_block = cls.ANSI_COLOR_MAP.get(color_id, "\033[37m██")
            
            # Pure White('255')를 제외한 모든 블록은 터미널 가독성을 위해 출력 후 리셋
            if color_id == '255':
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
    # 💡 [검증] 신형 단일 문자 패턴 압축 프로토콜 샘플 스트링
    # 녹색 테두리(028)와 골드 코어(248), 화 Core(255)가 섞인 16x16 규격 검증 데이터
    new_protocol_sample = (
        "p4_000 p8_028 p4_000 1X "
        "p2_000 p2_028 p8_248 p2_028 p2_000 1X "
        "p2_000 p2_028 p8_248 p2_028 p2_000 1X "
        "p1_000 p2_028 p10_248 p2_028 p1_000 1X "
        "p4_028 p8_255 p4_028 1X "
        "p4_028 p8_255 p4_028 1X "
        "p4_028 p8_255 p4_028 1X "
        "p4_028 p8_255 p4_028 1X "
        "p4_028 p8_255 p4_028 1X "
        "p4_028 p8_255 p4_028 1X "
        "p4_028 p8_255 p4_028 1X "
        "p4_028 p8_255 p4_028 1X "
        "p1_000 p2_028 p10_248 p2_028 p1_000 1X "
        "p2_000 p2_028 p8_248 p2_028 p2_000 1X "
        "p2_000 p2_028 p8_248 p2_028 p2_000 1X "
        "p4_000 p8_028 p4_000"
    )
    
    print("[*] 시스템 리팩토링 완료: 신형 p{count}_{colors} 규격 하이컬러 도트 그래픽 렌더러 가동")
    HighColorRenderer.render(new_protocol_sample)