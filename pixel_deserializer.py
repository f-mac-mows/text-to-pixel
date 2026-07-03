import os
from PIL import Image

class PixelDeserializer:
    COLOR_PALETTE = {
        'R': (255, 0, 0),       # red
        'B': (0, 0, 255),       # blue
        'G': (0, 255, 0),       # green
        'Y': (255, 255, 0),     # yellow
        'K': (0, 0, 0),         # black
        'P': (128, 0, 128),     # purple
        'O': (255, 165, 0),     # orange
        'H': (255, 192, 203),    # pink
        'W': (255, 255, 255),   # white
        'A': (128, 128, 128),   # gray
        'N': (165, 42, 42),     # brown
        'S': (135, 206, 235),   # sky blue
        'L': (50, 205, 50),     # lime
        'M': (170, 240, 209),   # mint
        'D': (255, 215, 0),     # gold
        'V': (192, 192, 192),   # silver
        'U': (0, 0, 128),       # navy
        'E': (128, 128, 0),     # olive
        'Z': (245, 245, 220),   # beige
        'J': (255, 0, 255)      # magenta
    }

    @classmethod
    def _get_nearest_color_char(cls, r, g, b):
        """[보정 최적화 버전] RGB 값의 특징을 추출하여 안티앨리어싱 오판정을 방어합니다."""
        
        # 1. 💡 [철벽 방어] 어쨌든 붉은색 성분(R)이 녹색(G)이나 청색(B)보다 현저히 높다면 'Red(R)'로 강제 정렬
        if r > 100 and r > g * 1.5 and r > b * 1.5:
            return 'R'
            
        # 2. 💡 [배경 방어] 세 성분이 모두 200 근처로 밝다면, 노이즈가 꼈더라도 회색이 아닌 'White(W)'로 정렬
        if r > 210 and g > 210 and b > 210:
            return 'W'
            
        # 3. 예외 조건에 걸리지 않는 일반적인 색상은 기존 유클리드 거리 알고리즘 적용
        min_distance = float('inf')
        closest_char = 'W'
        
        for char, palette_rgb in cls.COLOR_PALETTE.items():
            distance = (r - palette_rgb[0])**2 + (g - palette_rgb[1])**2 + (b - palette_rgb[2])**2
            if distance < min_distance:
                min_distance = distance
                closest_char = char
                
        return closest_char

    @classmethod
    def image_to_matrix(cls, image_path, target_size=16):
        img = Image.open(image_path).convert('RGB')
        if img.size != (target_size, target_size):
            img = img.resize((target_size, target_size), Image.Resampling.NEAREST)
            
        matrix = []
        for r in range(target_size):
            row = []
            for c in range(target_size):
                rgb = img.getpixel((c, r))
                row.append(cls._get_nearest_color_char(*rgb))
            matrix.append(row)
        return matrix

    @classmethod
    def _check_alternating_pattern(cls, row_strings):
        """
        [고도화 알고리즘] 체커보드, 격자, 줄무늬 등 교차 패턴 시그니처 감지 및 강제 치환
        기본 규칙: 위/아래 2줄 여백(Background), 내부 12줄이 A-B-A-B 형태로 교차 전개되는지 스캔
        """
        if len(row_strings) != 16:
            return None
            
        bg_row = row_strings[0]
        pattern_a = row_strings[2]
        pattern_b = row_strings[3]
        
        # 1. 상하단 2줄씩 여백(바탕색) 검증
        if not (row_strings[1] == bg_row and row_strings[14] == row_strings[15] == bg_row):
            return None
            
        # 2. 내부 12줄(2번~13번 행)이 정밀하게 A-B 교차 구조를 이루고 있는지 스캔
        is_alternating = True
        for idx in range(2, 14):
            expected = pattern_a if idx % 2 == 0 else pattern_b
            if row_strings[idx] != expected:
                is_alternating = False
                break
                
        if is_alternating:
            # 완벽히 일치하면 하드코딩된 복합 압축 포맷 문법으로 즉시 고밀도 변환하여 반환
            return f"[ Re2 ] [ {bg_row} 1X ] [ Re6 ] [ {pattern_a} 1X {pattern_b} 1X ] [ Re2 ] [ {bg_row} 1X ]"
            
        return None

    @classmethod
    def serialize_matrix(cls, matrix):
        row_strings = []
        
        # 1단계: 행별 RLE 압축
        for row in matrix:
            rle_tokens = []
            current_char = row[0]
            current_len = 1
            for char in row[1:]:
                if char == current_char:
                    current_len += 1
                else:
                    rle_tokens.append(f"{current_len}{current_char}")
                    current_char = char
                    current_len = 1
            rle_tokens.append(f"{current_len}{current_char}")
            row_strings.append(" ".join(rle_tokens))
            
        # 🔥 [고도화 결합] 단순 순차 압축을 하기 전, 교차 레이아웃 시그니처가 잡히는지 먼저 검사합니다.
        advanced_protocol = cls._check_alternating_pattern(row_strings)
        if advanced_protocol:
            return advanced_protocol

        # 2단계: 가로 행 간 일반 연속 패턴([Re]) 압축
        compressed_chunks = []
        i = 0
        num_rows = len(row_strings)
        
        while i < num_rows:
            current_row_str = row_strings[i]
            match_count = 1
            while i + match_count < num_rows and row_strings[i + match_count] == current_row_str:
                match_count += 1
                
            if match_count > 1:
                compressed_chunks.append(f"[ Re{match_count} ] [ {current_row_str} 1X ]")
                i += match_count
            else:
                compressed_chunks.append(f"{current_row_str} 1X")
                i += 1
                
        # 3단계: 공백 및 문자열 슬라이싱 안전 보정 공정 (버그 픽스)
        result = " ".join(compressed_chunks).strip()
        
        # 안전하게 토큰화하여 끝단의 '1X' 혹은 '1X ]' 문제를 완전 무결하게 정제합니다.
        tokens = result.split()
        if tokens[-1] == "]" and tokens[-2] == "1X":
            del tokens[-2]  # 끝이 '1X ]'로 끝나면 '1X'만 제거
        elif tokens[-1] == "1X":
            del tokens[-1]  # 끝이 그냥 '1X'로 끝나면 무조건 제거
            
        return " ".join(tokens)

    @classmethod
    def deserialize_image(cls, image_path):
        matrix = cls.image_to_matrix(image_path)
        return cls.serialize_matrix(matrix)