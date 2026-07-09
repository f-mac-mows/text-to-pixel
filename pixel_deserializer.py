import os
from PIL import Image

class PixelPaletteManager:
    COLOR_PALETTE = {}
    
    @classmethod
    def initialize_palette(cls):
        """RGB 3:3:2 비트 공간을 활용해 도트 그래픽용 256색 팔레트 자동 빌드"""
        if cls.COLOR_PALETTE:
            return
            
        idx = 0
        for r_bit in range(8):      # 3 bits
            for g_bit in range(8):  # 3 bits
                for b_bit in range(4): # 2 bits
                    r = int(r_bit * 255 / 7)
                    g = int(g_bit * 255 / 7)
                    b = int(b_bit * 255 / 3)
                    cls.COLOR_PALETTE[f"c{idx:03d}"] = (r, g, b)
                    idx += 1

    @classmethod
    def get_nearest_color_token(cls, r, g, b):
        """입력된 RGB와 가장 가까운 3자리 고유 컬러 토큰(cXXX)을 고속 매핑"""
        cls.initialize_palette()
        
        if r > 245 and g > 245 and b > 245: return 'c255' # Pure White
        if r < 10 and g < 10 and b < 10: return 'c000'     # Pure Black

        min_distance = float('inf')
        closest_token = 'c255'
        
        for token, palette_rgb in cls.COLOR_PALETTE.items():
            distance = (r - palette_rgb[0])**2 + (g - palette_rgb[1])**2 + (b - palette_rgb[2])**2
            if distance < min_distance:
                min_distance = distance
                closest_token = token
                
        return closest_token


class PixelDeserializer:
    @classmethod
    def image_to_matrix(cls, image_path, target_size=16):
        # 💡 RGBA 모드로 열어서 투명도(Alpha) 채널까지 확보합니다.
        img = Image.open(image_path).convert('RGBA')
        if img.size != (target_size, target_size):
            img = img.resize((target_size, target_size), Image.Resampling.NEAREST)
            
        # 💡 파일명이나 경로에 'black'이 들어가는지 감지합니다.
        is_black_subject = "black" in os.path.basename(image_path).lower()
        
        matrix = []
        for r in range(target_size):
            row = []
            for c in range(target_size):
                pixel = img.getpixel((c, r))
                r_val, g_val, b_val, a_val = pixel[0], pixel[1], pixel[2], pixel[3]
                
                # 1. 만약 완전히 투명한(Alpha=0) 픽셀이라면 배경으로 간주
                if a_val < 10:
                    token = 'c255' if is_black_subject else 'c000'
                else:
                    # 2. 불투명한 픽셀인데 오브젝트가 black 계열이고 검은색에 가까운 경우
                    token = PixelPaletteManager.get_nearest_color_token(r_val, g_val, b_val)
                    if is_black_subject and token == 'c000':
                        # 오브젝트 자체는 검은색을 유지해야 하므로 통과
                        pass
                    # 만약 배경이 투명 처리가 안 되어 있고 이미지 자체가 검은 배경에 검은 오브젝트라면
                    # 명도 기준(예: RGB 총합 < 30)으로 배경과 물체를 분리하는 로직을 추가할 수도 있습니다.
                
                row.append(token)
            matrix.append(row)
        return matrix

    @classmethod
    def _check_alternating_pattern(cls, row_strings):
        """체커보드, 격자 무늬 등의 레이아웃 특화 고밀도 압축 알고리즘"""
        if len(row_strings) != 16:
            return None
            
        bg_row = row_strings[0]
        pattern_a = row_strings[2]
        pattern_b = row_strings[3]
        
        if not (row_strings[1] == bg_row and row_strings[14] == row_strings[15] == bg_row):
            return None
            
        is_alternating = True
        for idx in range(2, 14):
            expected = pattern_a if idx % 2 == 0 else pattern_b
            if row_strings[idx] != expected:
                is_alternating = False
                break
                
        if is_alternating:
            compressed = f"[ Re2 ] [ {bg_row} 1X ] [ Re6 ] [ {pattern_a} 1X {pattern_b} 1X ] [ Re2 ] [ {bg_row} 1X ]"
            if compressed.endswith("1X ]"): 
                compressed = compressed[:-4] + " ]"
            return compressed
            
        return None

    @classmethod
    def serialize_matrix(cls, matrix):
        row_strings = []
        
        # 1단계: 행별 RLE 압축 (cXXX 토큰 단위)
        for row in matrix:
            rle_tokens = []
            current_token = row[0]
            current_len = 1
            for token in row[1:]:
                if token == current_token:
                    current_len += 1
                else:
                    rle_tokens.append(f"{current_len}{current_token}")
                    current_token = token
                    current_len = 1
            rle_tokens.append(f"{current_len}{current_token}")
            row_strings.append(" ".join(rle_tokens))
            
        # 2단계: 특수 기하학 패턴(체커보드 등) 시그니처 선행 검사
        advanced_protocol = cls._check_alternating_pattern(row_strings)
        if advanced_protocol:
            return advanced_protocol

        # 3단계: 가로 행 간 연속 행([Re]) 압축 진행
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
                
        # 4단계: 끝단 1X 보정 및 토큰 정제 공정
        result = " ".join(compressed_chunks).strip()
        tokens = result.split()
        if not tokens: return ""
        
        if tokens[-1] == "]" and len(tokens) >= 2 and tokens[-2] == "1X":
            del tokens[-2]
        elif tokens[-1] == "1X":
            del tokens[-1]
            
        return " ".join(tokens)

    @classmethod
    def deserialize_image(cls, image_path):
        PixelPaletteManager.initialize_palette()
        matrix = cls.image_to_matrix(image_path)
        return cls.serialize_matrix(matrix)