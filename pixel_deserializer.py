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
        """PNG 이미지를 읽어 투명도 보정 후 16x16 cXXX 토큰 2D 리스트(행렬)로 반환"""
        img = Image.open(image_path).convert('RGBA')
        if img.size != (target_size, target_size):
            img = img.resize((target_size, target_size), Image.Resampling.NEAREST)
            
        is_black_subject = "black" in os.path.basename(image_path).lower()
        
        matrix = []
        for r in range(target_size):
            row = []
            for c in range(target_size):
                pixel = img.getpixel((c, r))
                r_val, g_val, b_val, a_val = pixel[0], pixel[1], pixel[2], pixel[3]
                
                # 1. 완전히 투명한(Alpha=0) 픽셀 처리
                if a_val < 10:
                    token = 'c255' if is_black_subject else 'c000'
                else:
                    # 2. 불투명한 픽셀 색상 토큰 고속 매핑
                    token = PixelPaletteManager.get_nearest_color_token(r_val, g_val, b_val)
                
                row.append(token)
            matrix.append(row)
        return matrix

    @classmethod
    def serialize_matrix(cls, matrix):
        """
        [정밀 정렬] 16x16 행렬 데이터를 신형 p{count}_{colors} 및 1X 조합의 
        최종 시퀀스 스트링으로 완벽하게 압축 및 직렬화합니다.
        """
        tokens = []
        num_rows = len(matrix)
        
        for r in range(num_rows):
            # 대화형 추론 시스템 규격에 맞춰 'c' 접두사를 배제한 순수 ID 데이터로 가공
            row = [pixel.replace("c", "") for pixel in matrix[r]]
            c_idx = 0
            row_len = len(row)
            
            while c_idx < row_len:
                # 1단계: 단일 색상 연속 스캔 (예: 000이 연속되는 구간)
                run_length = 1
                while c_idx + run_length < row_len and row[c_idx] == row[c_idx + run_length]:
                    run_length += 1
                
                if run_length >= 2:
                    tokens.append(f"p{run_length}_{row[c_idx]}")
                    c_idx += run_length
                    continue
                
                # 2단계: 2개 색상 교차 패턴(체크무늬/격자) 최소 조건 탐지
                # p2_A_B 규격은 2쌍(총 4글자: A B A B)이 완전히 매칭될 때 작동합니다.
                if c_idx + 3 < row_len and row[c_idx] == row[c_idx + 2] and row[c_idx + 1] == row[c_idx + 3]:
                    color_a = row[c_idx]
                    color_b = row[c_idx + 1]
                    
                    # 확장 가능성 확인 (A B 가 뒤에 더 반복되는지 스캔)
                    pattern_count = 2
                    while c_idx + (pattern_count * 2) + 1 < row_len:
                        next_a = row[c_idx + (pattern_count * 2)]
                        next_b = row[c_idx + (pattern_count * 2) + 1]
                        if next_a == color_a and next_b == color_b:
                            pattern_count += 1
                        else:
                            break
                            
                    tokens.append(f"p{pattern_count}_{color_a}_{color_b}")
                    c_idx += (pattern_count * 2)
                    continue
                
                # 3단계: 단독 픽셀 처리 (p1_{color})
                tokens.append(f"p1_{row[c_idx]}")
                c_idx += 1
                
            # 행 마지막에 개행 지시자 '1X' 주입 (단, 가장 마지막 행 뒤에는 붙이지 않음)
            if r < num_rows - 1:
                tokens.append("1X")
                
        return " ".join(tokens)

    @classmethod
    def deserialize_image(cls, image_path):
        """에셋 파일 스캔 메인 파이프라인 진입점: PNG 이미지 경로 -> 단일 문자 압축 프로토콜 스트링"""
        PixelPaletteManager.initialize_palette()
        matrix = cls.image_to_matrix(image_path)
        return cls.serialize_matrix(matrix)