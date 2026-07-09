import json
import random
import os
from PIL import Image
from pixel_config import PixelPaths

# ==========================================
# 1. 16x16 행 단위 압축 및 [ BPE 친화적 ] 반복 패턴 압축 함수
# ==========================================
def encode_with_bracket_repeat(matrix: list) -> str:
    row_strings = []
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
            
    result = " ".join(compressed_chunks).strip()
    
    # 끝단 1X 보정 필터 수정 (split 기반 토큰 안전 정제)
    tokens = result.split()
    if not tokens: return ""
    if tokens[-1] == "]" and len(tokens) >= 2 and tokens[-2] == "1X":
        del tokens[-2]
    elif tokens[-1] == "1X":
        del tokens[-1]
    return " ".join(tokens)


def encode_checkerboard_smart(matrix: list) -> str:
    """ 체커보드 및 격자 패턴 전용 고밀도 교차 반복 압축 """
    row_strings = []
    for row in matrix:
        rle_tokens = []
        current_char = row[0]
        current_len = 1
        for char in row[1:]:
            if char == current_char: current_len += 1
            else:
                rle_tokens.append(f"{current_len}{current_char}")
                current_char = char
                current_len = 1
        rle_tokens.append(f"{current_len}{current_char}")
        row_strings.append(" ".join(rle_tokens))

    bg_row = row_strings[0]      
    pattern_a = row_strings[2]   
    pattern_b = row_strings[3]   

    compressed = f"[ Re2 ] [ {bg_row} 1X ] [ Re6 ] [ {pattern_a} 1X {pattern_b} 1X ] [ Re2 ] [ {bg_row} 1X ]"
    if compressed.endswith("1X ]"): 
        compressed = compressed[:-4] + " ]"
    return compressed


def encode_stripes_smart(matrix: list) -> str:
    """ 가로 줄무늬 전용 고밀도 교차 반복 압축 """
    row_strings = []
    for row in matrix:
        rle_tokens = []
        current_char = row[0]
        current_len = 1
        for char in row[1:]:
            if char == current_char: current_len += 1
            else:
                rle_tokens.append(f"{current_len}{current_char}")
                current_char = char
                current_len = 1
        rle_tokens.append(f"{current_len}{current_char}")
        row_strings.append(" ".join(rle_tokens))

    bg_row = row_strings[0]
    pattern = row_strings[2]

    compressed = f"[ Re2 ] [ {bg_row} 1X ] [ Re6 ] [ {pattern} 1X {bg_row} 1X ] [ Re2 ] [ {bg_row} 1X ]"
    if compressed.endswith("1X ]"): 
        compressed = compressed[:-4] + " ]"
    return compressed


# ==========================================
# 1.5 실전 스프라이트 이미지 역직렬화 및 [다이어트] 프롬프트 증강 서브루틴
# ==========================================
COLOR_PALETTE = {
    'R': (255, 0, 0), 'B': (0, 0, 255), 'G': (0, 255, 0), 'Y': (255, 255, 0), 'K': (0, 0, 0),
    'P': (128, 0, 128), 'O': (255, 165, 0), 'H': (255, 192, 203), 'W': (255, 255, 255), 'A': (128, 128, 128),
    'N': (165, 42, 42), 'S': (135, 206, 235), 'L': (50, 205, 50), 'M': (170, 240, 209), 'D': (255, 215, 0),
    'V': (192, 192, 192), 'U': (0, 0, 128), 'E': (128, 128, 0), 'Z': (245, 245, 220), 'J': (255, 0, 255)
}

def get_nearest_color_char(r, g, b):
    min_distance = float('inf')
    closest_char = 'W'
    for char, palette_rgb in COLOR_PALETTE.items():
        distance = (r - palette_rgb[0])**2 + (g - palette_rgb[1])**2 + (b - palette_rgb[2])**2
        if distance < min_distance:
            min_distance = distance
            closest_char = char
    return closest_char

def image_to_matrix(image_path, target_size=16):
    img = Image.open(image_path).convert('RGB')
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.Resampling.NEAREST)
    matrix = []
    for r in range(target_size):
        row = []
        for c in range(target_size):
            rgb = img.getpixel((c, r))
            row.append(get_nearest_color_char(*rgb))
        matrix.append(row)
    return matrix

def generate_prompts_from_filename(filename):
    """ 파일명을 가공하여 불필요한 노이즈를 뺀 에센셜 실전 프롬프트 세트만 반환합니다. """
    base_name = os.path.splitext(filename)[0]
    clean_name = base_name.replace("_", " ").replace("-", " ").strip()
    
    prompts = [
        clean_name, 
        clean_name.capitalize(), 
        f"a {clean_name}", 
        f"draw a {clean_name}", 
        f"make a {clean_name} pixel art", 
        f"generate a {clean_name} sprite"
    ]
    return list(dict.fromkeys(prompts))


# ==========================================
# 2. 고정 크기(16x16) 도형 매트릭스 빌더 (배경색 bg_char 주입 가능 구조)
# ==========================================
def create_empty_matrix(size=16, bg_char='W'):
    return [[bg_char for _ in range(size)] for _ in range(size)]

def draw_square(size=16, fill_char='R', border_only=False, bg_char='W'):
    matrix = create_empty_matrix(size, bg_char)
    margin = 3
    start, end = margin, size - margin - 1
    for r in range(start, end + 1):
        for c in range(start, end + 1):
            if border_only:
                if r == start or r == end or c == start or c == end: matrix[r][c] = fill_char
            else: matrix[r][c] = fill_char
    return matrix

def draw_triangle(size=16, fill_char='B', border_only=False, bg_char='W'):
    matrix = create_empty_matrix(size, bg_char)
    mid = size // 2
    for r in range(2, size - 2):
        width = r - 2
        start_c = max(0, mid - width)
        end_c = min(size, mid + width + 1)
        for c in range(start_c, end_c):
            if border_only:
                if r == 2 or r == size - 3 or c == start_c or c == end_c - 1: matrix[r][c] = fill_char
            else: matrix[r][c] = fill_char
    return matrix

def draw_cross(size=16, fill_char='K', bg_char='W'):
    matrix = create_empty_matrix(size, bg_char)
    mid = size // 2
    for i in range(2, size - 2):
        matrix[mid][i] = fill_char
        matrix[i][mid] = fill_char
    return matrix

def draw_checkerboard(size=16, fill_char='Y', bg_char='W'):
    matrix = create_empty_matrix(size, bg_char)
    for r in range(2, size - 2):
        for c in range(2, size - 2):
            if (r + c) % 2 == 0: matrix[r][c] = fill_char
    return matrix

def draw_diamond(size=16, fill_char='P', bg_char='W'):
    matrix = create_empty_matrix(size, bg_char)
    mid = size // 2
    margin = 2
    for r in range(margin, size - margin + 1):
        dist = abs(r - mid)
        width = (mid - margin) - dist
        for c in range(mid - width, mid + width + 1):
            matrix[r][c] = fill_char
    return matrix

def draw_stripes(size=16, fill_char='O', vertical=True, bg_char='W'):
    matrix = create_empty_matrix(size, bg_char)
    for r in range(2, size - 2):
        for c in range(2, size - 2):
            if vertical and c % 2 == 0: matrix[r][c] = fill_char
            elif not vertical and r % 2 == 0: matrix[r][c] = fill_char
    return matrix

def draw_border_frame(size=16, fill_char='A', bg_char='W'):
    matrix = create_empty_matrix(size, bg_char)
    for r in range(0, size):
        for c in range(0, size):
            if r < 2 or r >= size - 2 or c < 2 or c >= size - 2: matrix[r][c] = fill_char
    return matrix


# ==========================================
# 3. 자연어 명령어 데이터 [다이어트형] 증강
# ==========================================
def generate_prompts(color_name, shape_name):
    """ 어텐션 분산을 막기 위해 60개에 달하던 서술문을 핵심 6개 구문으로 타이트하게 압축 """
    core = f"{color_name} {shape_name}"
    prompts = [
        core,
        core.capitalize(),
        f"a {core}",
        f"draw a {core}",
        f"make a {core} pixel art",
        f"generate a {core}"
    ]
    return list(dict.fromkeys(prompts))


# ==========================================
# 4. 데이터셋 빌드 가동
# ==========================================
if __name__ == "__main__":
    print(f"[*] 🚀 고효율 정제형 하이브리드 파이프라인 가동...")
    print(f" └─ 생성될 데이터셋 타깃: {PixelPaths.TOTAL_DATA}\n")
    
    dataset = []

    colors = [
        ('red', 'R'), ('blue', 'B'), ('green', 'G'), ('yellow', 'Y'), ('black', 'K'),
        ('purple', 'P'), ('orange', 'O'), ('pink', 'H'), ('white', 'W'), ('gray', 'A'),
        ('brown', 'N'), ('sky blue', 'S'), ('lime', 'L'), ('mint', 'M'), ('gold', 'D'),
        ('silver', 'V'), ('navy', 'U'), ('olive', 'E'), ('beige', 'Z'), ('magenta', 'J')
    ]

    # [1. 조합형 수식 데이터 생성 엔진]
    for color_name, char in colors:
        # 💡 핵심 수정 사항: 그릴 색상이 하얀색('W')이면 배경을 검은색('K')으로 설정, 그 외엔 하얀색('W')
        bg = 'K' if char == 'W' else 'W'

        # 단색 숏 프롬프트 대응 빌드 수정
        dataset.append({"input": color_name, "output": encode_with_bracket_repeat(draw_square(16, char, False, bg_char=bg))})
        dataset.append({"input": color_name.capitalize(), "output": encode_with_bracket_repeat(draw_square(16, char, False, bg_char=bg))})

        for pmt in generate_prompts(color_name, "square"):
            dataset.append({"input": pmt, "output": encode_with_bracket_repeat(draw_square(16, char, False, bg_char=bg))})
        for pmt in generate_prompts(color_name, "hollow square"):
            dataset.append({"input": pmt, "output": encode_with_bracket_repeat(draw_square(16, char, True, bg_char=bg))})
        for pmt in generate_prompts(color_name, "triangle"):
            dataset.append({"input": pmt, "output": encode_with_bracket_repeat(draw_triangle(16, char, False, bg_char=bg))})
        for pmt in generate_prompts(color_name, "hollow triangle"):
            dataset.append({"input": pmt, "output": encode_with_bracket_repeat(draw_triangle(16, char, True, bg_char=bg))})
        for pmt in generate_prompts(color_name, "cross"):
            dataset.append({"input": pmt, "output": encode_with_bracket_repeat(draw_cross(16, char, bg_char=bg))})
        for pmt in generate_prompts(color_name, "diamond"):
            dataset.append({"input": pmt, "output": encode_with_bracket_repeat(draw_diamond(16, char, bg_char=bg))})
        for pmt in generate_prompts(color_name, "border frame"):
            dataset.append({"input": pmt, "output": encode_with_bracket_repeat(draw_border_frame(16, char, bg_char=bg))})
        for pmt in generate_prompts(color_name, "horizontal stripes"):
            dataset.append({"input": pmt, "output": encode_stripes_smart(draw_stripes(16, char, vertical=False, bg_char=bg))})
        for pmt in generate_prompts(color_name, "checkerboard"):
            dataset.append({"input": pmt, "output": encode_checkerboard_smart(draw_checkerboard(16, char, bg_char=bg))})

    # [2. 순수 도형 단독 데이터 생성] - 기본 배경 'W'에 검은색('K')으로 단독 도형 렌더링
    shape_names = ["square", "hollow square", "triangle", "hollow triangle", "cross", "diamond", "border frame", "horizontal stripes", "checkerboard"]
    for shape_name in shape_names:
        if shape_name == "horizontal stripes":
            out_str = encode_stripes_smart(draw_stripes(16, 'K', vertical=False, bg_char='W'))
        elif shape_name == "checkerboard":
            out_str = encode_checkerboard_smart(draw_checkerboard(16, 'K', bg_char='W'))
        elif shape_name == "square":
            out_str = encode_with_bracket_repeat(draw_square(16, 'K', False, bg_char='W'))
        elif shape_name == "hollow square":
            out_str = encode_with_bracket_repeat(draw_square(16, 'K', True, bg_char='W'))
        elif shape_name == "triangle":
            out_str = encode_with_bracket_repeat(draw_triangle(16, 'K', False, bg_char='W'))
        elif shape_name == "hollow triangle":
            out_str = encode_with_bracket_repeat(draw_triangle(16, 'K', True, bg_char='W'))
        elif shape_name == "cross":
            out_str = encode_with_bracket_repeat(draw_cross(16, 'K', bg_char='W'))
        elif shape_name == "diamond":
            out_str = encode_with_bracket_repeat(draw_diamond(16, 'K', bg_char='W'))
        elif shape_name == "border frame":
            out_str = encode_with_bracket_repeat(draw_border_frame(16, 'K', bg_char='W'))

        dataset.append({"input": shape_name, "output": out_str})
        dataset.append({"input": shape_name.capitalize(), "output": out_str})

    # [3. 실전 수작업 스프라이트 픽셀 이미지 파싱 공정 (하위폴더 재귀 스캔)]
    sprite_dir = os.path.expanduser("~/Assets/Sprites")
    if os.path.exists(sprite_dir):
        print(f"[*] 📂 커스텀 스프라이트 저장소 탐색 시작: {sprite_dir}")
        parsed_count = 0
        
        for root, dirs, files in os.walk(sprite_dir):
            image_files = [f for f in files if f.lower().endswith('.png')]
            for img_file in image_files:
                try:
                    full_path = os.path.join(root, img_file)
                    matrix = image_to_matrix(full_path)
                    out_str = encode_with_bracket_repeat(matrix)
                    
                    for input_prompt in generate_prompts_from_filename(img_file):
                        dataset.append({"input": input_prompt, "output": out_str})
                        
                    parsed_count += 1
                except Exception as e:
                    print(f" └─ ⚠️ 파일 파싱 실패 ({img_file}): {e}")
                    
        print(f" ├─ 재귀 스캔 완료! 총 {parsed_count}개의 커스텀 에셋이 결합되었습니다.")

    # [마지막 정렬 및 출력]
    random.shuffle(dataset)

    # 💡 디렉터리 경로 방어 코드 수정 (빈 문자열이 아닐 때만 디렉터리 자동 생성)
    dir_name = os.path.dirname(PixelPaths.TOTAL_DATA)
    if dir_name: 
        os.makedirs(dir_name, exist_ok=True)
    
    with open(PixelPaths.TOTAL_DATA, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n[*] 완료: 총 {len(dataset)}개의 고효율 에센셜 JSONL 데이터셋 생성 성공!")