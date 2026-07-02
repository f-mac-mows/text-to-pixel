import json
import random

# ==========================================
# 1. 16x16 행 단위 압축 및 괄호 기반 반복 패턴 압축 함수
# ==========================================
def encode_with_bracket_repeat(matrix: list) -> str:
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
        
    compressed_chunks = []
    i = 0
    num_rows = len(row_strings)
    
    while i < num_rows:
        current_row_str = row_strings[i]
        match_count = 1
        while i + match_count < num_rows and row_strings[i + match_count] == current_row_str:
            match_count += 1
            
        if match_count > 1:
            compressed_chunks.append(f"[Re{match_count}][{current_row_str} 1L]")
            i += match_count
        else:
            compressed_chunks.append(f"{current_row_str} 1L")
            i += 1
            
    result = " ".join(compressed_chunks).strip()
    if result.endswith("1L]"): result = result[:-3] + "]"
    elif result.endswith("1L"): result = result[:-2].strip()
    return result


def encode_checkerboard_smart(matrix: list) -> str:
    """ 체커보드 전용 고밀도 교차 반복 압축 """
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

    bg_row = row_strings[0]      # "16W"
    pattern_a = row_strings[2]   # "2W 1G 1W 1G ..."
    pattern_b = row_strings[3]   # "3W 1G 1W 1G ..."

    compressed = f"[Re2][{bg_row} 1L] [Re6][{pattern_a} 1L {pattern_b} 1L] [Re2][{bg_row} 1L]"
    if compressed.endswith("1L]"): compressed = compressed[:-3] + "]"
    return compressed

# ==========================================
# 2. 고정 크기(16x16) 도형 매트릭스 빌더
# ==========================================
def create_empty_matrix(size=16):
    return [['W' for _ in range(size)] for _ in range(size)]

def draw_square(size=16, fill_char='R', border_only=False):
    matrix = create_empty_matrix(size)
    margin = 3
    start, end = margin, size - margin - 1
    for r in range(start, end + 1):
        for c in range(start, end + 1):
            if border_only:
                if r == start or r == end or c == start or c == end: matrix[r][c] = fill_char
            else: matrix[r][c] = fill_char
    return matrix

def draw_triangle(size=16, fill_char='B', border_only=False):
    matrix = create_empty_matrix(size)
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

def draw_cross(size=16, fill_char='K'):
    matrix = create_empty_matrix(size)
    mid = size // 2
    for i in range(2, size - 2):
        matrix[mid][i] = fill_char
        matrix[i][mid] = fill_char
    return matrix

def draw_checkerboard(size=16, fill_char='Y'):
    matrix = create_empty_matrix(size)
    for r in range(2, size - 2):
        for c in range(2, size - 2):
            if (r + c) % 2 == 0: matrix[r][c] = fill_char
    return matrix

# ==========================================
# 3. 자연어 명령어 데이터 증강 (가짓수를 5개 -> 35개로 확장)
# ==========================================
def generate_prompts(color_name, shape_name):
    """
    다양한 어조, 대소문자, 문장 구조를 조합하여 하나의 도형/색상 쌍마다 
    35개의 유니크한 프롬프트를 확정적으로 생성합니다.
    """
    templates = [
        f"draw a {color_name} {shape_name}",
        f"can you create a {color_name} {shape_name}?",
        f"generate a {shape_name} in {color_name}",
        f"show me a {color_name} {shape_name} pixel art",
        f"make a {shape_name} colored {color_name}",
        f"render a {color_name} {shape_name} grid",
        f"please create a {color_name} {shape_name}",
        f"paint a {shape_name} with {color_name}",
        f"i want a {color_name} {shape_name}",
        f"build a {shape_name} using {color_name} color",
        # 대문자 믹스 및 명령조 변형 추가
        f"Draw a {color_name} {shape_name}",
        f"Generate a {color_name} {shape_name} pixel art",
        f"Make a {color_name} {shape_name}",
        f"Create a beautiful {color_name} {shape_name}",
        f"Could you draw a {color_name} {shape_name} for me?",
        f"Display a {shape_name} colored {color_name}",
        f"Sketch a {color_name} {shape_name}",
        f"Output a {color_name} {shape_name} matrix",
        f"Plot a {shape_name} in {color_name}",
        f"Design a {color_name} {shape_name} pattern",
    ]
    
    # 템플릿의 양을 좀 더 확보하기 위해 추가 변형 결합
    extended = []
    for t in templates:
        extended.append(t)
        extended.append(t.replace("a ", "a simple "))
    
    # 중복 제거 후 정확히 상위 35개만 슬라이싱하여 일관된 볼륨 유지
    unique_prompts = list(dict.fromkeys(extended))
    return unique_prompts[:35]

# ==========================================
# 4. 데이터셋 빌드 실행 및 저장 (10개 색상 전면 반영 - 총 2,100개 세트)
# ==========================================
if __name__ == "__main__":
    output_file = "pixel_dataset_v3.jsonl"
    dataset = []

    # 🎨 렌더러의 COLOR_MAP과 완벽하게 일치하는 10개 색상 라인업 확장
    colors = [
        ('red', 'R'), ('blue', 'B'), ('green', 'G'), ('yellow', 'Y'), ('black', 'K'),
        ('purple', 'P'), ('orange', 'O'), ('pink', 'H'), ('white', 'W'), ('gray', 'A')
    ]

    # 색상(10) * 도형종류(6) * 프롬프트수(35) = 정확히 2,100개 샘플 대량 빌드
    for color_name, char in colors:
        for pmt in generate_prompts(color_name, "square"):
            dataset.append({"input": pmt, "output": encode_with_bracket_repeat(draw_square(16, char, False))})
            
        for pmt in generate_prompts(color_name, "hollow square"):
            dataset.append({"input": pmt, "output": encode_with_bracket_repeat(draw_square(16, char, True))})
            
        for pmt in generate_prompts(color_name, "triangle"):
            dataset.append({"input": pmt, "output": encode_with_bracket_repeat(draw_triangle(16, char, False))})
            
        for pmt in generate_prompts(color_name, "hollow triangle"):
            dataset.append({"input": pmt, "output": encode_with_bracket_repeat(draw_triangle(16, char, True))})
            
        for pmt in generate_prompts(color_name, "cross"):
            dataset.append({"input": pmt, "output": encode_with_bracket_repeat(draw_cross(16, char))})
            
        # 체커보드도 10개 색상 전체에 대해 고밀도 교차 압축 적용
        for pmt in generate_prompts(color_name, "checkerboard"):
            dataset.append({"input": pmt, "output": encode_checkerboard_smart(draw_checkerboard(16, char))})

    # 학습 효율 극대화를 위한 무작위 셔플
    random.shuffle(dataset)

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[*] 스케일업 완료: 10개 전색상 반영 고밀도 데이터 총 {len(dataset)}개 빌드 완료!")
    print(f"[*] 데이터 구성 확인:")
    print(f" - 전체 데이터 개수: {len(dataset)} 개 (2000개 고지 돌파)")