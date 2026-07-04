import json
import random
import os
from pixel_config import PixelPaths
from pixel_deserializer import PixelPaletteManager, PixelDeserializer
# 💡 격리형 세부 로그 관리를 위한 모듈 임포트
from pixel_logger import setup_logger

# ==========================================
# 1. 고정 크기(16x16) 수학적 도형 매트릭스 빌더
# ==========================================
def create_empty_matrix(size=16, bg_char='c255'):
    return [[bg_char for _ in range(size)] for _ in range(size)]

def draw_square(size=16, fill_char='c000', border_only=False, bg_char='c255'):
    matrix = create_empty_matrix(size, bg_char)
    margin = 3
    start, end = margin, size - margin - 1
    for r in range(start, end + 1):
        for c in range(start, end + 1):
            if border_only:
                if r == start or r == end or c == start or c == end: matrix[r][c] = fill_char
            else: matrix[r][c] = fill_char
    return matrix

def draw_triangle(size=16, fill_char='c000', border_only=False, bg_char='c255'):
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

def draw_cross(size=16, fill_char='c000', bg_char='c255'):
    matrix = create_empty_matrix(size, bg_char)
    mid = size // 2
    for i in range(2, size - 2):
        matrix[mid][i] = fill_char
        matrix[i][mid] = fill_char
    return matrix

def draw_checkerboard(size=16, fill_char='c000', bg_char='c255'):
    matrix = create_empty_matrix(size, bg_char)
    for r in range(2, size - 2):
        for c in range(2, size - 2):
            if (r + c) % 2 == 0: matrix[r][c] = fill_char
    return matrix

def draw_diamond(size=16, fill_char='c000', bg_char='c255'):
    matrix = create_empty_matrix(size, bg_char)
    mid = size // 2
    margin = 2
    for r in range(margin, size - margin + 1):
        dist = abs(r - mid)
        width = (mid - margin) - dist
        for c in range(mid - width, mid + width + 1):
            matrix[r][c] = fill_char
    return matrix

def draw_stripes(size=16, fill_char='c000', vertical=True, bg_char='c255'):
    matrix = create_empty_matrix(size, bg_char)
    for r in range(2, size - 2):
        for c in range(2, size - 2):
            if vertical and c % 2 == 0: matrix[r][c] = fill_char
            elif not vertical and r % 2 == 0: matrix[r][c] = fill_char
    return matrix

def draw_border_frame(size=16, fill_char='c000', bg_char='c255'):
    matrix = create_empty_matrix(size, bg_char)
    for r in range(0, size):
        for c in range(0, size):
            if r < 2 or r >= size - 2 or c < 2 or c >= size - 2: matrix[r][c] = fill_char
    return matrix


# ==========================================
# 2. 자연어 명령어 프롬프트 생성기
# ==========================================
def generate_prompts(color_name, shape_name):
    core = f"{color_name} {shape_name}"
    prompts = [
        core, core.capitalize(), f"a {core}", f"draw a {core}", f"make a {core} pixel art", f"generate a {core}"
    ]
    return list(dict.fromkeys(prompts))

def generate_prompts_from_filename(filename):
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
# 3. 데이터셋 빌드 메인스트림
# ==========================================
if __name__ == "__main__":
    # 💡 세부 공정 격리 로거 초기화 (logs/data_generator/ 폴더 하위에 파일 생성)
    logger, log_path = setup_logger(mode="data_generator")

    logger.info("[*] 🚀 256색 고밀도 레이아웃 지원 하이브리드 파이프라인 가동...")
    logger.info(f" └─ 생성될 데이터셋 타깃: {PixelPaths.TOTAL_DATA}\n")
    
    dataset = []

    # 3:3:2 비트 컬러 명세 구조
    colors_spec = [
        ('red', 'c224'), ('blue', 'c003'), ('green', 'c028'), ('yellow', 'c252'), ('black', 'c000'),
        ('white', 'c255'), ('purple', 'c131'), ('orange', 'c248'), ('pink', 'c243'), ('gray', 'c146'),
        ('brown', 'c073'), ('sky blue', 'c155'), ('lime', 'c061'), ('mint', 'c123'), ('gold', 'c249'),
        ('silver', 'c182'), ('navy', 'c001'), ('olive', 'c072'), ('beige', 'c251'), ('magenta', 'c227')
    ]

    # [공정 1: 수식형 벡터 조합 데이터 빌드 생성기]
    for color_name, char in colors_spec:
        bg = 'c000' if char == 'c255' else 'c255'

        # 단색 매핑 보정 추가
        dataset.append({"input": color_name, "output": PixelDeserializer.serialize_matrix(draw_square(16, char, False, bg_char=bg))})
        dataset.append({"input": color_name.capitalize(), "output": PixelDeserializer.serialize_matrix(draw_square(16, char, False, bg_char=bg))})

        for pmt in generate_prompts(color_name, "square"):
            dataset.append({"input": pmt, "output": PixelDeserializer.serialize_matrix(draw_square(16, char, False, bg_char=bg))})
        for pmt in generate_prompts(color_name, "hollow square"):
            dataset.append({"input": pmt, "output": PixelDeserializer.serialize_matrix(draw_square(16, char, True, bg_char=bg))})
        for pmt in generate_prompts(color_name, "triangle"):
            dataset.append({"input": pmt, "output": PixelDeserializer.serialize_matrix(draw_triangle(16, char, False, bg_char=bg))})
        for pmt in generate_prompts(color_name, "hollow triangle"):
            dataset.append({"input": pmt, "output": PixelDeserializer.serialize_matrix(draw_triangle(16, char, True, bg_char=bg))})
        for pmt in generate_prompts(color_name, "cross"):
            dataset.append({"input": pmt, "output": PixelDeserializer.serialize_matrix(draw_cross(16, char, bg_char=bg))})
        for pmt in generate_prompts(color_name, "diamond"):
            dataset.append({"input": pmt, "output": PixelDeserializer.serialize_matrix(draw_diamond(16, char, bg_char=bg))})
        for pmt in generate_prompts(color_name, "border frame"):
            dataset.append({"input": pmt, "output": PixelDeserializer.serialize_matrix(draw_border_frame(16, char, bg_char=bg))})
        for pmt in generate_prompts(color_name, "horizontal stripes"):
            dataset.append({"input": pmt, "output": PixelDeserializer.serialize_matrix(draw_stripes(16, char, vertical=False, bg_char=bg))})
        for pmt in generate_prompts(color_name, "checkerboard"):
            dataset.append({"input": pmt, "output": PixelDeserializer.serialize_matrix(draw_checkerboard(16, char, bg_char=bg))})

    # [공정 2: 순수 도형 단독 데이터 생성]
    shape_names = ["square", "hollow square", "triangle", "hollow triangle", "cross", "diamond", "border frame", "horizontal stripes", "checkerboard"]
    for shape_name in shape_names:
        if shape_name == "horizontal stripes":
            out_str = PixelDeserializer.serialize_matrix(draw_stripes(16, 'c000', vertical=False, bg_char='c255'))
        elif shape_name == "checkerboard":
            out_str = PixelDeserializer.serialize_matrix(draw_checkerboard(16, 'c000', bg_char='c255'))
        elif shape_name == "square":
            out_str = PixelDeserializer.serialize_matrix(draw_square(16, 'c000', False, bg_char='c255'))
        elif shape_name == "hollow square":
            out_str = PixelDeserializer.serialize_matrix(draw_square(16, 'c000', True, bg_char='c255'))
        elif shape_name == "triangle":
            out_str = PixelDeserializer.serialize_matrix(draw_triangle(16, 'c000', False, bg_char='c255'))
        elif shape_name == "hollow triangle":
            out_str = PixelDeserializer.serialize_matrix(draw_triangle(16, 'c000', True, bg_char='c255'))
        elif shape_name == "cross":
            out_str = PixelDeserializer.serialize_matrix(draw_cross(16, 'c000', bg_char='c255'))
        elif shape_name == "diamond":
            out_str = PixelDeserializer.serialize_matrix(draw_diamond(16, 'c000', bg_char='c255'))
        elif shape_name == "border frame":
            out_str = PixelDeserializer.serialize_matrix(draw_border_frame(16, 'c000', bg_char='c255'))

        dataset.append({"input": shape_name, "output": out_str})
        dataset.append({"input": shape_name.capitalize(), "output": out_str})

    # [공정 3: 실전 고해상도 수작업 에셋 파일 스캔 인코딩 공정 (통합 엔진 연동)]
    sprite_dir = os.path.expanduser("~/Assets/Sprites")
    if os.path.exists(sprite_dir):
        logger.info(f"[*] 📂 수작업 고화질 에셋 저장소 스캔 가동: {sprite_dir}")
        parsed_count = 0
        
        for root, dirs, files in os.walk(sprite_dir):
            image_files = [f for f in files if f.lower().endswith('.png')]
            for img_file in image_files:
                try:
                    full_path = os.path.join(root, img_file)
                    
                    # 고성능 체커보드 탐지 엔진으로 다이어트 문자열 추출
                    out_str = PixelDeserializer.deserialize_image(full_path)
                    
                    for input_prompt in generate_prompts_from_filename(img_file):
                        dataset.append({"input": input_prompt, "output": out_str})
                        
                    parsed_count += 1
                except Exception as e:
                    logger.error(f" └─ ⚠️ 파일 파싱 실패 ({img_file}): {e}")
                    
        logger.info(f" ├─ 파싱 완료! 총 {parsed_count}개의 고해상도 커스텀 에셋이 프로토콜화되었습니다.")

    # 셔플 및 디렉토리 생성 안전장치 고도화
    random.shuffle(dataset)
    dir_name = os.path.dirname(PixelPaths.TOTAL_DATA)
    if dir_name: 
        os.makedirs(dir_name, exist_ok=True)
    
    with open(PixelPaths.TOTAL_DATA, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"\n[*] 완료: 총 {len(dataset)}개의 256색 교차 패턴 대응형 통합 데이터셋 빌드 성공!")
    logger.info(f"[✓] 데이터셋 빌더 세부 로그 저장 완료: {log_path}")