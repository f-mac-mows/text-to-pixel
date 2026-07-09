import json
import random
import os
from pixel_config import PixelPaths
from pixel_deserializer import PixelDeserializer
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
# 2. 하이브리드 프롬프트 확장 엔진
# ==========================================
def generate_advanced_prompts(color_name, shape_name):
    """[공정 1] 색상 + 도형 페어용 단순/강화형 프롬프트 세트 빌더"""
    # 1) 단순 키워드 스타일 (Simple)
    simple = [
        f"{color_name} {shape_name}",
        f"{color_name} {shape_name} pixel",
        f"pixel art {color_name} {shape_name}"
    ]
    # 2) 서사적 문장 구조 스타일 (Enhanced)
    enhanced = [
        f"a {color_name} {shape_name}",
        f"a pixel art sprite of a {color_name} {shape_name}",
        f"draw a {color_name} {shape_name} icon",
        f"generate a simple {color_name} {shape_name} graphic",
        f"a 16x16 pixel art shape of {color_name} {shape_name}"
    ]
    
    raw_list = simple + enhanced
    # 대문자 버전 자동 증강 및 중복 제거
    final_list = []
    for p in raw_list:
        final_list.append(p)
        final_list.append(p.capitalize())
    return list(dict.fromkeys(final_list))

def generate_shape_only_prompts(shape_name):
    """[공정 2] 순수 도형 단독용 단순/강화형 프롬프트 세트 빌더"""
    # 1) 단순 키워드 스타일 (Simple)
    simple = [
        shape_name,
        f"{shape_name} shape",
        f"pixel {shape_name}"
    ]
    # 2) 서사적 문장 구조 스타일 (Enhanced)
    if "stripes" in shape_name:
        enhanced = [f"a pattern of horizontal stripes", f"black and white horizontal stripes pixel art"]
    elif "checkerboard" in shape_name:
        enhanced = [f"a black and white checkerboard pattern", f"classic 16x16 checkerboard grid"]
    elif "hollow" in shape_name:
        enhanced = [f"an outline of a {shape_name.replace('hollow ', '')}", f"a hollow geometric {shape_name.replace('hollow ', '')} shape"]
    else:
        enhanced = [f"a simple geometric {shape_name}", f"a 16x16 minimalist {shape_name} sprite"]

    raw_list = simple + enhanced
    final_list = []
    for p in raw_list:
        final_list.append(p)
        final_list.append(p.capitalize())
    return list(dict.fromkeys(final_list))


# ============================================================
# [중간 공정] 추상 텍스트 -> 시각적 영어 자연어 치환 매퍼
# ============================================================
class TokenMapper:
    POTION_STYLE_MAP = {'t1': 'basic', 't2': 'sleek', 't3': 'ornate', 't4': 'reinforced'}
    CONSUMABLE_SIZE_MAP = {'t1': 'small', 't2': 'medium', 't3': 'large', 't4': 'grand'}
    BOOK_STYLE_MAP = {
        'type1': 'gem_basic', 'type2': 'gem_refined', 'type3': 'rune_alpha', 'type4': 'rune_beta',
        'type5': 'ornate_border', 'type6': 'magical_glow', 'type7': 'ancient_cross', 'type8': 'royal_gold',
        'type9': 'enchanted', 'type10': 'mythic'
    }

    @classmethod
    def refine_filename_tokens(cls, root_dir, filename):
        base_name = os.path.splitext(filename)[0]
        tokens = base_name.split('_')
        norm_root = root_dir.lower()

        if "potions" in norm_root:
            refined = [cls.POTION_STYLE_MAP.get(t, t) for t in tokens]
            return " ".join(refined)
        elif "consumables" in norm_root:
            refined = [cls.CONSUMABLE_SIZE_MAP.get(t, t) for t in tokens]
            return " ".join(refined).replace("  ", " ")
        elif "books" in norm_root:
            refined = [cls.BOOK_STYLE_MAP.get(t, t) for t in tokens]
            return " ".join(refined)
        else:
            return base_name.replace('_', ' ')
        

# ==========================================
# 3. 데이터셋 빌드 메인스트림
# ==========================================
if __name__ == "__main__":
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

    # [공정 1: 수식형 벡터 조합 데이터 빌드 생성기 (고도화)]
    for color_name, char in colors_spec:
        bg = 'c000' if char == 'c255' else 'c255'

        # 단색 본연의 데이터 매핑
        dataset.append({"input": color_name, "output": PixelDeserializer.serialize_matrix(draw_square(16, char, False, bg_char=bg))})
        dataset.append({"input": color_name.capitalize(), "output": PixelDeserializer.serialize_matrix(draw_square(16, char, False, bg_char=bg))})
        dataset.append({"input": f"solid {color_name} color", "output": PixelDeserializer.serialize_matrix(draw_square(16, char, False, bg_char=bg))})

        # 다중 앵글 프롬프트 팩 주입
        for pmt in generate_advanced_prompts(color_name, "square"):
            dataset.append({"input": pmt, "output": PixelDeserializer.serialize_matrix(draw_square(16, char, False, bg_char=bg))})
        for pmt in generate_advanced_prompts(color_name, "hollow square"):
            dataset.append({"input": pmt, "output": PixelDeserializer.serialize_matrix(draw_square(16, char, True, bg_char=bg))})
        for pmt in generate_advanced_prompts(color_name, "triangle"):
            dataset.append({"input": pmt, "output": PixelDeserializer.serialize_matrix(draw_triangle(16, char, False, bg_char=bg))})
        for pmt in generate_advanced_prompts(color_name, "hollow triangle"):
            dataset.append({"input": pmt, "output": PixelDeserializer.serialize_matrix(draw_triangle(16, char, True, bg_char=bg))})
        for pmt in generate_advanced_prompts(color_name, "cross"):
            dataset.append({"input": pmt, "output": PixelDeserializer.serialize_matrix(draw_cross(16, char, bg_char=bg))})
        for pmt in generate_advanced_prompts(color_name, "diamond"):
            dataset.append({"input": pmt, "output": PixelDeserializer.serialize_matrix(draw_diamond(16, char, bg_char=bg))})
        for pmt in generate_advanced_prompts(color_name, "border frame"):
            dataset.append({"input": pmt, "output": PixelDeserializer.serialize_matrix(draw_border_frame(16, char, bg_char=bg))})
        for pmt in generate_advanced_prompts(color_name, "horizontal stripes"):
            dataset.append({"input": pmt, "output": PixelDeserializer.serialize_matrix(draw_stripes(16, char, vertical=False, bg_char=bg))})
        for pmt in generate_advanced_prompts(color_name, "checkerboard"):
            dataset.append({"input": pmt, "output": PixelDeserializer.serialize_matrix(draw_checkerboard(16, char, bg_char=bg))})

    # [공정 2: 순수 도형 단독 데이터 생성 (고도화)]
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

        # 고도화된 단독 도형 프롬프트 세트 루프 적재
        for shape_prompt in generate_shape_only_prompts(shape_name):
            dataset.append({"input": shape_prompt, "output": out_str})

    # [공정 3: 실전 고해상도 수작업 에셋 파일 스캔 인코딩 공정]
    sprite_dir = os.path.expanduser("~/Assets/Sprites")
    if os.path.exists(sprite_dir):
        logger.info(f"[*] 📂 수작업 고화질 에셋 저장소 스캔 가동: {sprite_dir}")
        parsed_count = 0
        
        for root, dirs, files in os.walk(sprite_dir):
            image_files = [f for f in files if f.lower().endswith('.png')]
            for img_file in image_files:
                try:
                    full_path = os.path.join(root, img_file)
                    out_str = PixelDeserializer.deserialize_image(full_path)
                    
                    norm_root = root.lower()
                    asset_prompts = []

                    refined_desc = TokenMapper.refine_filename_tokens(root, img_file)
                    tokens = refined_desc.split()

                    if "books" in norm_root:
                        color = " ".join([t for t in tokens if t in ['dark', 'light', 'gray', 'silver', 'red', 'orange', 'gold', 'yellow', 'lime', 'green', 'teal', 'cyan', 'blue', 'purple', 'pink']])
                        style = tokens[-1] if tokens[-1] in ['basic', 'refined', 'alpha', 'beta', 'border', 'glow', 'cross', 'gold', 'enchanted', 'mythic'] else 'basic'
                        
                        asset_prompts.append(f"{color} book {style}")
                        asset_prompts.append(f"pixel art {color} book {style}")
                        
                        if style in ['basic', 'refined']:
                            rich_desc = f"a book with a {color} gem"
                        elif 'rune' in refined_desc:
                            rich_desc = f"a {color} book engraved with an ancient rune"
                        elif style in ['glow', 'enchanted', 'mythic']:
                            rich_desc = f"an enchanted {color} magical book"
                        else:
                            rich_desc = f"a {color} book in {style} style"
                            
                        asset_prompts.append(rich_desc)
                        asset_prompts.append(f"a pixel art of {rich_desc}")

                    elif "potions" in norm_root:
                        color = tokens[0]
                        shape = tokens[1] if len(tokens) > 1 else 'standard'
                        deco = tokens[2] if len(tokens) > 2 else 'basic'
                        
                        asset_prompts.append(f"{color} {shape} {deco} potion")
                        asset_prompts.append(f"{color} {shape} potion {deco}")
                        
                        rich_desc = f"a {shape} {color} potion bottle with a {deco} top"
                        asset_prompts.append(rich_desc)
                        asset_prompts.append(f"a pixel art sprite of {rich_desc}")

                    elif "consumables" in norm_root:
                        if len(tokens) >= 5:
                            mat, container, size, liquid_type, color = tokens[0], tokens[1], tokens[2], tokens[3], tokens[4]
                            asset_prompts.append(f"{size} {mat} {container} {color} {liquid_type}")
                            
                            rich_desc = f"a {size} {mat} {container} filled with {color} {liquid_type} liquid"
                            asset_prompts.append(rich_desc)
                            asset_prompts.append(f"pixel art of {rich_desc}")
                        else:
                            asset_prompts.append(refined_desc.replace('_', ' '))

                    else:
                        asset_prompts.append(refined_desc)
                        asset_prompts.append(f"a {refined_desc}")
                        asset_prompts.append(f"pixel art of {refined_desc}")

                    extended_prompts = []
                    for pmt in asset_prompts:
                        extended_prompts.append(pmt)
                        extended_prompts.append(pmt.capitalize())
                        
                    final_prompts = list(dict.fromkeys(extended_prompts))
                    
                    for input_prompt in final_prompts:
                        dataset.append({"input": input_prompt, "output": out_str})
                        
                    parsed_count += 1
                except Exception as e:
                    logger.error(f" └─ ⚠️ 파일 파싱 실패 ({img_file}): {e}")

    # 셔플 및 안전장치
    random.shuffle(dataset)
    dir_name = os.path.dirname(PixelPaths.TOTAL_DATA)
    if dir_name: 
        os.makedirs(dir_name, exist_ok=True)
    
    with open(PixelPaths.TOTAL_DATA, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"\n[*] 완료: 총 {len(dataset)}개의 256색 교차 패턴 대응형 통합 데이터셋 빌드 성공!")
    logger.info(f"[✓] 데이터셋 빌더 세부 로그 저장 완료: {log_path}")