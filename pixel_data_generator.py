import json
import random
import os
import numpy as np
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
    simple = [
        f"{color_name} {shape_name}",
        f"{color_name} {shape_name} pixel",
        f"pixel art {color_name} {shape_name}"
    ]
    enhanced = [
        f"a {color_name} {shape_name}",
        f"a pixel art sprite of a {color_name} {shape_name}",
        f"draw a {color_name} {shape_name} icon",
        f"generate a simple {color_name} {shape_name} graphic",
        f"a 16x16 pixel art shape of {color_name} {shape_name}"
    ]
    raw_list = simple + enhanced
    final_list = []
    for p in raw_list:
        final_list.append(p)
        final_list.append(p.capitalize())
    return list(dict.fromkeys(final_list))

def generate_shape_only_prompts(shape_name):
    simple = [
        shape_name,
        f"{shape_name} shape",
        f"pixel {shape_name}"
    ]
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


# ============================================================
# ⚙️ [신규 개발] 패턴 압축형 픽셀 데이터 파서 및 기하학적 증강 파이프라인
# ============================================================
class ProtocolAugmenter:
    """단일 문자 패턴 인코딩(p{count}_{colors}) 방식을 사용하는 변환 및 증강 엔진"""

    @classmethod
    def deserialize_to_matrix(cls, protocol_str, size=16):
        """압축 패턴 토큰 시퀀스(또는 기존 구형 문자열 호환)를 16x16 numpy 2D 행렬로 복원"""
        # 혹시 기존 구형 [ Re ] 포맷이 들어올 경우를 대비한 방어 로직
        if "[" in protocol_str or "Re" in protocol_str:
            return cls._legacy_deserialize(protocol_str, size)

        tokens = protocol_str.split()
        grid = []
        current_row = []

        for token in tokens:
            if token == "1X":
                while len(current_row) < size: 
                    current_row.append('c255')
                grid.append(current_row[:size])
                current_row = []
                continue
                
            if token.startswith("p"):
                # 패턴 분해 (예: p2_245_235 -> count=2, colors=['c245', 'c235'])
                parts = token.split("_")
                try:
                    repeat_count = int(parts[0].replace("p", ""))
                    colors = [f"c{color}" if not color.startswith("c") else color for color in parts[1:]]
                    
                    # 정의된 반복 횟수만큼 패턴 순서대로 전개
                    for _ in range(repeat_count):
                        for color in colors:
                            current_row.append(color)
                except ValueError:
                    # 파싱 예외 방어
                    current_row.append('c255')

        if current_row:
            while len(current_row) < size: 
                current_row.append('c255')
            grid.append(current_row[:size])
            
        while len(grid) < size:
            grid.append(['c255'] * size)
            
        return np.array(grid[:size], dtype=object)

    @classmethod
    def serialize_from_matrix(cls, grid):
        """16x16 행렬을 하나의 단일 문자 압축 패턴 토큰 시퀀스로 고밀도 직렬화"""
        tokens = []
        for r in range(grid.shape[0]):
            row = [pixel.replace("c", "") for pixel in grid[r]]
            c_idx = 0
            
            while c_idx < len(row):
                # 1단계: 단일 색상 연속 압축 검사 (최대 가용 범위까지 스캔)
                run_length = 1
                while c_idx + run_length < len(row) and row[c_idx] == row[c_idx + run_length]:
                    run_length += 1
                
                if run_length >= 2:
                    tokens.append(f"p{run_length}_{row[c_idx]}")
                    c_idx += run_length
                    continue
                
                # 2단계: 2개 색상 교차 패턴 압축 검사 (예: c245 c235 c245 c235 구조 대응)
                if c_idx + 3 < len(row) and row[c_idx] == row[c_idx+2] and row[c_idx+1] == row[c_idx+3]:
                    tokens.append(f"p2_{row[c_idx]}_{row[c_idx+1]}")
                    c_idx += 4
                    continue
                
                # 3단계: 압축되지 않는 고유 단독 픽셀 보존
                tokens.append(f"p1_{row[c_idx]}")
                c_idx += 1
                
            # 행 구분자 주입
            if r < grid.shape[0] - 1:
                tokens.append("1X")
        return " ".join(tokens)

    @classmethod
    def generate_all_variants(cls, prompt, protocol_str):
        """하나의 인풋 소스로부터 단일 토큰 기반의 기하학적 좌우 반전 및 90도 회전 증강 팩 생성.
        대칭 도형은 flip/rotate해도 output이 원본과 동일해지므로, 실제로 output이
        달라지는 variant만 데이터셋에 포함시킨다."""
        variants = []
        
        grid = cls.deserialize_to_matrix(protocol_str)
        base_proto = cls.serialize_from_matrix(grid)
        variants.append({"prompt": prompt, "protocol": base_proto})
        
        try:
            flipped_grid = np.fliplr(grid)
            flipped_proto = cls.serialize_from_matrix(flipped_grid)
            if flipped_proto != base_proto:
                variants.append({"prompt": f"{prompt} flipped", "protocol": flipped_proto})
            
            rotated_grid = np.rot90(grid, 1)
            rotated_proto = cls.serialize_from_matrix(rotated_grid)
            if rotated_proto != base_proto:
                variants.append({"prompt": f"{prompt} rotated", "protocol": rotated_proto})
        except Exception:
            pass
            
        return variants

    @classmethod
    def _legacy_deserialize(cls, protocol_str, size=16):
        """구형 RLE 프로토콜 파싱 및 마이그레이션 백업 엔진"""
        tokens = protocol_str.split()
        grid = []
        current_row = []
        in_repeat = False
        repeat_count = 1
        repeat_tokens = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token == "[" and i + 1 < len(tokens) and tokens[i + 1].startswith("Re"):
                in_repeat = True
                repeat_count = int(tokens[i + 1].replace("Re", ""))
                i += 3
                if i < len(tokens) and tokens[i] == "[": i += 1
                continue
            if in_repeat and token == "]":
                for _ in range(repeat_count):
                    for r_tk in repeat_tokens:
                        current_row = cls._parse_legacy_token(r_tk, current_row, grid, size)
                in_repeat = False
                repeat_tokens = []
                i += 1
                continue
            if in_repeat:
                repeat_tokens.append(token)
            else:
                current_row = cls._parse_legacy_token(token, current_row, grid, size)
            i += 1
        if current_row:
            while len(current_row) < size: current_row.append('c255')
            grid.append(current_row[:size])
        while len(grid) < size: grid.append(['c255'] * size)
        return np.array(grid[:size], dtype=object)

    @classmethod
    def _parse_legacy_token(cls, token, current_row, grid, size):
        if token == "1X":
            while len(current_row) < size: current_row.append('c255')
            grid.append(current_row[:size])
            return []
        if "c" in token:
            parts = token.split("c")
            count = int(parts[0]) if parts[0] != "" else 1
            color = "c" + parts[1]
            for _ in range(count): current_row.append(color)
        return current_row


# ==========================================
# 3. 데이터셋 빌드 메인스트림 (Seq2Seq 대전환)
# ==========================================
if __name__ == "__main__":
    logger, log_path = setup_logger(mode="data_generator")

    logger.info("[*] 🚀 멀티모달 대칭형 패턴 압축 파이프라인 및 기하학적 증강 엔진 가동...")
    logger.info(f" └─ 생성될 데이터셋 타깃: {PixelPaths.TOTAL_DATA}\n")
    
    raw_pairs = []

    # 3:3:2 비트 컬러 명세 구조
    colors_spec = [
        ('red', 'c224'), ('blue', 'c003'), ('green', 'c028'), ('yellow', 'c252'), ('black', 'c000'),
        ('white', 'c255'), ('purple', 'c131'), ('orange', 'c248'), ('pink', 'c243'), ('gray', 'c146'),
        ('brown', 'c073'), ('sky blue', 'c155'), ('lime', 'c061'), ('mint', 'c123'), ('gold', 'c249'),
        ('silver', 'c182'), ('navy', 'c001'), ('olive', 'c072'), ('beige', 'c251'), ('magenta', 'c227')
    ]

    # [공정 1: 수식형 벡터 조합 데이터 빌드]
    for color_name, char in colors_spec:
        bg = 'c000' if char == 'c255' else 'c255'

        # 단색 본연의 데이터 매핑은 간소화 프로토콜 변환을 위해 행렬 자체를 원본 파이프라인으로 투하
        raw_pairs.append((color_name, ProtocolAugmenter.serialize_from_matrix(np.array(draw_square(16, char, False, bg_char=bg), dtype=object))))
        raw_pairs.append((color_name.capitalize(), ProtocolAugmenter.serialize_from_matrix(np.array(draw_square(16, char, False, bg_char=bg), dtype=object))))
        raw_pairs.append((f"solid {color_name} color", ProtocolAugmenter.serialize_from_matrix(np.array(draw_square(16, char, False, bg_char=bg), dtype=object))))

        for pmt in generate_advanced_prompts(color_name, "square"):
            raw_pairs.append((pmt, ProtocolAugmenter.serialize_from_matrix(np.array(draw_square(16, char, False, bg_char=bg), dtype=object))))
        for pmt in generate_advanced_prompts(color_name, "hollow square"):
            raw_pairs.append((pmt, ProtocolAugmenter.serialize_from_matrix(np.array(draw_square(16, char, True, bg_char=bg), dtype=object))))
        for pmt in generate_advanced_prompts(color_name, "triangle"):
            raw_pairs.append((pmt, ProtocolAugmenter.serialize_from_matrix(np.array(draw_triangle(16, char, False, bg_char=bg), dtype=object))))
        for pmt in generate_advanced_prompts(color_name, "hollow triangle"):
            raw_pairs.append((pmt, ProtocolAugmenter.serialize_from_matrix(np.array(draw_triangle(16, char, True, bg_char=bg), dtype=object))))
        for pmt in generate_advanced_prompts(color_name, "cross"):
            raw_pairs.append((pmt, ProtocolAugmenter.serialize_from_matrix(np.array(draw_cross(16, char, bg_char=bg), dtype=object))))
        for pmt in generate_advanced_prompts(color_name, "diamond"):
            raw_pairs.append((pmt, ProtocolAugmenter.serialize_from_matrix(np.array(draw_diamond(16, char, bg_char=bg), dtype=object))))
        for pmt in generate_advanced_prompts(color_name, "border frame"):
            raw_pairs.append((pmt, ProtocolAugmenter.serialize_from_matrix(np.array(draw_border_frame(16, char, bg_char=bg), dtype=object))))
        for pmt in generate_advanced_prompts(color_name, "horizontal stripes"):
            raw_pairs.append((pmt, ProtocolAugmenter.serialize_from_matrix(np.array(draw_stripes(16, char, vertical=False, bg_char=bg), dtype=object))))
        for pmt in generate_advanced_prompts(color_name, "checkerboard"):
            raw_pairs.append((pmt, ProtocolAugmenter.serialize_from_matrix(np.array(draw_checkerboard(16, char, bg_char=bg), dtype=object))))

    # [공정 2: 순수 도형 단독 데이터 생성]
    shape_names = ["square", "hollow square", "triangle", "hollow triangle", "cross", "diamond", "border frame", "horizontal stripes", "checkerboard"]
    for shape_name in shape_names:
        if shape_name == "horizontal stripes":
            mat = np.array(draw_stripes(16, 'c000', vertical=False, bg_char='c255'), dtype=object)
        elif shape_name == "checkerboard":
            mat = np.array(draw_checkerboard(16, 'c000', bg_char='c255'), dtype=object)
        elif shape_name == "square":
            mat = np.array(draw_square(16, 'c000', False, bg_char='c255'), dtype=object)
        elif shape_name == "hollow square":
            mat = np.array(draw_square(16, 'c000', True, bg_char='c255'), dtype=object)
        elif shape_name == "triangle":
            mat = np.array(draw_triangle(16, 'c000', False, bg_char='c255'), dtype=object)
        elif shape_name == "hollow triangle":
            mat = np.array(draw_triangle(16, 'c000', True, bg_char='c255'), dtype=object)
        elif shape_name == "cross":
            mat = np.array(draw_cross(16, 'c000', bg_char='c255'), dtype=object)
        elif shape_name == "diamond":
            mat = np.array(draw_diamond(16, 'c000', bg_char='c255'), dtype=object)
        elif shape_name == "border frame":
            mat = np.array(draw_border_frame(16, 'c000', bg_char='c255'), dtype=object)

        out_str = ProtocolAugmenter.serialize_from_matrix(mat)
        for shape_prompt in generate_shape_only_prompts(shape_name):
            raw_pairs.append((shape_prompt, out_str))

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
                    # 이미지 원본 복원 및 신규 파서 파이프라인 연결을 위한 징검다리 매핑
                    legacy_str = PixelDeserializer.deserialize_image(full_path)
                    temp_grid = ProtocolAugmenter.deserialize_to_matrix(legacy_str)
                    out_str = ProtocolAugmenter.serialize_from_matrix(temp_grid)
                    
                    norm_root = root.lower()
                    asset_prompts = []
                    refined_desc = TokenMapper.refine_filename_tokens(root, img_file)
                    tokens = refined_desc.split()

                    if "books" in norm_root:
                        color = " ".join([t for t in tokens if t in ['dark', 'light', 'gray', 'silver', 'red', 'orange', 'gold', 'yellow', 'lime', 'green', 'teal', 'cyan', 'blue', 'purple', 'pink']])
                        style_raw = tokens[-1]
                        style_key = style_raw.split('_')[-1] if '_' in style_raw else style_raw
                        style = style_key if style_key in ['basic', 'refined', 'alpha', 'beta', 'border', 'glow', 'cross', 'gold', 'enchanted', 'mythic'] else 'basic'
                        
                        asset_prompts.append(f"{color} book {style}")
                        asset_prompts.append(f"pixel art {color} book {style}")
                        
                        if style == 'basic': rich_desc = f"a book with a {color} gem"
                        elif style == 'refined': rich_desc = f"a book with a polished {color} gem"
                        elif style == 'alpha': rich_desc = f"a {color} book engraved with an ancient rune"
                        elif style == 'beta': rich_desc = f"a {color} book engraved with a mystic rune"
                        elif style == 'border': rich_desc = f"a {color} book with an ornate border"
                        elif style == 'glow': rich_desc = f"a glowing {color} magical book"
                        elif style == 'cross': rich_desc = f"a {color} book marked with an ancient cross"
                        elif style == 'gold': rich_desc = f"a royal golden {color} book"
                        elif style == 'enchanted': rich_desc = f"an enchanted {color} magical book"
                        elif style == 'mythic': rich_desc = f"a mythic {color} magical book"
                        else: rich_desc = f"a {color} book in {style} style"
                            
                        asset_prompts.append(rich_desc)
                        asset_prompts.append(f"a pixel art of {rich_desc}")

                    elif "potions" in norm_root:
                        color = tokens[0]
                        if len(tokens) >= 4:
                            # test_tube류처럼 shape 자체가 두 단어인 경우 (예: test tube)
                            shape = f"{tokens[1]} {tokens[2]}"
                            deco = tokens[3]
                        else:
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
                        raw_pairs.append((input_prompt, out_str))
                        
                    parsed_count += 1
                except Exception as e:
                    logger.error(f" └─ ⚠️ 파일 파싱 실패 ({img_file}): {e}")

    # ============================================================
    # 🚀 [최종 빌드 공정] 데이터 양방향 전개 및 증강 파이프라인 가동
    # ============================================================
    final_dataset = []
    logger.info("[*] 🔄 단일 문자 패턴 증강(Flip/Rotation) 및 대칭형 멀티태스크 구조 인코딩 시작...")

    for prompt, proto_str in raw_pairs:
        # 단일 문자 압축형 기반의 증강 수집 생성 (원본, 좌우반전, 90도 회전 순차 전개)
        variants = ProtocolAugmenter.generate_all_variants(prompt, proto_str)
        
        for var in variants:
            p_text = var["prompt"]
            p_proto = var["protocol"]
            
            # Task A: Text-to-Pixel 생성 기능 정렬 데이터 구조
            final_dataset.append({
                "input": f"<TASK_GEN> {p_text}",
                "output": p_proto
            })
            
            # Task B: Pixel-to-Text 캡셔닝 역방향 기능 정렬 데이터 구조
            final_dataset.append({
                "input": f"<TASK_DESC> {p_proto}",
                "output": p_text
            })

    # 데이터 전수 랜덤 셔플 및 타깃 파일 안전 영구 투하
    random.shuffle(final_dataset)
    dir_name = os.path.dirname(PixelPaths.TOTAL_DATA)
    if dir_name: 
        os.makedirs(dir_name, exist_ok=True)
    
    with open(PixelPaths.TOTAL_DATA, "w", encoding="utf-8") as f:
        for entry in final_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"\n[*] 완료: 원본 데이터가 고밀도 압축 및 양방향 최소 토큰 단위로 리팩토링되었습니다.")
    logger.info(f" └─ 📊 최종 통합 멀티모달 데이터셋 라인 수: {len(final_dataset)} 개")
    logger.info(f"[✓] 데이터셋 빌더 세부 로그 저장 완료: {log_path}")