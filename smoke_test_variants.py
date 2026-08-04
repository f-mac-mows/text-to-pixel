from pixel_data_generator import ProtocolAugmenter, draw_square, draw_cross, draw_checkerboard, draw_triangle
import numpy as np

# 대칭 도형 (variant 안 늘어나야 정상)
square_proto = ProtocolAugmenter.serialize_from_matrix(np.array(draw_square(16, 'c000', False, bg_char='c255'), dtype=object))
cross_proto = ProtocolAugmenter.serialize_from_matrix(np.array(draw_cross(16, 'c000', bg_char='c255'), dtype=object))
checker_proto = ProtocolAugmenter.serialize_from_matrix(np.array(draw_checkerboard(16, 'c000', bg_char='c255'), dtype=object))

# 비대칭 도형 (variant 3개 다 나와야 정상 - 삼각형은 좌우/회전에 비대칭)
triangle_proto = ProtocolAugmenter.serialize_from_matrix(np.array(draw_triangle(16, 'c000', False, bg_char='c255'), dtype=object))

for name, proto in [("square(대칭)", square_proto), ("cross(대칭)", cross_proto),
                     ("checkerboard(대칭)", checker_proto), ("triangle(비대칭)", triangle_proto)]:
    variants = ProtocolAugmenter.generate_all_variants("test", proto)
    print(f"{name}: variant 개수 = {len(variants)}  (프롬프트: {[v['prompt'] for v in variants]})")