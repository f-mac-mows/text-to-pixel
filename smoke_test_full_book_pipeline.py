# smoke_test_full_book_pipeline.py
from pixel_data_generator import TokenMapper

# refine_filename_tokens까지 거친 뒤, 실제 생성기 코드와 똑같은 로직으로 style을 뽑아본다
for i in range(1, 11):
    filename = f"red_book_type{i}.png"
    refined_desc = TokenMapper.refine_filename_tokens("Books", filename)
    tokens = refined_desc.split()
    style_raw = tokens[-1]
    style_key = style_raw.split('_')[-1] if '_' in style_raw else style_raw
    style = style_key if style_key in ['basic', 'refined', 'alpha', 'beta', 'border', 'glow', 'cross', 'gold', 'enchanted', 'mythic'] else 'basic'
    print(f"type{i}: style_raw='{style_raw}' -> style='{style}'")