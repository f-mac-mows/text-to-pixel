from pixel_data_generator import TokenMapper

test_files = [
    "red_book_type1.png", "red_book_type2.png", "red_book_type3.png",
    "red_book_type4.png", "red_book_type5.png", "red_book_type6.png",
    "red_book_type7.png", "red_book_type8.png", "red_book_type9.png",
    "red_book_type10.png",
]

for f in test_files:
    refined = TokenMapper.refine_filename_tokens("Books", f)
    print(f"{f}  ->  refined='{refined}'")