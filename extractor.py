import os
from PIL import Image

def slice_and_label_v2(image_path, grid_w, grid_h, name_matrix, subfolder_name, offset_x=0, offset_y=0):
    """
    정형화된 시트를 격자 크기대로 자르되, 시작 오프셋(offset_x, offset_y)을 반영하여 저장합니다.
    """
    output_dir = os.path.expanduser(f"~/Assets/Sprites/{subfolder_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(image_path):
        print(f"❌ 파일을 찾을 수 없습니다: {image_path}")
        return

    img = Image.open(image_path).convert('RGBA')
    success_count = 0
    
    for r_idx, row in enumerate(name_matrix):
        for c_idx, label in enumerate(row):
            if not label or label.strip() == "":
                continue
                
            # 💡 시작 오프셋을 더해서 크롭 영역을 한 칸씩 이동시킵니다.
            left = (c_idx * grid_w) + offset_x
            top = (r_idx * grid_h) + offset_y
            right = left + grid_w
            bottom = top + grid_h
            
            # 이미지 경계 검사
            if right <= img.width and bottom <= img.height:
                tile = img.crop((left, top, right, bottom))
                clean_filename = label.strip().replace(' ', '_') + ".png"
                tile.save(os.path.join(output_dir, clean_filename))
                success_count += 1
            
    print(f"✅ {image_path} 보정 추출 완료! -> {success_count}개의 파일이 '{subfolder_name}' 폴더에 저장되었습니다.")


if __name__ == "__main__":
    # 1. 과일 & 야채 데이터셋 정의
    fruits_flat = [
        "pineapple", "watermelon", "banana", "orange fruit",
        "peach", "cherry small", "apple", "grapes",
        "lemon", "avocado", "dragon fruit", "pear",
        "coconut", "kiwi", "cherries", "tomato big",
        "blueberries", "meat chunk", "cauliflower", "cabbage",
        "onion purple", "strawberries", "paprika yellow", "paprika green",
        "pumpkin", "potatoes", "tomato small", "eggplant",
        "corn", "ginger", "carrot", "garlic"
    ]
    fruit_matrix = [fruits_flat[i:i+4] for i in range(0, len(fruits_flat), 4)]
    
    # 💡 과일 시트 실행: 가로 세로 8픽셀씩 밀어서 시작하도록 설정
    slice_and_label_v2(
        image_path="vegetable &fruit（.png", 
        grid_w=16, 
        grid_h=16, 
        name_matrix=fruit_matrix, 
        subfolder_name="Fruits",
        offset_x=6,  # 16px의 1/2
        offset_y=9   # 16px의 1/2
    )

    # 2. 포션 데이터셋 정의 (포션은 원래대로 offset=0)
    potion_colors = ["gray", "silver", "red", "orange", "yellow", "lime", "green", "teal", "blue", "indigo", "purple", "pink", "brown", "white"]
    potion_types  = ["slender bottle", "standard potion", "large potion", "flask", "brewed potion", "reagent", "test tube"]
    
    potion_matrix = []
    for p_type in potion_types:
        row = [f"{color} {p_type}" for color in potion_colors]
        potion_matrix.append(row)
        
    slice_and_label_v2(
        image_path="potions.png", 
        grid_w=16, 
        grid_h=16, 
        name_matrix=potion_matrix, 
        subfolder_name="Potions",
        offset_x=0,
        offset_y=0
    )