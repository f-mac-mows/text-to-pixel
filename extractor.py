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

def extract_consumables_v4(image_path="consumables.png"):
    output_dir = os.path.expanduser("~/Assets/Sprites/Consumables")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(image_path):
        print(f"❌ 파일을 찾을 수 없습니다: {image_path}")
        return

    img = Image.open(image_path).convert('RGBA')
    grid_size = 16
    half_width = 22 * grid_size  # 352 픽셀
    
    # 1. 좌우 분할
    left_img = img.crop((0, 0, half_width, img.height))
    right_img = img.crop((half_width, 0, img.width, img.height))
    
    # 💡 [핵심] 가로 22개 열에 직관적인 내용물/상태 자연어 키워드 부여
    liquid_types = [
        "empty", "water", "beer", "honey", "green_juice", 
        "mana_blue", "poison_purple", "wine_red", "milk", "oil_yellow", 
        "dark_sludge", "lava_orange", "pink_potion", "white_froth", "cyan_elixir",
        "gold_liquid", "blood_red", "mint_tea", "shadow_essence", "special_stardust",
        "royal_brew", "mythic_glow"
    ]
    
    success_count = 0

    # ============================================================
    # 📦 [왼쪽 구역] 그룹 순회
    # ============================================================
    left_groups = [
        {"count": 4, "name": "wooden_mug"},
        {"count": 2, "name": "bamboo_cup"},
        {"count": 3, "name": "leather_pouch"},
        {"count": 2, "name": "metal_cup"},
        {"count": 2, "name": "iron_tankard"},
        {"count": 4, "name": "ceramic_cup"}
    ]
    
    current_row = 0
    for g_idx, group in enumerate(left_groups):
        for sub_row in range(group["count"]):
            # r1, r2 대신에 단계를 뜻하는 tier1, tier2 나 색상 구분을 주면 프롬프트 구성이 더 쉬워집니다.
            row_tag = f"t{sub_row+1}" 
            
            for c_idx in range(22):
                left = c_idx * grid_size
                top = current_row * grid_size
                
                tile = left_img.crop((left, top, left + grid_size, top + grid_size))
                
                # 🎯 출력 예시: wooden_mug_t1_mana_blue.png
                # 모델 input 프롬프트 예시: "a wooden mug with tier 1 design containing mana blue liquid"
                filename = f"{group['name']}_{row_tag}_{liquid_types[c_idx]}.png"
                
                tile.save(os.path.join(output_dir, filename))
                success_count += 1
            current_row += 1

    # ============================================================
    # 📦 [오른쪽 구역] 그룹 순회
    # ============================================================
    right_groups = [
        {"count": 2, "name": "modern_mug"},
        {"count": 3, "name": "copper_pitcher"},
        {"count": 3, "name": "silver_goblet"},
        {"count": 4, "name": "golden_chalice"}
    ]
    
    current_row = 0
    for g_idx, group in enumerate(right_groups):
        for sub_row in range(group["count"]):
            row_tag = f"t{sub_row+1}"
            
            for c_idx in range(22):
                left = c_idx * grid_size
                top = current_row * grid_size
                
                if top + grid_size <= right_img.height:
                    tile = right_img.crop((left, top, left + grid_size, top + grid_size))
                    
                    # 🎯 출력 예시: golden_chalice_t3_wine_red.png
                    filename = f"{group['name']}_{row_tag}_{liquid_types[c_idx]}.png"
                    
                    tile.save(os.path.join(output_dir, filename))
                    success_count += 1
            current_row += 1

    print(f"✅ {image_path} 자연어 라벨링 분할 완료! 총 {success_count}개 파일 저장됨.")

if __name__ == "__main__":
    # # 1. 과일 & 야채 데이터셋 정의
    # fruits_flat = [
    #     "pineapple", "watermelon", "banana", "orange fruit",
    #     "peach", "cherry small", "apple", "grapes",
    #     "lemon", "avocado", "dragon fruit", "pear",
    #     "coconut", "kiwi", "cherries", "tomato big",
    #     "blueberries", "meat chunk", "cauliflower", "cabbage",
    #     "onion purple", "strawberries", "paprika yellow", "paprika green",
    #     "pumpkin", "potatoes", "tomato small", "eggplant",
    #     "corn", "ginger", "carrot", "garlic"
    # ]
    # fruit_matrix = [fruits_flat[i:i+4] for i in range(0, len(fruits_flat), 4)]
    
    # # 💡 과일 시트 실행: 가로 세로 8픽셀씩 밀어서 시작하도록 설정
    # slice_and_label_v2(
    #     image_path="vegetable &fruit（.png", 
    #     grid_w=16, 
    #     grid_h=16, 
    #     name_matrix=fruit_matrix, 
    #     subfolder_name="Fruits",
    #     offset_x=6,  # 16px의 1/2
    #     offset_y=9   # 16px의 1/2
    # )

    # 2. 포션 데이터셋 정의 (포션은 원래대로 offset=0)
    # ============================================================
    # 🧪 포션 데이터셋 정의 (15행 × 21열 정밀 Column-Major 매핑)
    # ============================================================
    # 💡 이미지에 존재하는 실제 22가지 색상/재질 톤 (열 순서)
    potion_colors = [
        "gray", "silver", "dark_gray", 
        "red", "caramel", "orange", 
        "yellow", "drak_green", "green",
        "teal", "mint", "sky", "blue", 
        "purple", "violet", "orchid", 
        "pink", "hot_pink", "bright_red", "brown", "white"
    ]
    
    # 💡 이미지에 존재하는 실제 15가지 포션 모양/단계 외형 (행 순서)
    # slender, standard, large, flask 등에서 파생된 다양한 티어/디자인 단계
    potion_types  = [
        "slender_t1", "slender_t2", "slender_t3", "slender_t4",
        "standard_t1", "standard_t2", "standard_t3", "standard_t4",
        "flask_t1", "flask_t2", "flask_t3", "flask_t4",
        "reagent_bottle", "test_tube_t1", "test_tube_t2"
    ]
    
    # 1. 15행 × 22열 빈 Matrix 초기화
    potion_matrix = [["" for _ in range(22)] for _ in range(15)]
    
    # 2. Column-Major 방식으로 데이터 매핑 (세로축으로 먼저 훑으며 이름 맵핑)
    for c_idx, color in enumerate(potion_colors):
        for r_idx, p_type in enumerate(potion_types):
            # 행렬의 [r_idx][c_idx] 위치에 명명 (예: "gray_slender_t1")
            potion_matrix[r_idx][c_idx] = f"{color} {p_type}"
            
    # 3. 크롭 및 라벨링 실행 (오프셋 없이 원점 기준)
    slice_and_label_v2(
        image_path="potions.png", 
        grid_w=16, 
        grid_h=16, 
        name_matrix=potion_matrix, 
        subfolder_name="Potions",
        offset_x=0,
        offset_y=0
    )

    # ============================================================
    # 3. 책 (Books) 데이터셋 추출 (14열 x 10행)
    # ============================================================
    # book_colors = [
    #     "dark gray", "light gray", "silver", "red", "orange", 
    #     "bronze", "gold", "lime", "green", "teal", 
    #     "cyan", "blue", "purple", "pink"
    # ]
    
    # book_matrix = []
    # # 총 10개의 행에 대해 색상별 책 이름 지정
    # for row_idx in range(10):
    #     row = [f"{color} book type{row_idx+1}" for color in book_colors]
    #     book_matrix.append(row)
        
    # slice_and_label_v2(
    #     image_path="books.png", 
    #     grid_w=16, 
    #     grid_h=16, 
    #     name_matrix=book_matrix, 
    #     subfolder_name="Books",
    #     offset_x=0,
    #     offset_y=0
    # )

    # # ============================================================
    # # 4. 상자 (Chests) 데이터셋 추출 (8열 x 6행)
    # # ============================================================
    # chest_matrix = [
    #     ["small wooden chest", "normal wooden chest", "large wooden chest", "small bronze chest", "normal bronze chest", "large bronze chest", "silver lock chest", "gold lock chest"],
    #     ["small silver chest", "normal silver chest", "large silver chest", "small gold chest", "normal gold chest", "large gold chest", "red gem chest", "yellow gem chest"],
    #     ["small steel chest", "normal steel chest", "large steel chest", "small shiny gold chest", "normal shiny gold chest", "large shiny gold chest", "cyan gem chest", "blue gem chest"],
    #     ["small white chest", "normal white chest", "large white chest", "small yellow chest", "normal yellow chest", "large yellow chest", "orange gem chest", "pink gem chest"],
    #     ["small dark chest", "normal dark chest", "large dark chest", "small orange chest", "normal orange chest", "large orange chest", "gray skull chest", "gold royal chest"],
    #     ["small skeletal chest", "normal skeletal chest", "large skeletal chest", "small brown chest", "normal brown chest", "large brown chest", "magical red chest", "magical pink chest"]
    # ]
        
    # slice_and_label_v2(
    #     image_path="chests.png", 
    #     grid_w=16, 
    #     grid_h=16, 
    #     name_matrix=chest_matrix, 
    #     subfolder_name="Chests",
    #     offset_x=0,
    #     offset_y=0
    # )

    # ============================================================
    # 5. 소비템 (Consumables) 데이터셋 추출 (축 방향 보정본)
    # ============================================================
    # 💡 세로(행)로 내려갈 때 바뀌는 색상/재질 리스트 (총 15행)
    #extract_consumables_v4()


    # ============================================================
    # 6. UI 및 키보드/게임패드 (UI & Inputs) 전체 데이터셋 정의 (34열 x 24행)
    # ============================================================    
    # base_ui_matrix = [
    #     # Row 0~4: 키보드 기능 및 일반 텍스트 문자열 (Keyboard Row 1 ~ Row 5)
    #     ["key esc", "key f1", "key f2", "key f3", "key f4", "key f5", "key f6", "key f7", "key f8", "key f9", "key f10", "key f11", "key f12", "key printscreen", "key scrolllock", "key pause", "key tilde", "key 1", "key 2", "key 3", "key 4", "key 5", "key 6", "key 7", "key 8", "key 9", "key 0", "key minus", "key equals", "key backspace", "key insert", "key home", "key pageup", "key numlock"],
    #     ["key tab", "key q", "key w", "key e", "key r", "key t", "key y", "key u", "key i", "key o", "key p", "key bracket open", "key bracket close", "key backslash", "key delete", "key end", "key pagedown", "key numpad 7", "key numpad 8", "key numpad 9", "key numpad plus", "key caps lock", "key a", "key s", "key d", "key f", "key g", "key h", "key j", "key k", "key l", "key semicolon", "key quote", "key enter"],
    #     ["key left shift", "key z", "key x", "key c", "key v", "key b", "key n", "key m", "key comma", "key period", "key slash", "key right shift", "key numpad 4", "key numpad 5", "key numpad 6", "key left ctrl", "key left win", "key left alt", "key space", "key right alt", "key right win", "key menu", "key right ctrl", "key numpad 1", "key numpad 2", "key numpad 3", "key numpad enter", "key arrow up", "key arrow down", "key arrow left", "key arrow right", "key numpad 0", "key numpad dot", "key numpad minus"],
    #     ["key space long", "key backspace small", "key enter small", "key shift small", "key ctrl small", "key alt small", "key win small", "key caps small", "key tab small", "key esc small", "key up small", "key down small", "key left small", "key right small", "key numpad slash", "key numpad asterisk", "key home small", "key end small", "key pgup small", "key pgdn small", "key del small", "key ins small", "key menu small", "key fn small", "key prtsc small", "key scrlk small", "key pause small", "key tilde small", "key minus small", "key plus small", "key equal small", "key bracket_l small", "key bracket_r small", "key slash small"],
    #     ["key backslash small", "key semi small", "key quote small", "key comma small", "key dot small", "key clear", "key select", "key submit", "key cancel", "key help", "key back", "key forward", "key refresh", "key search", "key favorite", "key home web", "key mail", "key media play", "key media pause", "key media stop", "key media next", "key media prev", "key volume mute", "key volume up", "key volume down", "key brightness up", "key brightness down", "key power", "key sleep", "key wake", "key calc", "key my computer", "key app1", "key app2"],

    #     # Row 5~7: 마우스 아이콘 및 커서 무브먼트 기믹 (Mouse Icons & System UI)
    #     ["mouse outline", "mouse left click", "mouse right click", "mouse middle click", "mouse scroll up", "mouse scroll down", "mouse move left", "mouse move right", "mouse move up", "mouse move down", "mouse scroll tilt left", "mouse scroll tilt right", "mouse left hold", "mouse right hold", "mouse double click", "mouse trackball", "mouse pointer generic", "mouse pointer hand", "mouse pointer crosshair", "mouse pointer text", "mouse pointer help", "mouse pointer forbidden", "mouse pointer wait", "mouse pointer resize v", "mouse pointer resize h", "mouse pointer resize d1", "mouse pointer resize d2", "mouse pointer move", "mouse pointer pen", "mouse pointer zoom in", "mouse pointer zoom out", "mouse pointer grab", "mouse pointer grabbing", "mouse pointer link"],
    #     ["ui arrow up", "ui arrow down", "ui arrow left", "ui arrow right", "ui arrow up double", "ui arrow down double", "ui arrow left double", "ui arrow right double", "ui arrow corner ur", "ui arrow corner ul", "ui arrow corner dr", "ui arrow corner dl", "ui arrow rotate cw", "ui arrow rotate ccw", "ui arrow expand", "ui arrow shrink", "ui caret up", "ui caret down", "ui caret left", "ui caret right", "ui plus", "ui minus", "ui check", "ui cross", "ui question", "ui exclamation", "ui info", "ui warning", "ui settings", "ui filter", "ui sort", "ui menu hamburger", "ui menu dots v", "ui menu dots h"],
    #     ["ui heart solid", "ui heart outline", "ui star solid", "ui star outline", "ui lock locked", "ui lock unlocked", "ui eye visible", "ui eye hidden", "ui trash bin", "ui edit pen", "ui save disk", "ui load folder", "ui share", "ui download", "ui upload", "ui cloud", "ui wifi solid", "ui wifi slash", "ui bluetooth", "ui battery full", "ui battery half", "ui battery empty", "ui battery charging", "ui sound high", "ui sound low", "ui sound mute", "ui mic active", "ui mic muted", "ui camera video", "ui camera photo", "ui shopping cart", "ui bag", "ui trophy", "ui medal"],

    #     # Row 8~11: 엑스박스 스타일 컨트롤러 (Xbox Series / One / 360)
    #     ["xbox button a", "xbox button b", "xbox button x", "xbox button y", "xbox dpad up", "xbox dpad down", "xbox dpad left", "xbox dpad right", "xbox dpad center", "xbox stick left", "xbox stick right", "xbox stick left click", "xbox stick right click", "xbox bumper lb", "xbox bumper rb", "xbox trigger lt", "xbox trigger rt", "xbox button view", "xbox button menu", "xbox button nexus", "xbox button share", "xbox stick left up", "xbox stick left down", "xbox stick left left", "xbox stick left right", "xbox stick right up", "xbox stick right down", "xbox stick right left", "xbox stick right right", "xbox paddle p1", "xbox paddle p2", "xbox paddle p3", "xbox paddle p4", "xbox headset connector"],
    #     ["xbox a outline", "xbox b outline", "xbox x outline", "xbox y outline", "xbox dpad up solid", "xbox dpad down solid", "xbox dpad left solid", "xbox dpad right solid", "xbox dpad all", "xbox stick l rotation", "xbox stick r rotation", "xbox lb outline", "xbox rb outline", "xbox lt outline", "xbox rt outline", "xbox view outline", "xbox menu outline", "xbox share outline", "xbox battery 3", "xbox battery 2", "xbox battery 1", "xbox battery 0", "xbox l stick push u", "xbox l stick push d", "xbox l stick push l", "xbox l stick push r", "xbox r stick push u", "xbox r stick push d", "xbox r stick push l", "xbox r stick push r", "xbox button menu alt", "xbox button view alt", "xbox custom light", "xbox profile switch"],
    #     ["xbox neon a", "xbox neon b", "xbox neon x", "xbox neon y", "xbox dpad up green", "xbox dpad down red", "xbox dpad left blue", "xbox dpad right yellow", "xbox guide light", "xbox series dpad top", "xbox series dpad bottom", "xbox series dpad left", "xbox series dpad right", "xbox series dpad center", "xbox elite stick tall", "xbox elite stick dome", "xbox back button", "xbox start button", "xbox sync button", "xbox accessory slot", "xbox 360 ring 1", "xbox 360 ring 2", "xbox 360 ring 3", "xbox 360 ring 4", "xbox elite lock", "xbox elite profile 1", "xbox elite profile 2", "xbox elite profile 3", "xbox thumbstick click", "xbox back arrow", "xbox forward arrow", "xbox play symbol", "xbox pause symbol", "xbox stop symbol"],
    #     ["xbox button text a", "xbox button text b", "xbox button text x", "xbox button text y", "xbox text lb", "xbox text rb", "xbox text lt", "xbox text rt", "xbox text lsb", "xbox text rsb", "xbox text back", "xbox text start", "xbox text view", "xbox text menu", "xbox text share", "xbox text dpad", "xbox text lstick", "xbox text rstick", "xbox text ls", "xbox text rs", "xbox text guide", "xbox text sync", "xbox text home", "xbox text options", "xbox text select", "xbox text mode", "xbox text turbo", "xbox text clear", "xbox text auto", "xbox text macro", "xbox text fn", "xbox text reset", "xbox text setup", "xbox text config"],

    #     # Row 12~15: 플레이스테이션 스타일 컨트롤러 (PS5 / PS4 / PS3)
    #     ["ps button cross", "ps button circle", "ps button square", "ps button triangle", "ps dpad up", "ps dpad down", "ps dpad left", "ps dpad right", "ps dpad center", "ps stick l3", "ps stick r3", "ps stick l3 click", "ps stick r3 click", "ps bumper l1", "ps bumper r1", "ps trigger l2", "ps trigger r2", "ps button share", "ps button options", "ps button ps guide", "ps touchpad click", "ps stick l3 up", "ps stick l3 down", "ps stick l3 left", "ps stick l3 right", "ps stick r3 up", "ps stick r3 down", "ps stick r3 left", "ps stick r3 right", "ps paddle l1", "ps paddle r1", "ps paddle l2", "ps paddle r2", "ps mute indicator"],
    #     ["ps cross outline", "ps circle outline", "ps square outline", "ps triangle outline", "ps dpad up solid", "ps dpad down solid", "ps dpad left solid", "ps dpad right solid", "ps dpad all", "ps stick l3 rotation", "ps stick r3 rotation", "ps l1 outline", "ps r1 outline", "ps l2 outline", "ps r2 outline", "ps share outline", "ps options outline", "ps touchpad swipe u", "ps touchpad swipe d", "ps touchpad swipe l", "ps touchpad swipe r", "ps l3 push u", "ps l3 push d", "ps l3 push l", "ps l3 push r", "ps r3 push u", "ps r3 push d", "ps r3 push l", "ps r3 push r", "ps button select", "ps button start", "ps button analog", "ps indicator light 1", "ps indicator light 2"],
    #     ["ps neon cross", "ps neon circle", "ps neon square", "ps neon triangle", "ps dpad blue up", "ps dpad red down", "ps dpad pink left", "ps dpad green right", "ps vita dpad up", "ps vita dpad down", "ps vita dpad left", "ps vita dpad right", "ps vita dpad center", "ps edge fn left", "ps edge fn right", "ps edge profile", "ps move button", "ps move trigger", "ps vr button", "ps portal link", "ps3 controller num 1", "ps3 controller num 2", "ps3 controller num 3", "ps3 controller num 4", "ps touchpad tap", "ps touchpad hold", "ps touchpad double tap", "ps touchpad multi pinch", "ps thumb stick click", "ps lightbar red", "ps lightbar blue", "ps lightbar green", "ps lightbar pink", "ps speaker sound"],
    #     ["ps button text cross", "ps button text circle", "ps button text square", "ps button text triangle", "ps text l1", "ps text r1", "ps text l2", "ps text r2", "ps text l3", "ps text r3", "ps text select", "ps text start", "ps text share", "ps text options", "ps text create", "ps text ps", "ps text touchpad", "ps text dpad", "ps text lstick", "ps text rstick", "ps text ls", "ps text rs", "ps text analog", "ps text mute", "ps text function", "ps text back", "ps text next", "ps text enter", "ps text exit", "ps text home", "ps text menu", "ps text power", "ps text reset", "ps text sync"],

    #     # Row 16~19: 닌텐도 스위치 및 에뮬레이터 인터페이스 (Nintendo Switch / GameCube)
    #     ["switch button b", "switch button a", "switch button y", "switch button x", "switch dpad up", "switch dpad down", "switch dpad left", "switch dpad right", "switch dpad center", "switch stick left", "switch stick right", "switch stick left click", "switch stick right click", "switch bumper l", "switch bumper r", "switch trigger zl", "switch trigger zr", "switch button minus", "switch button plus", "switch button home", "switch button capture", "switch stick left up", "switch stick left down", "switch stick left left", "switch stick left right", "switch stick right up", "switch stick right down", "switch stick right left", "switch stick right right", "switch joycon l", "switch joycon r", "switch joycon l horizontal", "switch joycon r horizontal", "switch rail attachment"],
    #     ["switch b outline", "switch a outline", "switch y outline", "switch x outline", "switch dpad up solid", "switch dpad down solid", "switch dpad left solid", "switch dpad right solid", "switch dpad all", "switch stick l rotation", "switch stick r rotation", "switch l outline", "switch r outline", "switch zl outline", "switch zr outline", "switch minus outline", "switch plus outline", "switch home outline", "switch capture outline", "switch joycon sl l", "switch joycon sr l", "switch joycon sl r", "switch joycon sr r", "switch l stick push u", "switch l stick push d", "switch l stick push l", "switch l stick push r", "switch r stick push u", "switch r stick push d", "switch r stick push l", "switch r stick push r", "switch pro icon", "switch handheld icon", "switch dogear grip"],
    #     ["gc button a", "gc button b", "gc button x", "gc button y", "gc button start", "gc stick main", "gc stick c", "gc dpad up", "gc dpad down", "gc dpad left", "gc dpad right", "gc bumper l", "gc bumper r", "gc trigger z", "wii button 1", "wii button 2", "wii button a", "wii button b", "wii button plus", "wii button minus", "wii button home", "wii dpad center", "wii nunchuk c", "wii nunchuk z", "wii nunchuk stick", "wii classic l", "wii classic r", "wii classic zl", "wii classic zr", "nes dpad", "nes button a", "nes button b", "nes select", "nes start"],
    #     ["switch button text b", "switch button text a", "switch button text y", "switch button text x", "switch text l", "switch text r", "switch text zl", "switch text zr", "switch text lsb", "switch text rsb", "switch text minus", "switch text plus", "switch text home", "switch text capture", "switch text sl", "switch text sr", "switch text dpad", "switch text lstick", "switch text rstick", "switch text ls", "switch text rs", "switch text pro", "switch text gc a", "switch text gc b", "switch text gc x", "switch text gc y", "switch text gc z", "switch text wii 1", "switch text wii 2", "switch text nunchuk", "switch text classic", "switch text nes a", "switch text nes b", "switch text select"],

    #     # Row 20~23: 모바일 터치 제스처, 아케이드 스틱 및 스페셜 플레어 아이콘 (Touch, Arcade & Flairs)
    #     ["touch tap", "touch double tap", "touch hold", "touch swipe up", "touch swipe down", "touch swipe left", "touch swipe right", "touch swipe up long", "touch swipe down long", "touch swipe left long", "touch swipe right long", "touch drag generic", "touch pinch in", "touch pinch out", "touch rotate cw", "touch rotate ccw", "touch two fingers tap", "touch two fingers hold", "touch three fingers tap", "touch multi drag", "touch palm press", "touch edge swipe l", "touch edge swipe r", "touch edge swipe u", "touch edge swipe d", "touch fingerprint lock", "touch biometric scan", "touch stylus press", "touch stylus hover", "touch pressure deep", "touch tilt canvas", "touch shake device", "touch rotate device", "touch flick gesture"],
    #     ["arcade joystick center", "arcade joystick up", "arcade joystick down", "arcade joystick left", "arcade joystick right", "arcade joystick ul", "arcade joystick ur", "arcade joystick dl", "arcade joystick dr", "arcade joystick cw", "arcade joystick ccw", "arcade button 1", "arcade button 2", "arcade button 3", "arcade button 4", "arcade button 5", "arcade button 6", "arcade button 7", "arcade button 8", "arcade coin slot", "arcade start 1p", "arcade start 2p", "arcade trackball move", "arcade spinner turn", "arcade lightgun trigger", "arcade steering wheel", "arcade pedal gas", "arcade pedal brake", "arcade slider h", "arcade slider v", "arcade toggle switch", "arcade key lock", "arcade dip switch", "arcade service button"],
    #     ["flair forbidden sign", "flair check mark", "flair cross mark", "flair exclamation mark", "flair question mark", "flair alert info", "flair warning triangle", "flair lock container", "flair unlock container", "flair eye show", "flair eye hide", "flair star favorite", "flair heart like", "flair skull death", "flair clock time", "flair hourglass wait", "flair infinite loop", "flair refresh sync", "flair gear option", "flair wrench tool", "flair hammer craft", "flair key credential", "flair shield protect", "flair sword attack", "flair boot speed", "flair potion health", "flair scroll magic", "flair coin currency", "flair gem premium", "flair gift reward", "flair mail notification", "flair chat message", "flair Mic audio", "flair speaker sound"],
    #     ["flair arrow up", "flair arrow down", "flair arrow left", "flair arrow right", "flair arrow up left", "flair arrow up right", "flair arrow down left", "flair arrow down right", "flair arrow loop cw", "flair arrow loop ccw", "flair plus add", "flair minus sub", "flair divide math", "flair multiply math", "flair equal result", "flair percent stat", "flair dollar cost", "flair euro cost", "flair question small", "flair exclamation small", "flair dot red", "flair dot green", "flair dot blue", "flair dot yellow", "flair circle red", "flair circle green", "flair circle blue", "flair flag start", "flair flag finish", "flair target aim", "flair crosshair shoot", "flair fire damage", "flair ice freeze", "flair lightning shock"]
    # ]

    # # 두 가지 테마의 파일과 앞에 붙일 색상 접두사 설정
    # ui_themes = [
    #     {"file": "tilemap_white_packed.png", "color_prefix": "white"},
    #     {"file": "tilemap_black_packed.png", "color_prefix": "black"}
    # ]

    # for theme in ui_themes:
    #     # 1. 색상 접두사가 결합된 새로운 이름 매트릭스 생성
    #     themed_matrix = []
    #     for row in base_ui_matrix:
    #         themed_row = []
    #         for label in row:
    #             if label.strip() == "":
    #                 themed_row.append("")  # 빈 칸 유지
    #             else:
    #                 # 예: "white key esc", "black mouse left click"
    #                 themed_row.append(f"{theme['color_prefix']} {label}")
    #         themed_matrix.append(themed_row)
            
    #     # 2. 크롭 및 라벨링 자동 실행
    #     slice_and_label_v2(
    #         image_path=theme["file"], 
    #         grid_w=16, 
    #         grid_h=16, 
    #         name_matrix=themed_matrix, 
    #         subfolder_name="UI_Inputs",  
    #         offset_x=0,
    #         offset_y=0
    #     )