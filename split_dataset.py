import json
from sklearn.model_selection import train_test_split
from pixel_config import PixelPaths

def split_pixel_dataset_3way():
    print(f"[*] 📂 작업 시작 파이프라인 명세:")
    print(f" ├─ 원본 로드 소스: {PixelPaths.TOTAL_DATA}")
    print(f" └─ 대상 타깃 파일: {PixelPaths.TRAIN_DATA} / {PixelPaths.VAL_DATA} / {PixelPaths.TEST_DATA}\n")

    # 1. 원본 데이터 로드
    dataset = []
    try:
        with open(PixelPaths.TOTAL_DATA, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line.strip()))
    except FileNotFoundError:
        print(f"[❌ 오류] {PixelPaths.TOTAL_DATA} 파일이 현재 경로에 존재하지 않습니다. 버전을 다시 확인해주세요.")
        return
                
    print(f"[*] 원본 데이터 로드 완료: 총 {len(dataset)}개")

    # 2. 첫 번째 분리: 전체의 10%를 최종 Test 데이터로 분리
    remaining_data, test_data = train_test_split(
        dataset, 
        test_size=0.1, 
        random_state=42, 
        shuffle=True
    )

    # 3. 두 번째 분리: 남은 데이터 중 11.11%를 Validation으로 분리 (정확한 8:1:1 매칭)
    train_data, val_data = train_test_split(
        remaining_data, 
        test_size=0.1111, 
        random_state=42, 
        shuffle=True
    )

    # 4. 자동 생성된 파일명으로 데이터 저장
    with open(PixelPaths.TRAIN_DATA, "w", encoding="utf-8") as f:
        for entry in train_data: f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(PixelPaths.VAL_DATA, "w", encoding="utf-8") as f:
        for entry in val_data: f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(PixelPaths.TEST_DATA, "w", encoding="utf-8") as f:
        for entry in test_data: f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n[*] 🚀 버전 반영 3비율 데이터셋 분리 성공! (8:1:1)")
    print(f" ├─ 학습 (Train Set - 80%)      : {len(train_data)}개 -> {PixelPaths.TRAIN_DATA}")
    print(f" ├─ 검증 (Validation Set - 10%) : {len(val_data)}개 -> {PixelPaths.VAL_DATA}")
    print(f" └─ 시험 (Test Set - 10%)       : {len(test_data)}개 -> {PixelPaths.TEST_DATA}")


if __name__ == "__main__":
    split_pixel_dataset_3way()