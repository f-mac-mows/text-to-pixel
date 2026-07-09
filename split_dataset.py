import sys
import json
from sklearn.model_selection import train_test_split
from pixel_config import PixelPaths
# 💡 격리형 세부 로그 관리를 위한 모듈 임포트
from pixel_logger import setup_logger

def split_pixel_dataset_3way():
    # 💡 데이터 분할 전용 격리 로거 초기화 (logs/dataset_splitter/ 폴더 하위)
    logger, log_path = setup_logger(mode="dataset_splitter")

    logger.info(f"[*] 📂 작업 시작 파이프라인 명세:")
    logger.info(f" ├─ 원본 로드 소스: {PixelPaths.TOTAL_DATA}")
    logger.info(f" └─ 대상 타깃 파일: {PixelPaths.TRAIN_DATA} / {PixelPaths.VAL_DATA} / {PixelPaths.TEST_DATA}\n")

    # 1. 원본 데이터 로드
    dataset = []
    try:
        with open(PixelPaths.TOTAL_DATA, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line.strip()))
    except FileNotFoundError:
        logger.error(f"[❌ 오류] {PixelPaths.TOTAL_DATA} 파일이 현재 경로에 존재하지 않습니다. 버전을 다시 확인해주세요.")
        # 💡 단순 return 대신 에러 코드를 반환해 부모 파이프라인의 폭주를 방지합니다.
        sys.exit(1)
                
    logger.info(f"[*] 원본 데이터 로드 완료: 총 {len(dataset)}개")

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

    logger.info(f"\n[*] 🚀 버전 반영 3비율 데이터셋 분리 성공! (8:1:1)")
    logger.info(f" ├─ 학습 (Train Set - 80%)      : {len(train_data)}개 -> {PixelPaths.TRAIN_DATA}")
    logger.info(f" ├─ 검증 (Validation Set - 10%) : {len(val_data)}개 -> {PixelPaths.VAL_DATA}")
    logger.info(f" └─ 시험 (Test Set - 10%)       : {len(test_data)}개 -> {PixelPaths.TEST_DATA}")
    logger.info(f"[✓] 데이터셋 분할기 세부 로그 저장 완료: {log_path}")


if __name__ == "__main__":
    split_pixel_dataset_3way()