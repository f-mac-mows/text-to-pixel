import os
import json
from dataclasses import dataclass

VERSION_FILE = "pixel_versions.json"

def load_or_init_versions():
    """json 파일에서 버전을 읽어오거나, 없으면 초기값으로 생성"""
    default_versions = {
        "TOTAL_DATASET": 0,
        "DATASET": 0,
        "MODEL": 0,
        "TOKENIZER": 0
    }
    
    if os.path.exists(VERSION_FILE):
        try:
            with open(VERSION_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            print(f"⚠️ {VERSION_FILE} 읽기 실패. 기본값으로 초기화합니다.")
            
    # 파일이 없거나 깨졌을 경우 생성
    with open(VERSION_FILE, "w", encoding="utf-8") as f:
        json.dump(default_versions, f, indent=4)
    return default_versions


# 런타임에 실시간으로 JSON 버전을 로드
_v = load_or_init_versions()

@dataclass(frozen=True)
class PixelPaths:
    TOTAL_DATASET_VER = _v["TOTAL_DATASET"]
    DATASET_VER = _v["DATASET"]
    MODEL_VER = _v["MODEL"]
    TOKENIZER_VER = _v["TOKENIZER"]

    # 📂 아카이빙 폴더 정의
    DATA_DIR = "dataset_history"
    MODEL_DIR = "models_history"
    TOKEN_DIR = "tokenizer_history"

    # 🎯 데이터셋 경로 명세
    TOTAL_DATA = os.path.join(DATA_DIR, f"pixel_dataset_v{TOTAL_DATASET_VER}.jsonl")
    TRAIN_DATA = os.path.join(DATA_DIR, f"pixel_train_v{DATASET_VER}.jsonl")
    VAL_DATA   = os.path.join(DATA_DIR, f"pixel_val_v{DATASET_VER}.jsonl")
    TEST_DATA  = os.path.join(DATA_DIR, f"pixel_test_v{DATASET_VER}.jsonl")

    # 💡 [핵심 통합] 2원화 토크나이저 타겟 경로 공식 지정
    TEXT_TOKENIZER  = os.path.join(TOKEN_DIR, f"pixel_text_tokenizer_v{TOKENIZER_VER}.json")
    PIXEL_TOKENIZER = os.path.join(TOKEN_DIR, f"pixel_pixel_tokenizer_v{TOKENIZER_VER}.json")
    
    # 모델 가중치 체크포인트 경로
    MODEL_CHECKPOINT = os.path.join(MODEL_DIR, f"pixel_model_v{MODEL_VER}.pt")

    @classmethod
    def log_summary(cls, logger=None):
        """현재 가동 중인 파이프라인 버전을 로거(또는 print)를 통해 기록"""
        write = logger.info if logger else print

        write("=" * 60)
        write(f"🛰️  [PIXEL-ART ENGINE CONFIG] ACTIVE VERSION MANAGEMENT")
        write(f" ├─ Dataset Version  : v{cls.DATASET_VER}")
        write(f" ├─ Tokenizer Version: v{cls.TOKENIZER_VER}")
        write(f" └─ Model Version    : v{cls.MODEL_VER}")
        write("-" * 60)
        write(f" 📝 [Source Train]   : {cls.TRAIN_DATA}")
        write(f" 📝 [Source Val]     : {cls.VAL_DATA}")
        write(f" 📝 [Source Test]    : {cls.TEST_DATA}")
        # 💡 요약 창에서도 분리된 사전을 직관적으로 볼 수 있게 수정
        write(f" 🔑 [Text Tokenizer] : {cls.TEXT_TOKENIZER}")
        write(f" 🔑 [Pixel Tokenizer]: {cls.PIXEL_TOKENIZER}")
        write(f" 💾 [Target Weights] : {cls.MODEL_CHECKPOINT}")
        write("=" * 60 + "\n")