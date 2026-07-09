import os
from enum import Enum
from dataclasses import dataclass

# 모델 8부터 트랜스포머 적용

class VersionConfig(Enum):
    TOTAL_DATASET = 11
    DATASET = 7  # 쪼개놓은 데이터셋(train, val, test)의 버전
    MODEL = 11    # 가중치(.pt)의 실험 버전
    TOKENIZER = 7

@dataclass(frozen=True)
class PixelPaths:
    TOTAL_DATASET_VER = VersionConfig.TOTAL_DATASET.value
    DATASET_VER = VersionConfig.DATASET.value
    MODEL_VER = VersionConfig.MODEL.value
    TOKENIZER_VER = VersionConfig.TOKENIZER.value

    TOTAL_DATA = f"pixel_dataset_v{TOTAL_DATASET_VER}.jsonl"
    TRAIN_DATA = f"pixel_train_v{DATASET_VER}.jsonl"
    VAL_DATA = f"pixel_val_v{DATASET_VER}.jsonl"
    TEST_DATA = f"pixel_test_v{DATASET_VER}.jsonl"

    TOKENIZER = f"pixel_bpe_tokenizer_v{TOKENIZER_VER}.json"
    MODEL_CHECKPOINT = f"pixel_model_v{MODEL_VER}.pt"

    @classmethod
    def log_summary(cls, logger=None):
        """현재 가동 중인 파이프라인 버전을 로거(또는 print)를 통해 기록"""
        # 로거가 넘어오면 logger.info를 쓰고, 없으면 그냥 print 함수를 매핑합니다.
        write = logger.info if logger else print

        write("=" * 60)
        write(f"🛰️  [PIXEL-ART ENGINE CONFIG] ACTIVE VERSION MANAGEMENT")
        write(f" ├─ Dataset Version  : v{cls.DATASET_VER}")
        write(f" ├─ BPE Version      : v{cls.TOKENIZER_VER}")
        write(f" └─ Model Version    : v{cls.MODEL_VER}")
        write("-" * 60)
        write(f" 📝 [Source Train]   : {cls.TRAIN_DATA}")
        write(f" 📝 [Source Val]     : {cls.VAL_DATA}")
        write(f" 📝 [Source Test]    : {cls.TEST_DATA}")
        write(f" 💾 [Target Weights] : {cls.MODEL_CHECKPOINT}")
        write("=" * 60 + "\n")