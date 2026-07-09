import os
import sys
import json
from pixel_config import PixelPaths
# 💡 2원화된 독립형 토크나이저 래퍼들로 교체 임포트
from pixel_tokenizer import TextBpeTokenizerWrapper, PixelWordTokenizerWrapper
from pixel_logger import setup_logger

def main():
    logger, log_path = setup_logger(mode="tokenizer_trainer")

    if not os.path.exists(PixelPaths.TOTAL_DATA):
        logger.error(f"[❌ 오류] '{PixelPaths.TOTAL_DATA}' 파일이 존재하지 않습니다.")
        sys.exit(1)

    logger.info(f"[*] 📂 2원화 분리형(BPE + WordLevel) 토크나이저 훈련 파이프라인 가동:")
    logger.info(f" ├─ 훈련용 말뭉치 소스: {PixelPaths.TOTAL_DATA}")
    logger.info(f" ├─ 저장될 텍스트 토크나이저 : {PixelPaths.TEXT_TOKENIZER}")
    logger.info(f" └─ 저장될 픽셀 토크나이저   : {PixelPaths.PIXEL_TOKENIZER}\n")

    # 1. 텍스트 토크나이저 빌드
    text_tokenizer = TextBpeTokenizerWrapper(vocab_size=5000, tokenizer_path=PixelPaths.TEXT_TOKENIZER)
    text_tokenizer.train_from_dataset(PixelPaths.TOTAL_DATA)
    logger.info(f" └─ [✓] 완료! 텍스트 사전 크기: {text_tokenizer.base_tokenizer.get_vocab_size()} 개")

    # 2. 픽셀 프로토콜 토크나이저 빌드
    pixel_tokenizer = PixelWordTokenizerWrapper(tokenizer_path=PixelPaths.PIXEL_TOKENIZER)
    pixel_tokenizer.train_from_dataset(PixelPaths.TOTAL_DATA)
    logger.info(f" └─ [✓] 완료! 픽셀 사전 크기: {pixel_tokenizer.base_tokenizer.get_vocab_size()} 개")

    # ----------------------------------------------------------------
    # 훈련 마무리 로그
    # ----------------------------------------------------------------
    logger.info(f"\n[+] 🚀 2원화 사전 구성 학습 및 영구 저장 완료!")
    logger.info(f"[✓] 토크나이저 빌더 세부 로그 저장 완료: {log_path}")

if __name__ == "__main__":
    main()