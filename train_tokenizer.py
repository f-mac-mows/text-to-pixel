import os
import sys
import json
from pixel_config import PixelPaths
from pixel_tokenizer import PixelArtTokenizerWrapper  # 새로 정리된 래퍼 임포트
# 💡 격리형 세부 로그 관리를 위한 모듈 임포트
from pixel_logger import setup_logger

def main():
    # 💡 토크나이저 훈련 전용 격리 로거 초기화 (logs/tokenizer_trainer/ 폴더 하위)
    logger, log_path = setup_logger(mode="tokenizer_trainer")

    if not os.path.exists(PixelPaths.TOTAL_DATA):
        logger.error(f"[❌ 오류] '{PixelPaths.TOTAL_DATA}' 파일이 존재하지 않습니다.")
        sys.exit(1)

    logger.info(f"[*] 📂 커스텀 스페셜 토큰 기반 토크나이저 훈련 파이프라인 가동:")
    logger.info(f" ├─ 훈련용 말뭉치 소스: {PixelPaths.TOTAL_DATA}")
    logger.info(f" └─ 저장될 토크나이저 : {PixelPaths.TOKENIZER}\n")

    # 1. 래퍼 초기화 (목표 Vocab Size 설정)
    tokenizer_wrapper = PixelArtTokenizerWrapper(vocab_size=2000, tokenizer_path=PixelPaths.TOKENIZER)

    # 2. 인풋 BPE 학습 + 아웃풋 통문장 스페셜 토큰 등록을 한 번에 처리
    # (내부에서 파일 읽기, set을 통한 중복 제거, BpeTrainer 주입이 모두 안전하게 일어납니다)
    tokenizer_wrapper.train_from_dataset(PixelPaths.TOTAL_DATA)

    # 3. 통합된 파일 구조 그대로 물리 저장
    tokenizer_wrapper.save()

    logger.info(f"\n[+] 🚀 통문장 스페셜 토큰 보호형 토크나이저 학습 및 저장 완료!")
    logger.info(f" └─ 최종 통합 사전 크기 (Input BPE + Output 스페셜 토큰): {tokenizer_wrapper.vocab_size} 개")
    logger.info(f"[✓] 토크나이저 빌더 세부 로그 저장 완료: {log_path}")

if __name__ == "__main__":
    main()