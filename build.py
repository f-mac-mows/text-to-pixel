import os
import sys
import subprocess
from pixel_config import PixelPaths
from pixel_logger import setup_logger  # 💡 로거 추가

def check_and_install_dependencies(logger):
    """필수 라이브러리(torch, tokenizers, sklearn) 설치 확인 및 자동 설치"""
    logger.info("[*] ⚙️  [1/6] 개발 환경 내 필수 패키지 검사 중...")
    
    requirements = {
        "torch": "torch",
        "tokenizers": "tokenizers",
        "scikit-learn": "sklearn"
    }
    
    for package_name, import_name in requirements.items():
        try:
            __import__(import_name)
            logger.info(f" [✓] {package_name} 라이브러리가 준비되어 있습니다.")
        except ImportError:
            logger.warning(f" [⚠️] {package_name} 라이브러리가 없습니다. 자동 설치 중...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                logger.info(f" [✓] {package_name} 설치 완료.")
            except Exception as e:
                logger.error(f" [❌ 에러] {package_name} 설치 실패: {e}")
                sys.exit(1)
    logger.info("-" * 60)

def run_script(script_name, step_num, logger):
    """지정한 파이썬 스크립트를 파이프라인의 하위 프로세스로 실행"""
    logger.info(f"\n🎬 [{step_num}/6] {script_name} 가동 중...")
    
    # 💡 하위 프로세스(자식 스크립트)들이 자체 로거로 찍는 내부 print/logger 메시지들은
    # 그대로 부모 터미널 화면(stdout)에 실시간으로 흘러나오도록 둡니다.
    result = subprocess.run([sys.executable, script_name])
    
    if result.returncode != 0:
        logger.error(f"\n[❌ 중단] {script_name} 실행 중 에러 발생.")
        sys.exit(result.returncode)
        
    logger.info(f"[✓] {script_name} 연산 성공.")
    logger.info("-" * 60)

def main():
    # 💡 1. 마스터 파이프라인 전용 로거 가동
    logger, log_path = setup_logger(mode="pipeline")

    logger.info("=" * 60)
    logger.info("🛰️  PIXEL-ART CORE TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # 2. 환경 및 필수 패키지 검사 (로거 전달)
    check_and_install_dependencies(logger)
    
    # 3. 현재 설정 상태 출력 (앞서 수정한 대상을 로거와 함께 호출)
    PixelPaths.log_summary(logger)
    
    # 4. 파이프라인 단계별 실행 (각 단계의 시작과 끝을 마스터 로그에 박제)
    run_script("pixel_data_generator.py", "2", logger)
    run_script("train_tokenizer.py", "3", logger)
    run_script("split_dataset.py", "4", logger)
    run_script("pixel_train.py", "5", logger)
    run_script("evaluate_model.py", "6", logger)
    
    logger.info("\n🎉 [SUCCESS] 학습 파이프라인 완주!")
    logger.info(f"💡 이제 다음 명령어로 추론 모델을 실행하세요: python pixel_inference.py\n")
    logger.info(f"[✓] 전체 공정 마스터 로그 저장 완료: {log_path}")

if __name__ == "__main__":
    main()