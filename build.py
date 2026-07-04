import os
import sys
import json
import subprocess
import importlib
from pixel_logger import setup_logger

VERSION_FILE = "pixel_versions.json"

def prepare_history_directories(logger):
    """스크린샷의 아카이빙 폴더들이 없으면 빌드 전 미리 자동 생성"""
    target_folders = ["dataset_history", "models_history", "tokenizer_history"]
    for folder in target_folders:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            logger.info(f" [✓] 디렉토리 신규 생성: {folder}")


def auto_increment_versions(logger):
    """빌드 시 버전을 자동으로 +1 하고 JSON에 세이브"""
    if not os.path.exists(VERSION_FILE):
        logger.warning(f"[⚠️] {VERSION_FILE}이 존재하지 않아 기본값으로 생성합니다.")
        import pixel_config
        importlib.reload(pixel_config)

    try:
        with open(VERSION_FILE, "r", encoding="utf-8") as f:
            versions = json.load(f)
        
        # 모든 활성 관리 버전 카운트 +1 상향
        for key in versions.keys():
            versions[key] += 1
            
        with open(VERSION_FILE, "w", encoding="utf-8") as f:
            json.dump(versions, f, indent=4)
            
        logger.info(f"[✓] 🔄 모든 버전 카운트 업 완료 (+1) -> {VERSION_FILE} 업데이트")
        
        # 부모 프로세스에 물려있는 캐시를 새로고침하여 리로드
        import pixel_config
        importlib.reload(pixel_config)
        
    except Exception as e:
        logger.error(f"[❌ 에러] 버전 자동 업데이트 중 오류 발생: {e}")
        sys.exit(1)


def check_and_install_dependencies(logger):
    """필수 라이브러리 설치 확인 및 자동 설치"""
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
    
    # 자식 스크립트들의 출력을 실시간으로 부모 터미널에 송출
    result = subprocess.run([sys.executable, script_name])
    
    if result.returncode != 0:
        logger.error(f"\n[❌ 중단] {script_name} 실행 중 에러 발생.")
        sys.exit(result.returncode)
        
    logger.info(f"[✓] {script_name} 연산 성공.")
    logger.info("-" * 60)


def main():
    # 💡 마스터 파이프라인 전용 로거 가동 -> 이제 logs/pipeline/ 폴더 밑으로 안착합니다.
    logger, log_path = setup_logger(mode="pipeline")

    logger.info("=" * 60)
    logger.info("🛰️  PIXEL-ART CORE TRAINING PIPELINE")
    logger.info("=" * 60)
    
    prepare_history_directories(logger)
    auto_increment_versions(logger)
    
    from pixel_config import PixelPaths

    check_and_install_dependencies(logger)
    PixelPaths.log_summary(logger)
    
    # 파이프라인 단계별 순차 프로세싱 실행
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