import os
import logging
from datetime import datetime

def setup_logger(mode="train"):
    """
    mode: 'pipeline', 'train', 'evaluate', 'inference' 등 실행 모드 지정
    각 모드별로 logs/<mode>/ 하위 폴더에 로그 파일을 격리 저장합니다.
    """
    # 1. 실행 모드별 세부 로그 폴더 분리 생성
    log_dir = os.path.join("logs", mode)
    os.makedirs(log_dir, exist_ok=True)

    # 2. 파일명 포맷 설정 (예: logs/pipeline/20260704_093000.log)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{timestamp}.log")

    # 4. 로거 객체 생성 및 기본 레벨 설정 (중복 호출 방지용 네임스페이스 격리)
    logger = logging.getLogger(f"pixel_{mode}_{timestamp}")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    # 4. 출력 포맷 설정
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 5. 파일 핸들러 (세부 격리 폴더에 기록)
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 6. 스트림 핸들러 (터미널 출력)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger, log_filename