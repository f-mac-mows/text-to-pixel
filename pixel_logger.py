import os
import logging
from datetime import datetime

def setup_logger(mode="train"):
    """
    mode: 'train' 또는 'evaluate' 등 실행 모드 지정
    터미널 출력과 파일 저장을 동시에 수행하는 로거를 반환합니다.
    """
    # 1. 로그를 저장할 폴더 생성
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 2. 파일명 포맷 설정 (예: logs/20260703_0034_train.log)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{timestamp}_{mode}.log")

    # 3. 로거 객체 생성 및 기본 레벨 설정
    logger = logging.getLogger(mode)
    logger.setLevel(logging.INFO)

    # 중복 출력 방지
    if logger.handlers:
        logger.handlers.clear()

    # 4. 출력 포맷 설정 (시간 [로그레벨] 메시지)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # 5. 파일 핸들러 (파일에 기록)
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 6. 스트림 핸들러 (터미널에 출력)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger, log_filename