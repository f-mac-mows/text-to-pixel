import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import sys  
import re   
import torch
from pixel_config import PixelPaths
from pixel_inference import load_model, generate_pixel_protocol
from pixel_logger import setup_logger

# ========================================================
# 🎨 오답 시각화를 위한 초경량 디코더 & 이모지 맵 (유지)
# ========================================================
RENDER_MAP = {
    'R': '🟥', 'B': '🟦', 'G': '🟩', 'Y': '🟨', 'K': '⬛',
    'P': '🟪', 'O': '🟧', 'H': '💗', 'W': '⬜', 'A': '🪙',
    'N': '🟫', 'S': '🔷', 'L': '💚', 'M': '🧼', 'D': '👑',
    'V': '🥈', 'U': '🔵', 'E': '🌿', 'Z': '📜', 'J': '🔮'
}

def visual_decode(encoded_str: str) -> str:
    """ 압축 프로토콜을 복원하여 텍스트 이모지 바둑판 문자열로 반환합니다. """
    try:
        encoded_str = re.sub(r'\s+', ' ', encoded_str).strip()
        bracket_pattern = re.compile(r'\[\s*Re(\d+)\s*\]\s*\[\s*([^\]]+)\s*\]')
        
        while True:
            match = bracket_pattern.search(encoded_str)
            if not match:
                break
            count = int(match.group(1))
            content = match.group(2).strip()
            if content.endswith(" 1X"):
                content = content[:-3].strip()
            expanded = " | ".join([content] * count)
            encoded_str = encoded_str[:match.start()] + expanded + encoded_str[match.end():]

        raw_rows, current_row_tokens = [], []
        for token in encoded_str.split():
            if token in ["|", "1X"]:
                if current_row_tokens:
                    raw_rows.append(current_row_tokens)
                    current_row_tokens = []
            else:
                current_row_tokens.append(token)
        if current_row_tokens:
            raw_rows.append(current_row_tokens)

        lines = []
        for r_tokens in raw_rows:
            row_pixels = []
            for token in r_tokens:
                match = re.match(r'(\d+)([A-Z])', token)
                if match:
                    row_pixels.extend([match.group(2)] * int(match.group(1)))
                elif len(token) == 1 and token.isalpha():
                    row_pixels.append(token)
            
            if len(row_pixels) < 16:
                row_pixels.extend(['W'] * (16 - len(row_pixels)))
            lines.append("".join([RENDER_MAP.get(px, '❓') for px in row_pixels[:16]]))
        
        while len(lines) < 16:
            lines.append("⬜" * 16)
        return "\n".join(lines[:16])
    except Exception:
        return "⚠️ [디코딩 실패: 프로토콜 문법 붕괴 또는 UNK 토큰 검출]"


# ========================================================
# 📊 메인 평가 루프
# ========================================================
def run_test_evaluation():
    PixelPaths.log_summary()
    
    if not os.path.exists(PixelPaths.TEST_DATA):
        print(f"[❌ 오류] 테스트 데이터셋 '{PixelPaths.TEST_DATA}'이 존재하지 않습니다.")
        return
        
    logger, log_path = setup_logger(mode="evaluate")

    print("[*] 🤖 트랜스포머 전수 검증을 위해 모델 가중치를 로드합니다...")
    model, vocab, device = load_model(PixelPaths.MODEL_CHECKPOINT)
    
    test_samples = []
    with open(PixelPaths.TEST_DATA, "r", encoding="utf-8") as f:
        for line in f:
            test_samples.append(json.loads(line))
            
    total_count = len(test_samples)
    correct_count = 0
    
    total_wrong_count = 0  
    wrong_samples = []     
    
    print(f"[*] 📊 총 {total_count}개의 미공개 테스트 샘플 트랜스포머 분류 검사 가동...\n")
    
    for idx, sample in enumerate(test_samples, 1):
        prompt = sample["input"]
        ground_truth = sample["output"].strip()
        
        # 💡 시퀀스 생성이 아니라 단일 인덱스 추론 후 통문장 복원
        predicted_protocol = generate_pixel_protocol(
            prompt, model, vocab, device
        ).strip()
        
        # 💡 분류 문제 특성상 문자열이 완벽히 일치하는지만 대조하면 됩니다.
        is_correct = (predicted_protocol == ground_truth)
        if is_correct:
            correct_count += 1
        else:
            total_wrong_count += 1
            # 20번째 오답마다 샘플링
            if total_wrong_count % 20 == 1: 
                wrong_samples.append({
                    "idx": idx,
                    "prompt": prompt,
                    "ground_truth": ground_truth,
                    "predicted": predicted_protocol,
                    "wrong_nth": total_wrong_count  
                })
            
        if idx % 10 == 0 or idx == total_count:
            current_acc = (correct_count / idx) * 100
            print(f"\r 🔍 Processing: {idx:4d}/{total_count} | 현재 누적 정확도: {current_acc:.1f}%\t", end="")
            sys.stdout.flush() 
            
            if idx % 100 == 0 or idx == total_count:
                logger.info(f"Processing: {idx}/{total_count} | Accum. Acc: {current_acc:.1f}%")
            
    print() 
    
    final_accuracy = (correct_count / total_count) * 100
    
    # [오답 리포트 파일 저장 공정]
    error_log_path = log_path.replace("_evaluate.log", "_error_analysis.txt")
    os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
    
    with open(error_log_path, "w", encoding="utf-8") as ef:
        ef.write("============================================================\n")
        ef.write(f"❌ TRANSFORMER ERROR ANALYSIS REPORT (총 {total_wrong_count}개의 오답 중 {len(wrong_samples)}개 샘플링 완료)\n")
        ef.write(f"🎯 최종 정확도: {final_accuracy:.2f}%\n")
        ef.write("============================================================\n\n")
        
        for w in wrong_samples:
            ef.write(f"============================================================\n")
            ef.write(f"🚨 [CASE #{w['idx']}] (오답 일련번호: #{w['wrong_nth']}) Prompt: {w['prompt']}\n")
            ef.write(f"============================================================\n")
            ef.write(f"▶️ Ground Truth Protocol:\n{w['ground_truth']}\n\n")
            ef.write(f"▶️ Predicted Protocol:\n{w['predicted']}\n\n")
            ef.write(f"🖼️ [시각 보정 비교]\n")
            
            gt_render = visual_decode(w['ground_truth']).split('\n')
            pd_render = visual_decode(w['predicted']).split('\n')
            
            ef.write(f"    [ 정답 (Ground Truth) ]              [ 모델 예측 (Predicted) ]\n")
            for gt_line, pd_line in zip(gt_render, pd_render):
                ef.write(f"    {gt_line}    |    {pd_line}\n")
            ef.write("\n\n")

    logger.info("============================================================")
    logger.info("🏆 [최종 성적표] TRANSFORMER TEST EVALUATION REPORT")
    logger.info(f" ├─ 평가 대상 파일: {PixelPaths.TEST_DATA}")
    logger.info(f" ├─ 총 테스트 문제: {total_count} 개")
    logger.info(f" ├─ 완벽 복원 성공: {correct_count} 개")
    logger.info(f" ├─ ❌ 총 오답 수: {total_wrong_count} 개")
    logger.info(f" └─ 🎯 최종 정확도 (Accuracy): {final_accuracy:.2f}%")
    logger.info("============================================================")
    logger.info(f"[✓] 테스트 로그 저장 완료: {log_path}")
    logger.info(f"[🔥] 시각화 오답 노트(20포인트 샘플링) 저장 완료: {error_log_path}")

if __name__ == "__main__":
    run_test_evaluation()