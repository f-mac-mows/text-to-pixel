import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import sys  
import re   
import torch
from torch.utils.data import DataLoader
from pixel_config import PixelPaths
from pixel_inference import load_model
from pixel_tokenizer import HybridPixelArtDataset # 💡 기존 데이터셋 재활용
from pixel_logger import setup_logger

# ========================================================
# 🎨 [정밀 보정] 256색 하이컬러 대응용 이모지 맵 & 디코더
# ========================================================
RENDER_MAP = {'000': '⬛', '255': '⬜', '028': '🟩', '248': '🟨'}

def visual_decode(encoded_str: str) -> str:
    try:
        encoded_str = re.sub(r'\s+', ' ', encoded_str).strip()
        tokens = encoded_str.split()
        raw_rows = []
        current_row_pixels = []
        
        for token in tokens:
            if token == "1X":
                if len(current_row_pixels) < 16:
                    current_row_pixels.extend(['000'] * (16 - len(current_row_pixels)))
                raw_rows.append(current_row_pixels[:16])
                current_row_pixels = []
                continue
            if token.startswith("p"):
                parts = token.split("_")
                if len(parts) < 2: continue
                try: count = int(parts[0][1:])
                except ValueError: continue
                colors = parts[1:]
                if len(colors) == 1:
                    current_row_pixels.extend([colors[0]] * count)
                elif len(colors) == 2:
                    current_row_pixels.extend(colors * count)
                    
        if current_row_pixels:
            if len(current_row_pixels) < 16:
                current_row_pixels.extend(['000'] * (16 - len(current_row_pixels)))
            raw_rows.append(current_row_pixels[:16])
        while len(raw_rows) < 16:
            raw_rows.append(['000'] * 16)
            
        return "\n".join(["".join([RENDER_MAP.get(px, '❓') for px in r_pixels]) for r_pixels in raw_rows[:16]])
    except Exception as e:
        return f"⚠️ [디코딩 실패: {str(e)}]"

# ========================================================
# 🚀 고속 배치 배치 추론 (Batch Autoregressive Decoding)
# ========================================================
def batch_generate(model, src_tensors, pixel_tok, device, max_len=128):
    """배치 단위로 토큰을 동시에 생성하여 속도를 극대화합니다."""
    model.eval()
    batch_size = src_tensors.size(0)
    
    # 각 배치의 시작을 [SOS] 토큰으로 채움
    tgt_ids = torch.full((batch_size, 1), pixel_tok.sos_id, dtype=torch.long, device=device)
    
    # 완성 여부 체크용 플래그
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        for _ in range(max_len):
            predictions = model(src_tensors, tgt_ids)
            next_tokens = predictions[:, -1, :].argmax(dim=-1)
            
            # EOS를 만났거나 이미 끝난 배치 가드
            next_tokens = torch.where(finished, torch.tensor(pixel_tok.pad_id, device=device), next_tokens)
            tgt_ids = torch.cat([tgt_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            finished |= (next_tokens == pixel_tok.eos_id)
            if finished.all():
                break
                
    return tgt_ids.cpu().tolist()

# ========================================================
# 📊 메인 평가 루프
# ========================================================
def run_test_evaluation():
    logger, log_path = setup_logger(mode="evaluate")
    
    logger.info("============================================================")
    logger.info(f"[*] ⚡ 고속 배치 생성형 검증 공정 가동:")
    logger.info(f" ├─ 테스트 데이터셋: {PixelPaths.TEST_DATA}")
    logger.info(f" └─ 체크포인트 로드: {PixelPaths.MODEL_CHECKPOINT}")
    logger.info("============================================================")
    
    if not os.path.exists(PixelPaths.TEST_DATA):
        logger.error(f"[❌ 오류] 테스트 데이터셋 파악 불가.")
        sys.exit(1)

    try:
        model, vocab, device = load_model(PixelPaths.MODEL_CHECKPOINT)
    except Exception as e:
        logger.error(f"[❌ 오류] 모델 로드 실패: {e}")
        sys.exit(1)
        
    text_tok = vocab["text"]
    pixel_tok = vocab["pixel"]
    
    # 💡 하이브리드 데이터셋 및 데이터로더 빌드 (패딩 가동)
    def collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        target_ids = [item["target_ids"] for item in batch]
        
        # 가변 길이를 패딩 토큰(0)으로 자동 정렬
        padded_inputs = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        
        # 정답지 원본 수집용
        raw_samples = batch
        return padded_inputs, raw_samples

    eval_dataset = HybridPixelArtDataset(PixelPaths.TEST_DATA, text_tok, pixel_tok)
    # 🎯 BATCH_SIZE를 64~128로 설정하여 속도를 끌어올립니다. Mac 사양에 맞게 조정 가능합니다.
    eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    
    total_count = len(eval_dataset)
    correct_count = 0
    total_wrong_count = 0
    wrong_samples = []
    
    logger.info(f"[*] 📊 총 {total_count}개의 미공개 테스트 샘플 고속 배치 검사 시동...\n")
    
    processed_idx = 0
    for src_tensors, raw_batch in eval_loader:
        src_tensors = src_tensors.to(device)
        
        # 고속 배치 생성
       # 고속 배치 생성 (이 부분은 유지)
        batch_output_ids = batch_generate(model, src_tensors, pixel_tok, device)
        
        for i, output_ids in enumerate(batch_output_ids):
            processed_idx += 1
            
            # 💡 [핵심 보정] [SOS] 제거 후, 첫 번째 [EOS]가 나타난 위치까지만 슬라이싱합니다.
            # output_ids[0]은 SOS이므로 output_ids[1:]부터 검사합니다.
            actual_ids = []
            for tid in output_ids[1:]:
                if tid == pixel_tok.eos_id:
                    break
                actual_ids.append(tid)
            
            # 💡 잘라낸 순수 아이디만 디코더에 전달하여 뒤쪽의 패딩/노이즈 토큰 오염을 원천 차단합니다.
            pred_str = pixel_tok.decode(actual_ids).strip()
            
            clean_pred = pred_str.replace("<SOS>", "").replace("<EOS>", "").replace("<PAD>", "").strip()
            clean_pred = re.sub(r'\s+', ' ', clean_pred)
            
            # 정답값 원본 복원
            idx_in_dataset = processed_idx - 1
            sample_data = eval_dataset.samples[idx_in_dataset]
            prompt = sample_data["input"].strip()
            ground_truth = sample_data["output"].strip()
            clean_gt = re.sub(r'\s+', ' ', ground_truth)
            
            is_correct = (clean_pred == clean_gt)
            if is_correct:
                correct_count += 1
            else:
                total_wrong_count += 1
                if total_wrong_count % 20 == 1: 
                    wrong_samples.append({
                        "idx": processed_idx,
                        "prompt": prompt,
                        "ground_truth": clean_gt,
                        "predicted": clean_pred if clean_pred else "[공백 출력]",
                        "wrong_nth": total_wrong_count
                    })
                    
        current_acc = (correct_count / processed_idx) * 100
        sys.stdout.write(f"\r 🔍 Processing: {processed_idx:4d}/{total_count} | 누적 시퀀스 일치율: {current_acc:.1f}%")
        sys.stdout.flush()
        
        if processed_idx % 640 == 0 or processed_idx == total_count:
            logger.info(f"Processing: {processed_idx}/{total_count} | Accum. Acc: {current_acc:.1f}%")
            
    print()
    final_accuracy = (correct_count / total_count) * 100
    
    # [오답 리포트 파일 저장 공정]
    error_log_path = log_path.replace("_evaluate.log", "_error_analysis.log")
    os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
    
    with open(error_log_path, "w", encoding="utf-8") as ef:
        ef.write("============================================================\n")
        ef.write(f"❌ TRANSFORMER GENERATION ERROR ANALYSIS REPORT (총 {total_wrong_count}개 중 오답 샘플링)\n")
        ef.write(f"🎯 최종 시퀀스 일치율(완벽 복원율): {final_accuracy:.2f}%\n")
        ef.write("============================================================\n\n")
        
        for w in wrong_samples:
            ef.write(f"============================================================\n")
            ef.write(f"🚨 [CASE #{w['idx']}] (오답 일련번호: #{w['wrong_nth']}) Prompt: {w['prompt']}\n")
            ef.write(f"============================================================\n")
            ef.write(f"▶️ Ground Truth: {w['ground_truth']}\n")
            ef.write(f"▶️ Predicted:    {w['predicted']}\n\n")
            ef.write(f"🖼️ [시각 보정 비교]\n")
            
            gt_render = visual_decode(w['ground_truth']).split('\n')
            pd_render = visual_decode(w['predicted']).split('\n')
            
            ef.write(f"    [ 정답 (Ground Truth) ]        [ 모델 예측 (Predicted) ]\n")
            ef.write(f"    -----------------------        -----------------------\n")
            for gt_line, pd_line in zip(gt_render, pd_render):
                ef.write(f"    {gt_line}    |    {pd_line}\n")
            ef.write(f"    -----------------------        -----------------------\n\n")

    logger.info("============================================================")
    logger.info("🏆 [최종 성적표] TRANSFORMER TEST EVALUATION REPORT")
    logger.info(f" ├─ 총 테스트 문제: {total_count} 개")
    logger.info(f" ├─ 완벽 시퀀스 일치: {correct_count} 개")
    logger.info(f" ├─ ❌ 문장 생성 불일치 수: {total_wrong_count} 개")
    logger.info(f" └─ 🎯 최종 완벽 복원율 (Accuracy): {final_accuracy:.2f}%")
    logger.info("============================================================")

if __name__ == "__main__":
    run_test_evaluation()