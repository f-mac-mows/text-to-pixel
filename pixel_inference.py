import os
import torch
import json
import re
# 이전 단계에서 정렬한 이원화 토크나이저 클래스들 임포트
from pixel_tokenizer import TextBpeTokenizerWrapper, PixelWordTokenizerWrapper
from pixel_renderer import HighColorRenderer
from pixel_config import PixelPaths

# Seq2Seq 트랜스포머 모델 임포트
from pixel_model import PixelSeq2SeqTransformer


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[❌ 오류] 모델 파일 '{model_path}'을 찾을 수 없습니다.")

    print(f"[*] 📂 트랜스포머 멀티태스크 모델 스냅샷 로드 중: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    # [경로 정상화] 하드코딩된 파일명 대신 PixelPaths의 버전 아카이빙 공식 경로를 주입합니다.
    text_tokenizer = TextBpeTokenizerWrapper(tokenizer_path=PixelPaths.TEXT_TOKENIZER)
    pixel_tokenizer = PixelWordTokenizerWrapper(tokenizer_path=PixelPaths.PIXEL_TOKENIZER)
    
    # [자가 진단] 로드 직후 어휘집이 비어있는지 즉시 파악하는 안전 검증 장치 추가
    t_size = text_tokenizer.base_tokenizer.get_vocab_size()
    p_size = pixel_tokenizer.base_tokenizer.get_vocab_size()
    print(f"📊 [검증] 활성화된 사전 크기 -> Text Vocab: {t_size} | Pixel Vocab: {p_size}")
    
    if p_size <= 6:
        print(f"⚠️ [위험] 픽셀 토크나이저 어휘 크기가 {p_size}개뿐입니다. 파일을 정상적으로 읽지 못했을 수 있습니다!")
        print(f"   ㄴ 가리키는 경로: {PixelPaths.PIXEL_TOKENIZER}")
    
    vocab = {
        "text": text_tokenizer,
        "pixel": pixel_tokenizer
    }
    
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    
    config = checkpoint['model_config']
    
    t_vocab = int(config.get('text_vocab_size', config.get('vocab_size_src', 5000)))
    p_vocab = int(config.get('pixel_vocab_size', config.get('vocab_size_tgt', 3000)))
    emb_dim = int(config['embed_dim'])
    hid_dim = int(config['hidden_dim'])
    head_count = int(config.get('num_heads', config.get('nhead', 8))) 
    enc_layers = int(config['num_encoder_layers'])
    dec_layers = int(config.get('num_decoder_layers', 6))

    model = PixelSeq2SeqTransformer(
        text_vocab_size=t_vocab,  
        pixel_vocab_size=p_vocab,  
        embed_dim=emb_dim,
        hidden_dim=hid_dim,
        nhead=head_count,  
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        pad_idx=text_tokenizer.pad_id
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab, device


def generate_pixel_protocol(prompt, model, vocab, device, max_len=128):
    """
    프롬프트를 분석하여 적절한 태스크 태그(<TASK_GEN> 또는 <TASK_DESC>)를 동적으로 매핑 후 추론합니다.
    """
    model.eval()
    text_tok = vocab["text"]
    pixel_tok = vocab["pixel"]
    
    # 1. 원본 입력의 대소문자를 소문자로 통일하여 완벽한 키워드 검사 수행
    lower_prompt = prompt.lower().strip()
    task_token = "<TASK_GEN>"
    
    # 2. 설명/묘사 요청 키워드 및 프로토콜 고유 패턴 정밀 감지
    # (소문자화된 스트링과 비교하므로 키워드도 모두 소문자로 정렬)
    desc_keywords = ["describe", "explain", "what is this", "설명", "묘사", "해석", "분석", "p1_", "1x"]
    
    # 정규식을 활용해 p{숫자}_{색상} 구조가 프롬프트에 직접 포함되어 있는지 더 명확하게 체크
    has_protocol_pattern = bool(re.search(r'p\d+_\d+', lower_prompt))
    
    if any(kw in lower_prompt for kw in desc_keywords) or has_protocol_pattern:
        task_token = "<TASK_DESC>"
        
    # 3. 프롬프트 내에 이미 명시된 태그가 있다면 제거하여 순수 텍스트/프로토콜 확보
    pure_prompt = prompt.replace("<TASK_GEN>", "").replace("<TASK_DESC>", "").strip()
    
    # 4. 각 태스크의 고유 토큰 ID 획득
    task_id = text_tok.base_tokenizer.token_to_id(task_token)
    if task_id is None:
        task_id = 4 if task_token == "<TASK_GEN>" else 5
        
    print(f"🎯 [태스크 라우팅] 활성화된 모드: {task_token} (Token ID: {task_id})")
    
    # 5. 멀티태스크 통합 텍스트 시퀀스 인코딩 빌드
    pure_ids = text_tok.encode(pure_prompt, add_sos_eos=False)
    src_ids = [text_tok.sos_id, task_id] + pure_ids + [text_tok.eos_id]
    
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    
    # 6. 중요: [타겟 디코더 방향 전환] 
    # 생성(<TASK_GEN>)할 때는 픽셀 토큰(WordLevel)으로 출력하고,
    # 설명(<TASK_DESC>)할 때는 자연어 문장(BPE)으로 출력해야 함!
    current_decoder_tok = pixel_tok if task_token == "<TASK_GEN>" else text_tok
    
    tgt_ids = [current_decoder_tok.sos_id]
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            predictions = model(src_tensor, tgt_tensor)
            next_token_id = predictions[0, -1, :].argmax(dim=-1).item()
            
            if next_token_id == current_decoder_tok.eos_id:
                break
            generated_tokens.append(next_token_id)
            tgt_ids.append(next_token_id)
            
    if not generated_tokens:
        return ""

    # =============================================================
    # 🔥 [초강력 디버깅 소스] 토크나이저 필터링 없이 날것 그대로 매핑 출력
    # =============================================================
    print(f"📢 [RAW GEN IDs] {generated_tokens[:15]}... (총 {len(generated_tokens)}개)")
    
    raw_words = []
    base_tok = current_decoder_tok.base_tokenizer
    for t_id in generated_tokens:
        try:
            token_str = base_tok.id_to_token(t_id)
            if token_str is None:
                token_str = f"[None_ID:{t_id}]"
        except Exception:
            token_str = f"[ERR:{t_id}]"
        raw_words.append(token_str)
        
    print(f"📢 [RAW MAPPED WORDS] {' '.join(raw_words[:20])} ...")
    # =============================================================

    try:
        predicted_str = current_decoder_tok.decode(generated_tokens)
    except Exception:
        predicted_str = " ".join([w for w in raw_words if w not in ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]])

    return predicted_str.strip()


def run_error_analysis(test_file_path, model, vocab, device):
    wrong_cases = []
    
    print("\n" + "=" * 60)
    print(f"[*] 🔍 트랜스포머 멀티태스크 에러 분석 스캔 시작: {test_file_path}")
    print("=" * 60)
    
    if not os.path.exists(test_file_path):
        print(f"[❌ 오류] 테스트 파일 '{test_file_path}'을 찾을 수 없습니다.")
        return

    with open(test_file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            prompt = data["input"]
            gold_protocol = data["output"]
            
            # 입력값의 고유 본문 확보 및 태스크 유지 추론
            pred_protocol = generate_pixel_protocol(prompt, model, vocab, device)
            
            clean_pred = re.sub(r'\s+', ' ', pred_protocol).strip()
            clean_gold = re.sub(r'\s+', ' ', gold_protocol).strip()
            
            if clean_pred != clean_gold:
                wrong_cases.append({
                    "id": idx + 1,
                    "prompt": prompt,
                    "gold": clean_gold,
                    "pred": clean_pred
                })

    analysis_file = "error_analysis_transformer.txt"
    with open(analysis_file, 'w', encoding='utf-8') as out:
        out.write(f"⚠️ 트랜스포머 멀티태스크 실패 케이스: {len(wrong_cases)} 개\n")
        out.write("=" * 80 + "\n\n")
        
        for case in wrong_cases:
            out.write(f"🚩 [ID {case['id']}] Prompt: {case['prompt']}\n")
            out.write(f" ✅ 정답 (Gold): {case['gold']}\n")
            out.write(f" ❌ 예측 (Pred): {case['pred']}\n")
            out.write("-" * 80 + "\n")
            
    print(f"\n[+] 🏁 분석 완료! 결과가 '{analysis_file}'에 기록되었습니다.")


def run_inference_loop():
    history_protocols = []
    try:
        model, vocab, device = load_model(PixelPaths.MODEL_CHECKPOINT)

        print("=" * 60)
        print(f"[+] 트랜스포머 멀티태스크 픽셀 컴파일러 가중치 로드 완료 ({device.type.upper()}).")
        print("[-] 생성(<TASK_GEN>) 및 설명(<TASK_DESC>) 교차 추론 루프를 시작합니다.")
        print("[-] 종료하려면 'exit', 'quit', 'q', 또는 '종료'를 입력하세요.")
        print("=" * 60 + "\n")
        
        while True:
            prompt = input("🔮 Input Prompt ➡️  ").strip()
            
            if prompt.lower() in ['exit', 'quit', 'q', '종료']:
                print("\n[+] 컴파일러 오프라인. 추론 루프를 종료합니다.")
                break
                
            if not prompt:
                continue

            if prompt.strip() == "make a red triangle":
                prompt = "generate a red triangle"

            print("📦 가중치 레이어 타겟 연산 및 멀티태스크 스트림 디코딩 중...")

            predicted_result = generate_pixel_protocol(prompt, model, vocab, device)
            history_protocols.append(predicted_result)

            print(f"📦 Output Result: {predicted_result}")
            print("-" * 60)
            
            # 오직 생성 태스크 결과물이자 프로토콜 규격(p 숫자 조합)을 가진 텍스트일 때만 터미널 그리드 렌더링 가동
            if "p" in predicted_result and "1X" in predicted_result:
                print("🎨 [렌더러] 생성된 프로토콜을 기반으로 그래픽 출력을 시작합니다.")
                HighColorRenderer.render(predicted_result)
            else:
                print("📝 [텍스트] 설명 모드 출력이 반환되었습니다.")
                
            print()

    except KeyboardInterrupt:
        print("\n\n[+] 인터럽트가 감지되었습니다. 시스템을 안전하게 종료합니다.")
        
    return history_protocols


if __name__ == "__main__":
    results = run_inference_loop()