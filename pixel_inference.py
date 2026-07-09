import os
import torch
import json
from pixel_tokenizer import PixelArtTokenizerWrapper
# 💡 [수정] 리팩토링된 256색 전용 HighColorRenderer 임포트
from pixel_renderer import HighColorRenderer
from pixel_config import PixelPaths

# 트랜스포머 인코더 기반 모델 임포트
from pixel_model import PixelTransformer


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[❌ 오류] 모델 파일 '{model_path}'을 찾을 수 없습니다.")

    print(f"[*] 📂 트랜스포머 모델 스냅샷 로드 중: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    tokenizer_file = checkpoint.get('tokenizer_file', PixelPaths.TOKENIZER)
    vocab = PixelArtTokenizerWrapper(tokenizer_path=tokenizer_file)
    
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    
    config = checkpoint['model_config']
    
    model = PixelTransformer(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        pad_idx=vocab.pad_id
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab, device


def generate_pixel_protocol(prompt, model, vocab, device):
    """
    💡 Many-to-One 패러다임에 맞춰 단 한 번의 인코더 연산으로 고유 스페셜 토큰을 분류(Classify)합니다.
    """
    model.eval()
    with torch.no_grad():
        input_ids = vocab.encode_input(prompt)
        src_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        predictions = model(src_tensor)
        predicted_class_id = predictions[0].argmax(dim=-1).item()
        
        # 💡 [체크] 질문자님의 다중 클래스 분류 구조에서는 단일 ID를 
        # 통문장으로 바꾸는 기존의 decode_output(predicted_class_id) 방식을 그대로 유지합니다.
        return vocab.decode_output(predicted_class_id)
    

def run_error_analysis(test_file_path, model, vocab, device):
    wrong_cases = []
    
    print("\n" + "=" * 60)
    print(f"[*] 🔍 트랜스포머 에러 분석 스캔 시작: {test_file_path}")
    print("=" * 60)
    
    if not os.path.exists(test_file_path):
        print(f"[❌ 오류] 테스트 파일 '{test_file_path}'을 찾을 수 없습니다.")
        return

    with open(test_file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            prompt = data["input"]
            gold_protocol = data["output"]
            
            pred_protocol = generate_pixel_protocol(prompt, model, vocab, device)
            
            if pred_protocol.strip() != gold_protocol.strip():
                wrong_cases.append({
                    "id": idx + 1,
                    "prompt": prompt,
                    "gold": gold_protocol,
                    "pred": pred_protocol
                })

    analysis_file = "error_analysis_transformer.txt"
    with open(analysis_file, 'w', encoding='utf-8') as out:
        out.write(f"⚠️ 트랜스포머 실패 케이스: {len(wrong_cases)} 개\n")
        out.write("=" * 80 + "\n\n")
        
        for case in wrong_cases:
            out.write(f"🚩 [ID {case['id']}] Prompt: {case['prompt']}\n")
            out.write(f" ✅ 정답 (Gold): {case['gold']}\n")
            out.write(f" ❌ 예측 (Pred): {case['pred']}\n")
            out.write("-" * 80 + "\n")
            
    print(f"\n[+] 🏁 분석 완료! 결과가 '{analysis_file}'에 기록되었습니다.")


if __name__ == "__main__":
    try:
        model, vocab, device = load_model(PixelPaths.MODEL_CHECKPOINT)

        print("=" * 60)
        print(f"[+] 트랜스포머 픽셀 매퍼 가중치 로드 완료 ({device.type.upper()}).")
        print("[-] 실시간 대화형 픽셀 아트 분류 및 컴파일을 시작합니다.")
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

            print("📦 가중치 단일 연산 및 고속 고유 토큰 매핑 중...")

            predicted_protocol = generate_pixel_protocol(prompt, model, vocab, device)

            print(f"📦 Output Protocol: {predicted_protocol}")
            print("-" * 60)
            
            # 💡 [핵심 수정] 새롭게 리팩토링된 HighColorRenderer의 클래스 메서드 호출로 변경
            HighColorRenderer.render(predicted_protocol)
            print()

    except KeyboardInterrupt:
        print("\n\n[+] 인터럽트가 감지되었습니다. 시스템을 안전하게 종료합니다.")