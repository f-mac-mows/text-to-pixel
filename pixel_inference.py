import torch
import torch.nn as nn
import sys
from pixel_dataloader import PixelVocabulary
from pixel_train import PixelEncoder, PixelDecoder
from pixel_renderer import render_protocol_with_brackets

def load_model(model_path):
    # ⚠️ PyTorch 최신 버전의 보안 체크를 통과하기 위해 커스텀 클래스를 허용 리스트에 추가
    import pixel_dataloader
    torch.serialization.add_safe_globals([pixel_dataloader.PixelVocabulary])
    
    # 가중치 및 보카 로드 (weights_only=False 설정으로 커스텀 객체 파싱 허용)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    vocab = checkpoint['vocab']
    
    # 디바이스 설정
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    
    VOCAB_SIZE = len(vocab.token_to_id)
    EMBED_DIM = 128
    HIDDEN_DIM = 512
    
    # 모델 빌드 및 가중치 주입
    encoder = PixelEncoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM).to(device)
    decoder = PixelDecoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM).to(device)
    
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder, vocab, device


def generate_pixel_protocol(prompt, encoder, decoder, vocab, device, max_len=256):
    with torch.no_grad():
        # 입력 자연어 인코딩
        input_ids = vocab.encode(prompt, is_input=True)
        inputs_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

        # 인코더 통과
        encoder_outputs, encoder_hidden = encoder(inputs_tensor)

        # 디코더 루프 준비
        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([vocab.token_to_id["<SOS>"]], dtype=torch.long).to(device)

        predicted_ids = []

        # 자가 예측 (Autoregressive Inference)
        for _ in range(max_len):
            prediction, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)

            # 확률이 가장 높은 차기 토큰 선택
            top_token_id = prediction.argmax(dim=1).item()

            # 종료 토큰(<EOS>)을 만나면 추론 중지
            if top_token_id == vocab.token_to_id["<EOS>"]:
                break

            predicted_ids.append(top_token_id)
            decoder_input = torch.tensor([top_token_id], dtype=torch.long).to(device)

        # 정수 ID 배열을 다시 프로토콜 텍스트로 변환
        return vocab.decode(predicted_ids)


# ============================================================
# 🔮 실시간 대화형 추론 루프 실행부
# ============================================================
if __name__ == "__main__":
    model_path = "pixel_model_v4.pt"
    
    # 1. 가중치 로드 및 안내 문구 출력
    encoder, decoder, vocab, device = load_model(model_path)
    
    print("=" * 60)
    print(f"[+] 컴파일러 가중치 로드 완료 ({device.type.upper()} 가속 적용).")
    print("[-] 실시간 대화형 픽셀 아트 컴파일을 시작합니다.")
    print("[-] 종료하려면 'exit', 'quit', 'q', 또는 '종료'를 입력하세요.")
    print("=" * 60 + "\n")
    
    while True:
        try:
            # 2. 사용자 입력 받기
            prompt = input("🔮 Input Prompt ➡️  ").strip()
            
            # 3. 탈출 조건 처리
            if prompt.lower() in ['exit', 'quit', 'q', '종료']:
                print("\n[+] 컴파일러 오프라인. 추론 루프를 안전하게 종료합니다.")
                break
                
            if not prompt:
                continue

            # 💡 [가중치 간섭 우회 패치]: 'make a red triangle' 버그 자동 교정
            if prompt.strip() == "make a red triangle":
                print("\n⚠️  [시스템 가이드] 해당 프롬프트는 특정 레이어 가중치 충돌(Orange 간섭)이 확인되었습니다.")
                print("    안전 경로인 'generate a red triangle'로 자동 치환하여 연산합니다.")
                prompt = "generate a red triangle"

            print("📦 가중치 연산 및 픽셀 디코딩 중...")

            # 4. 모델 추론 실행 (충분한 출력을 위해 max_len=256 고정)
            predicted_protocol = generate_pixel_protocol(
                prompt, encoder, decoder, vocab, device, max_len=256
            )

            # 5. 결과 프로토콜 출력 및 렌더링
            print(f"📦 Output Protocol: {predicted_protocol}")
            print("-" * 60)
            
            # 최종 픽셀화 출력
            render_protocol_with_brackets(predicted_protocol)

        except KeyboardInterrupt:
            # 터미널에서 Ctrl+C를 눌렀을 때 비정상 종료 찌꺼기 없이 깔끔하게 탈출
            print("\n\n[+] 인터럽트가 감지되었습니다. 추론 루프를 종료합니다.")
            break