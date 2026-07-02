import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pixel_dataloader import PixelVocabulary, PixelArtDataset, pixel_collate_fn

# ==========================================
# 1. 미니 루옹 어텐션 (Luong Attention) 레이어
# ==========================================
class LuongAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.wa = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [Batch, Hidden_dim] -> [Batch, 1, Hidden_dim]
        # encoder_outputs: [Batch, Seq_len, Hidden_dim]
        score = torch.bmm(encoder_outputs, self.wa(decoder_hidden).unsqueeze(2)) # [Batch, Seq_len, 1]
        attention_weights = torch.softmax(score, dim=1)

        # 문맥 벡터(Context Vector) 계산
        context = torch.bmm(encoder_outputs.transpose(1, 2), attention_weights).squeeze(2) # [Batch, Hidden_dim]
        return context, attention_weights
    

# ==========================================
# 2. Encoder-Decoder (Seq2Seq) 인공 신경망 정의
# ==========================================
class PixelEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embbeded = self.embedding(x)
        outputs, hidden = self.gru(embbeded)
        return outputs, hidden
    

class PixelDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = LuongAttention(hidden_dim)
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x, last_hidden, encoder_outputs):
        # x: [Batch] -> [Batch, 1]
        embedded = self.embedding(x.unsqueeze(1))

        # Attention 가동하여 인코더 문맥 가져오기
        context, weights = self.attention(last_hidden[-1], encoder_outputs)

        # 임베딩된 현재 입력과 어텐션 문맥 결합
        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        output, hidden = self.gru(rnn_input, last_hidden)

        # 가중치와 출력을 결합하여 최종 보카 사전 상의 단어 확률 예측
        output = torch.cat((output.squeeze(1), context), dim=1)
        prediction = self.fc(output)

        return prediction, hidden, weights
    

# ==========================================
# 3. 지도 학습 엔진 및 메인 실행부
# ==========================================
def main():
    # 하드웨어 장치 자동 지정 (MPS 가속 및 CUDA 대응)
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"[*] 활성화된 연산 가속 장치: {device}")

    dataset_file = "pixel_dataset_v3.jsonl"

    # 데이터 준비
    vocab = PixelVocabulary()
    vocab.build_vocab(dataset_file)
    pixel_dataset = PixelArtDataset(dataset_file, vocab)
    data_loader = DataLoader(pixel_dataset, batch_size=16, shuffle=True, collate_fn=pixel_collate_fn)

    # 하이퍼파라미터 세팅
    VOCAB_SIZE = len(vocab.token_to_id)
    EMBED_DIM = 128
    HIDDEN_DIM = 512
    LEARNING_RATE = 0.001
    EPOCHS = 55

    # 모델 선언 및 디바이스 업로드
    encoder = PixelEncoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM).to(device)
    decoder = PixelDecoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM).to(device)

    # 옵티마이저 및 손실함수 (패딩 토큰인 0은 계산에서 배제)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)

    print("[*] 픽셀 컴파일러 소형 신경망 지도 학습 가동...")

    encoder.train()
    decoder.train()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        for batch in data_loader:
            inputs = batch["input_ids"].to(device)
            targets = batch["target_ids"].to(device)

            optimizer.zero_grad()

            # 인코더 연산
            encoder_outputs, encoder_hidden = encoder(inputs)

            # 디코더 루프 준비
            decoder_hidden = encoder_hidden
            decoder_input = targets[:, 0] # 첫 단어는 언제나 <SOS> (ID: 1)

            loss = 0
            target_len = targets.size(1)

            # 디코더를 타겟 시퀀스 길이만큼 교사 강요(Teacher Forcing) 기반 학습
            for t in range(1, target_len):
                prediction, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(prediction, targets[:, t])
                decoder_input = targets[:, t] # 정답 라벨을 다음 입력으로 주입 (지도학습 구조)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() / target_len

        if epoch % 5 == 0 or epoch == 1:
            print(f" - Epoch {epoch:2d}/{EPOCHS} | Train Loss: {total_loss / len(data_loader):.7f}")

    # 가중치 영구 저장
    torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(), 'vocab': vocab}, "pixel_model_v4.pt")
    print("[+] 학습이 안전하게 종료되었으며 'pixel_model_v4.pt' 가중치가 생성되었습니다!")
    
if __name__ == "__main__":
    main()