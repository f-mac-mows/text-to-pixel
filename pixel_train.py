import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pixel_tokenizer import PixelArtTokenizerWrapper, PixelArtDataset, get_pixel_collate_fn
from pixel_config import PixelPaths
from pixel_logger import setup_logger
from pixel_model import PixelTransformer # 💡 주의: 이 모델이 Many-to-One 분류기(Classifier) 구조여야 합니다.


def main():
    # 디바이스 설정
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    
    logger, log_path = setup_logger(mode="train")
    logger.info("*" * 50)
    logger.info(f"활성화된 연산 가속 장치: {device}")

    if not os.path.exists(PixelPaths.TRAIN_DATA) or not os.path.exists(PixelPaths.VAL_DATA):
        print("[❌ 오류] 데이터 파일이 누락되었습니다.")
        return

    # 데이터셋 및 토크나이저 초기화
    # 💡 이전 답변의 'input_vocab_size' 프로퍼티 이름에 맞춰 수정하거나 기존 인스턴스 규칙 반영
    bpe_vocab = PixelArtTokenizerWrapper(vocab_size=2000, tokenizer_path=PixelPaths.TOKENIZER)
    pixel_collate_fn = get_pixel_collate_fn(bpe_vocab.pad_id)
    
    train_dataset = PixelArtDataset(PixelPaths.TRAIN_DATA, bpe_vocab)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=pixel_collate_fn)

    val_dataset = PixelArtDataset(PixelPaths.VAL_DATA, bpe_vocab)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=pixel_collate_fn)

    # 하이퍼파라미터 세팅
    # 💡 통합된 사전 크기를 그대로 사용합니다.
    VOCAB_SIZE = bpe_vocab.vocab_size 
    EMBED_DIM = 256     
    HIDDEN_DIM = 1024   
    NUM_HEADS = 8       
    NUM_LAYERS = 4      
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.05     
    EPOCHS = 60

    # 모델 선언
    model = PixelTransformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        pad_idx=bpe_vocab.pad_id
    ).to(device)

    # 💡 Many-to-One 분류 문제이므로 ignore_index 제거 (단일 클래스 타깃 비교)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    logger.info("🚀 Many-to-One 고유 아웃풋 토큰 분류 훈련을 시작합니다...")
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # --------------------------------------------------------
        # [STAGE 1] 모델 학습 (Train Mode)
        # --------------------------------------------------------
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            inputs = batch["input_ids"].to(device)
            # 💡 바뀐 데이터셋 구조에 맞춤 (1차원 벡터 텐서: [Batch_Size])
            targets = batch["target_ids"].to(device) 

            optimizer.zero_grad()

            # 💡 Shift 연산 전부 삭제. 인풋 자연어 시퀀스만 모델에 전달
            # 모델은 내부적으로 인코딩/풀링 후 최종적으로 [Batch_Size, VOCAB_SIZE] 로짓을 출력해야 함
            predictions = model(inputs) 

            # Loss 계산 (Predictions: [Batch, Vocab], Targets: [Batch])
            loss = criterion(predictions, targets)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --------------------------------------------------------
        # [STAGE 2] 모델 검증 (Validation Mode)
        # --------------------------------------------------------
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input_ids"].to(device)
                targets = batch["target_ids"].to(device)

                # 💡 검증도 학습과 동일하게 간소화
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        if epoch % 5 == 0 or epoch == 1:
            log_msg = f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {avg_train_loss:.7f} | Val Loss: {avg_val_loss:.7f}"
            logger.info(log_msg)

        # 최적 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_data = {
                'model_state_dict': model.state_dict(),
                'tokenizer_type': 'BPE_SPECIAL',
                'tokenizer_file': PixelPaths.TOKENIZER,
                'model_config': {
                    'vocab_size': VOCAB_SIZE, 
                    'embed_dim': EMBED_DIM, 
                    'hidden_dim': HIDDEN_DIM,
                    'num_heads': NUM_HEADS,
                    'num_layers': NUM_LAYERS
                },
                'epoch': epoch,
                'val_loss': best_val_loss
            }
            torch.save(save_data, PixelPaths.MODEL_CHECKPOINT)

    logger.info(f"[+] 🚀 모델 학습 완료! (Best Val Loss: {best_val_loss:.7f})")
    logger.info(f"[✓] 로그 파일이 기록되었습니다: {log_path}")

if __name__ == "__main__":
    PixelPaths.log_summary()
    main()