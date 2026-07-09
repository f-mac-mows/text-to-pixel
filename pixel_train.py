import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# 💡 분리된 하이브리드 토크나이저 및 데이터셋 로드
from pixel_tokenizer import TextBpeTokenizerWrapper, PixelWordTokenizerWrapper, HybridPixelArtDataset
from pixel_config import PixelPaths
from pixel_logger import setup_logger
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence
class HybridPixelCollate:
    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        target_ids = [item["target_ids"] for item in batch]
        task_types = [item["task_type"] for item in batch]
        
        # 둘 다 0번이 패딩이므로 값 고민 없이 0으로 배치 정렬 진행
        padded_inputs = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
        padded_targets = pad_sequence(target_ids, batch_first=True, padding_value=self.pad_id)
        
        return {
            "input_ids": padded_inputs,
            "target_ids": padded_targets,
            "task_types": task_types  # 필요한 경우 학습 루프 검증용으로 전달
        }

def main():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    
    logger, log_path = setup_logger(mode="train")
    logger.info("*" * 50)
    logger.info(f"활성화된 연산 가속 장치: {device}")

    if not os.path.exists(PixelPaths.TRAIN_DATA) or not os.path.exists(PixelPaths.VAL_DATA):
        logger.error("[❌ 오류] 학습용 또는 검증용 데이터 파일 세트가 누락되었습니다.")
        return

    # 💡 2원화 토크나이저 인프라 각각 로드 및 복원
    text_tokenizer = TextBpeTokenizerWrapper(vocab_size=5000, tokenizer_path=PixelPaths.TEXT_TOKENIZER)
    pixel_tokenizer = PixelWordTokenizerWrapper(tokenizer_path=PixelPaths.PIXEL_TOKENIZER)
    
    # 두 사전의 패딩 ID를 추출
    pixel_collate_fn = HybridPixelCollate(pad_id=0)
    
    # 💡 리팩토링된 하이브리드 데이터셋 적용
    train_dataset = HybridPixelArtDataset(PixelPaths.TRAIN_DATA, text_tokenizer, pixel_tokenizer)
    val_dataset = HybridPixelArtDataset(PixelPaths.VAL_DATA, text_tokenizer, pixel_tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=pixel_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=pixel_collate_fn, num_workers=4)

    # 3. 모델 하이퍼파라미터 정의 (독립 Vocab 반영)
    TEXT_VOCAB_SIZE = text_tokenizer.base_tokenizer.get_vocab_size()
    PIXEL_VOCAB_SIZE = pixel_tokenizer.base_tokenizer.get_vocab_size()
    EMBED_DIM = 256
    HIDDEN_DIM = 1024   
    NUM_HEADS = 8    
    NUM_ENCODER_LAYERS = 4      
    NUM_DECODER_LAYERS = 6      
    LEARNING_RATE = 0.0002      
    WEIGHT_DECAY = 0.01     
    EPOCHS = 30

    # 4. 양방향 분리 임베딩 대응 Transformer 모델 인스턴스화
    # 💡 사전 크기를 2개 명시적으로 던져주도록 설계가 맞물려야 합니다.
    from pixel_model import PixelSeq2SeqTransformer 
    model = PixelSeq2SeqTransformer(
        text_vocab_size=TEXT_VOCAB_SIZE,
        pixel_vocab_size=PIXEL_VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        nhead=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        pad_idx=0  # 통합 0번 패딩 타깃 제어
    ).to(device)

    loss_weights = torch.ones(PIXEL_VOCAB_SIZE, dtype=torch.float32).to(device)
    
    # 💡 [교정] 손실 함수 계산 시 ignore_index와 weight를 동시에 지정
    # 가중치 텐서의 0번 인덱스(PAD)는 어차피 ignore_index에 의해 계산에서 제외되므로 안전합니다.
    criterion = nn.CrossEntropyLoss(weight=loss_weights, ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    logger.info(f"🚀 [v{PixelPaths.TOKENIZER_VER}] 2원화 사전을 장착한 양방향 통합 시퀀스 학습 가동...")
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # --------------------------------------------------------
        # [STAGE 1] 모델 학습 (Train Mode)
        # --------------------------------------------------------
        model.train()
        total_train_loss = 0

        train_pbar = tqdm(train_loader, desc=f"🎬 Epoch {epoch:2d}/{EPOCHS} [Train]", leave=True)
        for batch in train_pbar:
            inputs = batch["input_ids"].to(device)       
            targets = batch["target_ids"].to(device)     

            decoder_input = targets[:, :-1]
            decoder_target = targets[:, 1:]

            optimizer.zero_grad()

            # 💡 모델 내부적으로 입출력의 첫 토큰을 확인하거나 하위 플래그를 보고 
            # text_embedding / pixel_embedding을 가변 스위칭하며 연산하게 됩니다.
            predictions = model(inputs, decoder_input) 

            # 손실 계산 시 정답 타깃 레이어 크기(영단어 타깃이냐 픽셀 타깃이냐)에 대응하여 유연하게 매핑
            current_vocab_dim = predictions.size(-1)
            loss = criterion(
                predictions.reshape(-1, current_vocab_dim), 
                decoder_target.reshape(-1)
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # --------------------------------------------------------
        # [STAGE 2] 모델 검증 (Validation Mode)
        # --------------------------------------------------------
        model.eval()
        total_val_loss = 0
        
        val_pbar = tqdm(val_loader, desc=f"🧪 Epoch {epoch:2d}/{EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                inputs = batch["input_ids"].to(device)
                targets = batch["target_ids"].to(device)

                decoder_input = targets[:, :-1]
                decoder_target = targets[:, 1:]

                predictions = model(inputs, decoder_input)
                current_vocab_dim = predictions.size(-1)
                loss = criterion(
                    predictions.reshape(-1, current_vocab_dim), 
                    decoder_target.reshape(-1)
                )
                total_val_loss += loss.item()
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step()  

        log_msg = f"🌟 [Epoch {epoch:2d}/{EPOCHS}] Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
        logger.info(log_msg)

        # --------------------------------------------------------
        # 💾 [STAGE 3] 에포크별 체크포인트 저장 및 실시간 인퍼런스 테스트
        # --------------------------------------------------------
        epoch_checkpoint_path = f"models_history/pixel_model_v{PixelPaths.MODEL_VER}_epochs/e{epoch}.pt"
        os.makedirs(f"models_history/pixel_model_v{PixelPaths.MODEL_VER}_epochs/", exist_ok=True)
        
        current_save_data = {
            'model_state_dict': model.state_dict(),
            'tokenizer_type': 'Hybrid_Split_BPE_WordLevel',
            'text_tokenizer_file': PixelPaths.TEXT_TOKENIZER,
            'pixel_tokenizer_file': PixelPaths.PIXEL_TOKENIZER,
            'model_config': {
                'text_vocab_size': TEXT_VOCAB_SIZE,
                'pixel_vocab_size': PIXEL_VOCAB_SIZE,
                'embed_dim': EMBED_DIM, 
                'hidden_dim': HIDDEN_DIM,
                'num_heads': NUM_HEADS,
                'num_encoder_layers': NUM_ENCODER_LAYERS,
                'num_decoder_layers': NUM_DECODER_LAYERS
            },
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }
        torch.save(current_save_data, epoch_checkpoint_path)
        logger.info(f"[💾] Epoch {epoch:2d} 체크포인트 저장 완료 -> {epoch_checkpoint_path}")

        # 💡 [실시간 검증 데이터셋 모니터링 라우팅 조율 - 고정형 & 랜덤형 듀얼 모니터링]
        with torch.no_grad():
            import random
            
            # 1. 기존의 고정 데이터 (Convergence 확인용)
            # 2. 이번 에포크의 성능 검증을 위한 무작위 데이터 (Generalization 확인용)
            indices_to_test = [0, random.randint(0, len(val_dataset) - 1)]
            # 혹시 우연히 두 인덱스가 같으면 고정 데이터 하나만 출력하도록 유도
            indices_to_test = list(dict.fromkeys(indices_to_test)) 
            
            logger.info("\n" + "="*70)
            logger.info(f"🔮 [Epoch {epoch:2d} 실시간 멀티모달 결과 모니터링]")
            
            for idx in indices_to_test:
                raw_sample = val_dataset.samples[idx]
                is_gen_task = raw_sample["input"].startswith("<TASK_GEN>")
                
                active_out_tokenizer = pixel_tokenizer if is_gen_task else text_tokenizer
                active_in_tokenizer = text_tokenizer if is_gen_task else pixel_tokenizer

                test_sample = val_dataset[idx]
                test_input_ids = test_sample["input_ids"].unsqueeze(0).to(device)
                
                # Autoregressive 생성 시작
                generated_ids = [active_out_tokenizer.sos_id]
                max_generate_len = 128
                
                for _ in range(max_generate_len):
                    decoder_input = torch.tensor([generated_ids], dtype=torch.long).to(device)
                    outputs = model(test_input_ids, decoder_input)
                    
                    next_token_id = outputs[0, -1, :].argmax().item()
                    generated_ids.append(next_token_id)
                    
                    if next_token_id == active_out_tokenizer.eos_id:
                        break
                
                ground_truth = active_out_tokenizer.decode(test_sample["target_ids"])
                prediction = active_out_tokenizer.decode(generated_ids)
                
                label_type = "🎯 [고정 벤치마크]" if idx == 0 else "🎲 [랜덤 체크포인트]"
                logger.info(f" {label_type} (Val Dataset[{idx}])")
                logger.info(f"   ├─ 입력 프롬프트 : {raw_sample['input'][:60]}...")
                logger.info(f"   ├─ 기대하는 정답 : {ground_truth[:60]}...")
                logger.info(f"   └─ 모델의 예측값 : {prediction[:60]}...")
                logger.info("-" * 50)
            logger.info("="*70 + "\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(current_save_data, PixelPaths.MODEL_CHECKPOINT)
            logger.info(f"[👑] Best Val Loss 갱신! 최적 모델 세이브 -> {PixelPaths.MODEL_CHECKPOINT}")

    logger.info(f"[+] 🚀 멀티태스크 양방향 Seq2Seq 모델 학습 완료! (Best Val Loss: {best_val_loss:.5f})")
    logger.info(f"[✓] 학습 추적 세부 로그 파일 영사 완료: {log_path}")

if __name__ == "__main__":
    PixelPaths.log_summary()
    main()