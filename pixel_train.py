import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from itertools import zip_longest
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
        
        padded_inputs = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id)
        padded_targets = pad_sequence(target_ids, batch_first=True, padding_value=self.pad_id)
        
        return {
            "input_ids": padded_inputs,
            "target_ids": padded_targets,
            "task_types": task_types
        }


def split_by_task(dataset):
    """💡 [핵심 수정] GEN/DESC가 한 배치 안에 섞이면 모델의 배치 단위 태스크
    라우팅(is_text_input[0] 기준 판단)이 깨진다. 애초에 서로 다른 Subset으로
    분리해서, 배치 하나에는 항상 한 가지 태스크만 담기도록 강제한다."""
    gen_indices = [i for i, s in enumerate(dataset.samples) if s["input"].startswith("<TASK_GEN>")]
    desc_indices = [i for i, s in enumerate(dataset.samples) if s["input"].startswith("<TASK_DESC>")]
    return Subset(dataset, gen_indices), Subset(dataset, desc_indices)


def run_one_epoch(model, gen_loader, desc_loader, criterion, device, optimizer=None, pbar_desc=""):
    """옵티마이저가 주어지면 학습 모드, 없으면 평가 모드로 GEN/DESC 로더를
    번갈아가며(round-robin) 처리한다. 두 로더의 길이가 다를 수 있어
    zip_longest로 짧은 쪽이 먼저 끝나도 남은 쪽은 계속 처리한다."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(zip_longest(gen_loader, desc_loader), total=max(len(gen_loader), len(desc_loader)),
                desc=pbar_desc, leave=is_train)

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for gen_batch, desc_batch in pbar:
            for batch in (gen_batch, desc_batch):
                if batch is None:
                    continue

                inputs = batch["input_ids"].to(device)
                targets = batch["target_ids"].to(device)

                decoder_input = targets[:, :-1]
                decoder_target = targets[:, 1:]

                if is_train:
                    optimizer.zero_grad()

                predictions = model(inputs, decoder_input)
                current_vocab_dim = predictions.size(-1)
                loss = criterion(
                    predictions.reshape(-1, current_vocab_dim),
                    decoder_target.reshape(-1)
                )

                if is_train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(num_batches, 1)


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

    text_tokenizer = TextBpeTokenizerWrapper(vocab_size=5000, tokenizer_path=PixelPaths.TEXT_TOKENIZER)
    pixel_tokenizer = PixelWordTokenizerWrapper(tokenizer_path=PixelPaths.PIXEL_TOKENIZER)
    
    pixel_collate_fn = HybridPixelCollate(pad_id=0)
    
    train_dataset = HybridPixelArtDataset(PixelPaths.TRAIN_DATA, text_tokenizer, pixel_tokenizer)
    val_dataset = HybridPixelArtDataset(PixelPaths.VAL_DATA, text_tokenizer, pixel_tokenizer)

    # 💡 [핵심 수정] GEN/DESC 분리된 Subset + 개별 DataLoader
    train_gen_subset, train_desc_subset = split_by_task(train_dataset)
    val_gen_subset, val_desc_subset = split_by_task(val_dataset)

    logger.info(f" ├─ Train GEN 샘플: {len(train_gen_subset)}개 / DESC 샘플: {len(train_desc_subset)}개")
    logger.info(f" └─ Val   GEN 샘플: {len(val_gen_subset)}개 / DESC 샘플: {len(val_desc_subset)}개")

    train_gen_loader = DataLoader(train_gen_subset, batch_size=64, shuffle=True, collate_fn=pixel_collate_fn, num_workers=4)
    train_desc_loader = DataLoader(train_desc_subset, batch_size=64, shuffle=True, collate_fn=pixel_collate_fn, num_workers=4)
    val_gen_loader = DataLoader(val_gen_subset, batch_size=64, shuffle=False, collate_fn=pixel_collate_fn, num_workers=4)
    val_desc_loader = DataLoader(val_desc_subset, batch_size=64, shuffle=False, collate_fn=pixel_collate_fn, num_workers=4)

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

    from pixel_model import PixelSeq2SeqTransformer 
    model = PixelSeq2SeqTransformer(
        text_vocab_size=TEXT_VOCAB_SIZE,
        pixel_vocab_size=PIXEL_VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        nhead=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        pad_idx=0
    ).to(device)

    loss_weights = torch.ones(PIXEL_VOCAB_SIZE, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights, ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    logger.info(f"🚀 [v{PixelPaths.TOKENIZER_VER}] 2원화 사전을 장착한 양방향 통합 시퀀스 학습 가동...")
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        avg_train_loss = run_one_epoch(
            model, train_gen_loader, train_desc_loader, criterion, device,
            optimizer=optimizer, pbar_desc=f"🎬 Epoch {epoch:2d}/{EPOCHS} [Train]"
        )

        avg_val_loss = run_one_epoch(
            model, val_gen_loader, val_desc_loader, criterion, device,
            optimizer=None, pbar_desc=f"🧪 Epoch {epoch:2d}/{EPOCHS} [Val]"
        )

        scheduler.step()  

        log_msg = f"🌟 [Epoch {epoch:2d}/{EPOCHS}] Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
        logger.info(log_msg)

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

        # 💡 [실시간 검증 데이터셋 모니터링] - 배치 크기 1이라 라우팅 버그 영향 없음, 그대로 유지
        with torch.no_grad():
            import random

            indices_to_test = [0, random.randint(0, len(val_dataset) - 1)]
            indices_to_test = list(dict.fromkeys(indices_to_test)) 
            
            logger.info("\n" + "="*70)
            logger.info(f"🔮 [Epoch {epoch:2d} 실시간 멀티모달 결과 모니터링]")
            
            for idx in indices_to_test:
                raw_sample = val_dataset.samples[idx]
                is_gen_task = raw_sample["input"].startswith("<TASK_GEN>")
                
                active_out_tokenizer = pixel_tokenizer if is_gen_task else text_tokenizer

                test_sample = val_dataset[idx]
                test_input_ids = test_sample["input_ids"].unsqueeze(0).to(device)
                
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